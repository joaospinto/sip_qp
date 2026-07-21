#include "sip_qp/sip_qp.hpp"

#include "sip/sip.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace sip::qp {
namespace {

void add_compensated(const double term, double &sum, double &correction) {
  const double next = sum + term;
  correction += std::abs(sum) >= std::abs(term) ? (sum - next) + term
                                                : (term - next) + sum;
  sum = next;
}

class CompensatedSum {
public:
  void add(const double value) { add_compensated(value, sum_, correction_); }

  auto value() const -> double { return sum_ + correction_; }

private:
  double sum_ = 0.0;
  double correction_ = 0.0;
};

auto matrix_nnz(const sip_qdldl::ConstSparseMatrix &matrix) -> int {
  return matrix.indptr[matrix.cols];
}

auto positive_mean_relative_error_bound(const int num_terms) -> double {
  constexpr double unit_roundoff = std::numeric_limits<double>::epsilon() / 2.0;
  const double accumulated_roundoff =
      static_cast<double>(num_terms) * unit_roundoff;
  return accumulated_roundoff / (1.0 - accumulated_roundoff);
}

void copy_matrix(const sip_qdldl::ConstSparseMatrix &source,
                 sip_qdldl::SparseMatrix &destination) {
  destination.rows = source.rows;
  destination.cols = source.cols;
  destination.is_transposed = source.is_transposed;
  std::copy_n(source.indptr, source.cols + 1, destination.indptr);
  std::copy_n(source.ind, matrix_nnz(source), destination.ind);
  std::copy_n(source.data, matrix_nnz(source), destination.data);
}

void copy_model(const Data &data, Workspace &workspace) {
  auto &model = workspace.scaled_model;
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();

  model.f = data.objective_constant;
  std::copy_n(data.linear_objective, x_dim, workspace.scaled_linear_objective);
  std::copy_n(data.linear_objective, x_dim, model.gradient_f);
  copy_matrix(data.upper_hessian, model.upper_hessian_lagrangian);

  std::copy_n(data.equality_offsets, y_dim, model.c);
  copy_matrix(data.transposed_equality_jacobian, model.jacobian_c);

  std::copy_n(data.inequality_offsets, s_dim, model.g);
  copy_matrix(data.transposed_inequality_jacobian, model.jacobian_g);
}

void update_symmetric_norm(double *norms, const int lhs, const int rhs,
                           const double value) {
  const double magnitude = std::abs(value);
  norms[lhs] = std::max(norms[lhs], magnitude);
  norms[rhs] = std::max(norms[rhs], magnitude);
}

void compute_scaling(const Input &input, const Settings &settings,
                     Workspace &workspace) {
  const Data &data = input.data;
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();
  const int kkt_dim = x_dim + y_dim + s_dim;

  std::fill_n(workspace.primal_scaling, x_dim, 1.0);
  std::fill_n(workspace.equality_scaling, y_dim, 1.0);
  std::fill_n(workspace.inequality_scaling, s_dim, 1.0);
  workspace.objective_scaling = 1.0;

  for (int iteration = 0; iteration < settings.scaling.max_iterations;
       ++iteration) {
    std::fill_n(workspace.scaling_norms, kkt_dim, 0.0);

    const auto &hessian = data.upper_hessian;
    for (int column = 0; column < x_dim; ++column) {
      for (int index = hessian.indptr[column];
           index < hessian.indptr[column + 1]; ++index) {
        const int row = hessian.ind[index];
        update_symmetric_norm(workspace.scaling_norms, row, column,
                              hessian.data[index] *
                                  workspace.primal_scaling[row] *
                                  workspace.primal_scaling[column]);
      }
    }

    const auto &equality_jacobian = data.transposed_equality_jacobian;
    for (int equality = 0; equality < y_dim; ++equality) {
      for (int index = equality_jacobian.indptr[equality];
           index < equality_jacobian.indptr[equality + 1]; ++index) {
        const int variable = equality_jacobian.ind[index];
        update_symmetric_norm(
            workspace.scaling_norms, variable, x_dim + equality,
            equality_jacobian.data[index] * workspace.primal_scaling[variable] *
                workspace.equality_scaling[equality]);
      }
    }

    const auto &inequality_jacobian = data.transposed_inequality_jacobian;
    for (int inequality = 0; inequality < s_dim; ++inequality) {
      for (int index = inequality_jacobian.indptr[inequality];
           index < inequality_jacobian.indptr[inequality + 1]; ++index) {
        const int variable = inequality_jacobian.ind[index];
        update_symmetric_norm(workspace.scaling_norms, variable,
                              x_dim + y_dim + inequality,
                              inequality_jacobian.data[index] *
                                  workspace.primal_scaling[variable] *
                                  workspace.inequality_scaling[inequality]);
      }
    }

    for (int variable = 0; variable < x_dim; ++variable) {
      if (std::isfinite(data.lower_bounds[variable]) ||
          std::isfinite(data.upper_bounds[variable])) {
        workspace.scaling_norms[variable] =
            std::max(workspace.scaling_norms[variable], 1.0);
      }
    }

    double max_change = 0.0;
    for (int index = 0; index < kkt_dim; ++index) {
      double norm = workspace.scaling_norms[index];
      norm = norm < settings.scaling.min_norm
                 ? 1.0
                 : std::min(norm, settings.scaling.max_norm);
      const double change = 1.0 / std::sqrt(norm);
      workspace.scaling_norms[index] = change;
      max_change = std::max(max_change, std::abs(1.0 - change));
    }

    for (int variable = 0; variable < x_dim; ++variable) {
      workspace.primal_scaling[variable] *= workspace.scaling_norms[variable];
    }
    for (int equality = 0; equality < y_dim; ++equality) {
      workspace.equality_scaling[equality] *=
          workspace.scaling_norms[x_dim + equality];
    }
    for (int inequality = 0; inequality < s_dim; ++inequality) {
      workspace.inequality_scaling[inequality] *=
          workspace.scaling_norms[x_dim + y_dim + inequality];
    }

    if (max_change <= settings.scaling.convergence_tolerance) {
      break;
    }
  }

  if (!settings.scaling.scale_homogeneous_objective) {
    return;
  }

  double linear_objective_norm = 0.0;
  for (int variable = 0; variable < x_dim; ++variable) {
    linear_objective_norm = std::max(
        linear_objective_norm, std::abs(workspace.primal_scaling[variable] *
                                        data.linear_objective[variable]));
  }
  if (!std::isfinite(linear_objective_norm) || linear_objective_norm > 0.0) {
    return;
  }

  std::fill_n(workspace.scaling_norms, x_dim, 0.0);
  const auto &hessian = data.upper_hessian;
  for (int column = 0; column < x_dim; ++column) {
    for (int index = hessian.indptr[column]; index < hessian.indptr[column + 1];
         ++index) {
      const int row = hessian.ind[index];
      update_symmetric_norm(workspace.scaling_norms, row, column,
                            hessian.data[index] *
                                workspace.primal_scaling[row] *
                                workspace.primal_scaling[column]);
    }
  }

  double hessian_norm = 0.0;
  for (int variable = 0; variable < x_dim; ++variable) {
    hessian_norm +=
        workspace.scaling_norms[variable] / static_cast<double>(x_dim);
  }
  if (!std::isfinite(hessian_norm) || hessian_norm <= 0.0) {
    return;
  }

  double objective_norm = hessian_norm;
  objective_norm = objective_norm < settings.scaling.min_norm
                       ? 1.0
                       : std::min(objective_norm, settings.scaling.max_norm);
  if (std::abs(objective_norm - 1.0) <=
      positive_mean_relative_error_bound(x_dim)) {
    objective_norm = 1.0;
  }
  workspace.objective_scaling = 1.0 / objective_norm;
}

void apply_scaling(const Input &input, Workspace &workspace) {
  const Data &data = input.data;
  auto &model = workspace.scaled_model;
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();
  const double objective_scaling = workspace.objective_scaling;

  for (int variable = 0; variable < x_dim; ++variable) {
    const double scaling = workspace.primal_scaling[variable];
    workspace.variable_bound_scaling[variable] = 1.0 / scaling;
    workspace.dual_residual_scaling[variable] = objective_scaling * scaling;
    workspace.scaled_lower_bounds[variable] =
        data.lower_bounds[variable] / scaling;
    workspace.scaled_upper_bounds[variable] =
        data.upper_bounds[variable] / scaling;
    workspace.scaled_linear_objective[variable] *= objective_scaling * scaling;
  }

  model.f *= objective_scaling;
  auto &hessian = model.upper_hessian_lagrangian;
  for (int column = 0; column < x_dim; ++column) {
    for (int index = hessian.indptr[column]; index < hessian.indptr[column + 1];
         ++index) {
      hessian.data[index] *= objective_scaling *
                             workspace.primal_scaling[hessian.ind[index]] *
                             workspace.primal_scaling[column];
    }
  }

  auto &equality_jacobian = model.jacobian_c;
  for (int equality = 0; equality < y_dim; ++equality) {
    model.c[equality] *= workspace.equality_scaling[equality];
    for (int index = equality_jacobian.indptr[equality];
         index < equality_jacobian.indptr[equality + 1]; ++index) {
      equality_jacobian.data[index] *=
          workspace.equality_scaling[equality] *
          workspace.primal_scaling[equality_jacobian.ind[index]];
    }
  }

  auto &inequality_jacobian = model.jacobian_g;
  for (int inequality = 0; inequality < s_dim; ++inequality) {
    model.g[inequality] *= workspace.inequality_scaling[inequality];
    for (int index = inequality_jacobian.indptr[inequality];
         index < inequality_jacobian.indptr[inequality + 1]; ++index) {
      inequality_jacobian.data[index] *=
          workspace.inequality_scaling[inequality] *
          workspace.primal_scaling[inequality_jacobian.ind[index]];
    }
  }
}

void evaluate_affine_rows(const sip_qdldl::SparseMatrix &matrix,
                          const double *unscaled_offsets,
                          const double *row_scaling, const double *x,
                          double *values) {
  for (int row = 0; row < matrix.cols; ++row) {
    double value = row_scaling[row] * unscaled_offsets[row];
    double correction = 0.0;
    for (int index = matrix.indptr[row]; index < matrix.indptr[row + 1];
         ++index) {
      add_compensated(matrix.data[index] * x[matrix.ind[index]], value,
                      correction);
    }
    values[row] = value + correction;
  }
}

void evaluate_model(const Data &data, Workspace &workspace, const double *x) {
  auto &model = workspace.scaled_model;
  const int x_dim = data.num_primal_variables();

  std::copy_n(workspace.scaled_linear_objective, x_dim, model.gradient_f);
  sip_qdldl::add_Ax_to_y_where_A_upper_symmetric(model.upper_hessian_lagrangian,
                                                 x, model.gradient_f);

  double objective = workspace.objective_scaling * data.objective_constant;
  double correction = 0.0;
  for (int variable = 0; variable < x_dim; ++variable) {
    add_compensated(0.5 * x[variable] *
                        (workspace.scaled_linear_objective[variable] +
                         model.gradient_f[variable]),
                    objective, correction);
  }
  model.f = objective + correction;

  evaluate_affine_rows(model.jacobian_c, data.equality_offsets,
                       workspace.equality_scaling, x, model.c);
  evaluate_affine_rows(model.jacobian_g, data.inequality_offsets,
                       workspace.inequality_scaling, x, model.g);
}

struct GapMetrics {
  double absolute_gap;
  double scale;
};

struct ResidualMetrics {
  double max_primal_violation;
  double primal_scale;
  double max_stationarity_residual;
  double stationarity_scale;
  CompensatedSum hessian_gap;
};

void recover_solution(const Data &data, Workspace &workspace);

auto original_coordinate_gap(const Data &data, const Workspace &workspace,
                             const CompensatedSum &hessian_term) -> GapMetrics {
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();
  CompensatedSum linear_term;
  CompensatedSum equality_term;
  CompensatedSum inequality_term;
  CompensatedSum lower_bound_term;
  CompensatedSum upper_bound_term;
  CompensatedSum common_gap = hessian_term;
  const auto accumulate_common = [&](const double term,
                                     CompensatedSum &component) {
    component.add(term);
    common_gap.add(term);
  };

  for (int variable = 0; variable < x_dim; ++variable) {
    const double primal = workspace.primal_solution[variable];
    accumulate_common(data.linear_objective[variable] * primal, linear_term);
  }

  for (int index = 0; index < y_dim; ++index) {
    accumulate_common(-data.equality_offsets[index] *
                          workspace.equality_dual_solution[index],
                      equality_term);
  }

  for (int index = 0; index < s_dim; ++index) {
    accumulate_common(-data.inequality_offsets[index] *
                          workspace.inequality_dual_solution[index],
                      inequality_term);
  }

  CompensatedSum gap = common_gap;
  for (int variable = 0; variable < x_dim; ++variable) {
    const double dual = workspace.variable_bound_dual_solution[variable];
    if (dual < 0.0 && std::isfinite(data.lower_bounds[variable])) {
      const double term = data.lower_bounds[variable] * dual;
      lower_bound_term.add(term);
      gap.add(term);
    } else if (dual > 0.0 && std::isfinite(data.upper_bounds[variable])) {
      const double term = data.upper_bounds[variable] * dual;
      upper_bound_term.add(term);
      gap.add(term);
    }
  }

  return {
      .absolute_gap = std::abs(gap.value()),
      .scale = std::max(
          {1.0, std::abs(hessian_term.value()), std::abs(linear_term.value()),
           std::abs(equality_term.value()), std::abs(inequality_term.value()),
           std::abs(lower_bound_term.value()),
           std::abs(upper_bound_term.value())}),
  };
}

auto gap_satisfied(const GapMetrics &gap, const TerminationSettings &settings)
    -> bool {
  return std::isfinite(gap.absolute_gap) && std::isfinite(gap.scale) &&
         (gap.absolute_gap < settings.max_absolute_duality_gap ||
          gap.absolute_gap / gap.scale < settings.max_relative_duality_gap);
}

auto original_coordinate_residuals(const Data &data, Workspace &workspace)
    -> ResidualMetrics {
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();
  const double *primal = workspace.primal_solution;
  double max_primal_violation = 0.0;
  double primal_scale = 0.0;
  bool all_finite = true;
  CompensatedSum hessian_gap;

  const auto evaluate_affine_residual =
      [&](const sip_qdldl::ConstSparseMatrix &transposed_jacobian,
          const double *offsets, const int row, const bool equality) {
        double affine_term = 0.0;
        double affine_correction = 0.0;
        for (int index = transposed_jacobian.indptr[row];
             index < transposed_jacobian.indptr[row + 1]; ++index) {
          add_compensated(transposed_jacobian.data[index] *
                              primal[transposed_jacobian.ind[index]],
                          affine_term, affine_correction);
        }
        affine_term += affine_correction;

        double residual = offsets[row];
        double residual_correction = 0.0;
        add_compensated(affine_term, residual, residual_correction);
        residual += residual_correction;
        all_finite = all_finite && std::isfinite(affine_term) &&
                     std::isfinite(residual) && std::isfinite(offsets[row]);
        const double violation =
            equality ? std::abs(residual) : std::max(residual, 0.0);
        max_primal_violation = std::max(max_primal_violation, violation);
        primal_scale = std::max(
            {primal_scale, std::abs(affine_term), std::abs(offsets[row])});
      };

  for (int equality = 0; equality < y_dim; ++equality) {
    evaluate_affine_residual(data.transposed_equality_jacobian,
                             data.equality_offsets, equality, true);
  }
  for (int inequality = 0; inequality < s_dim; ++inequality) {
    evaluate_affine_residual(data.transposed_inequality_jacobian,
                             data.inequality_offsets, inequality, false);
  }

  for (int variable = 0; variable < x_dim; ++variable) {
    all_finite = all_finite && std::isfinite(primal[variable]);
    if (std::isfinite(data.lower_bounds[variable])) {
      max_primal_violation = std::max(
          max_primal_violation,
          std::max(data.lower_bounds[variable] - primal[variable], 0.0));
      primal_scale =
          std::max({primal_scale, std::abs(data.lower_bounds[variable]),
                    std::abs(primal[variable])});
    }
    if (std::isfinite(data.upper_bounds[variable])) {
      max_primal_violation = std::max(
          max_primal_violation,
          std::max(primal[variable] - data.upper_bounds[variable], 0.0));
      primal_scale =
          std::max({primal_scale, std::abs(data.upper_bounds[variable]),
                    std::abs(primal[variable])});
    }
  }

  std::fill_n(workspace.original_coordinate_hessian_product, x_dim, 0.0);
  std::fill_n(workspace.original_coordinate_sum_correction, x_dim, 0.0);
  const auto &upper_hessian = data.upper_hessian;
  for (int column = 0; column < x_dim; ++column) {
    for (int index = upper_hessian.indptr[column];
         index < upper_hessian.indptr[column + 1]; ++index) {
      const int row = upper_hessian.ind[index];
      const double value = upper_hessian.data[index];
      const double gap_term = value * primal[row] * primal[column];
      hessian_gap.add(gap_term);
      add_compensated(value * primal[column],
                      workspace.original_coordinate_hessian_product[row],
                      workspace.original_coordinate_sum_correction[row]);
      if (row != column) {
        hessian_gap.add(gap_term);
        add_compensated(value * primal[row],
                        workspace.original_coordinate_hessian_product[column],
                        workspace.original_coordinate_sum_correction[column]);
      }
    }
  }
  for (int variable = 0; variable < x_dim; ++variable) {
    workspace.original_coordinate_hessian_product[variable] +=
        workspace.original_coordinate_sum_correction[variable];
  }

  std::copy_n(workspace.variable_bound_dual_solution, x_dim,
              workspace.original_coordinate_stationarity_residual);
  std::fill_n(workspace.original_coordinate_sum_correction, x_dim, 0.0);
  const auto add_transposed_product =
      [&](const sip_qdldl::ConstSparseMatrix &transposed_jacobian,
          const double *dual, const int num_rows) {
        for (int row = 0; row < num_rows; ++row) {
          for (int index = transposed_jacobian.indptr[row];
               index < transposed_jacobian.indptr[row + 1]; ++index) {
            const int variable = transposed_jacobian.ind[index];
            add_compensated(
                transposed_jacobian.data[index] * dual[row],
                workspace.original_coordinate_stationarity_residual[variable],
                workspace.original_coordinate_sum_correction[variable]);
          }
        }
      };
  add_transposed_product(data.transposed_equality_jacobian,
                         workspace.equality_dual_solution, y_dim);
  add_transposed_product(data.transposed_inequality_jacobian,
                         workspace.inequality_dual_solution, s_dim);

  double max_stationarity_residual = 0.0;
  double stationarity_scale = 0.0;
  for (int variable = 0; variable < x_dim; ++variable) {
    double constraint_dual =
        workspace.original_coordinate_stationarity_residual[variable];
    constraint_dual += workspace.original_coordinate_sum_correction[variable];
    double residual = data.linear_objective[variable];
    double correction = 0.0;
    add_compensated(workspace.original_coordinate_hessian_product[variable],
                    residual, correction);
    add_compensated(constraint_dual, residual, correction);
    residual += correction;
    workspace.original_coordinate_stationarity_residual[variable] = residual;
    all_finite = all_finite && std::isfinite(constraint_dual) &&
                 std::isfinite(
                     workspace.original_coordinate_hessian_product[variable]) &&
                 std::isfinite(data.linear_objective[variable]) &&
                 std::isfinite(residual);
    max_stationarity_residual =
        std::max(max_stationarity_residual, std::abs(residual));
    stationarity_scale = std::max(
        {stationarity_scale, std::abs(data.linear_objective[variable]),
         std::abs(workspace.original_coordinate_hessian_product[variable]),
         std::abs(constraint_dual)});
  }

  if (!all_finite) {
    const double infinity = std::numeric_limits<double>::infinity();
    return {
        .max_primal_violation = infinity,
        .primal_scale = infinity,
        .max_stationarity_residual = infinity,
        .stationarity_scale = infinity,
        .hessian_gap = hessian_gap,
    };
  }
  return {
      .max_primal_violation = max_primal_violation,
      .primal_scale = std::max(1.0, primal_scale),
      .max_stationarity_residual = max_stationarity_residual,
      .stationarity_scale = std::max(1.0, stationarity_scale),
      .hessian_gap = hessian_gap,
  };
}

auto optimality_satisfied(const ::sip::TerminationCallbackInput &iterate,
                          const Settings &settings, const Data &data,
                          Workspace &workspace) -> bool {
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();

  double primal_scale = 0.0;
  for (int equality = 0; equality < y_dim; ++equality) {
    const double scaling = workspace.equality_scaling[equality];
    const double offset = data.equality_offsets[equality];
    const double affine_term =
        iterate.equality_residual[equality] / scaling - offset;
    primal_scale =
        std::max({primal_scale, std::abs(affine_term), std::abs(offset)});
  }
  for (int inequality = 0; inequality < s_dim; ++inequality) {
    const double scaling = workspace.inequality_scaling[inequality];
    const double offset = data.inequality_offsets[inequality];
    const double affine_term =
        (iterate.inequality_residual[inequality] - iterate.s[inequality]) /
            scaling -
        offset;
    const double slack = iterate.s[inequality] / scaling;
    primal_scale = std::max({primal_scale, std::abs(affine_term),
                             std::abs(offset), std::abs(slack)});
  }

  int side = 0;
  for (int variable = 0; variable < x_dim; ++variable) {
    const double primal =
        workspace.primal_scaling[variable] * iterate.x[variable];
    const double slack_scaling = workspace.variable_bound_scaling[variable];
    if (std::isfinite(data.lower_bounds[variable])) {
      const double slack = iterate.bound_s[side] / slack_scaling;
      primal_scale =
          std::max({primal_scale, std::abs(primal),
                    std::abs(data.lower_bounds[variable]), std::abs(slack)});
      ++side;
    }
    if (std::isfinite(data.upper_bounds[variable])) {
      const double slack = iterate.bound_s[side] / slack_scaling;
      primal_scale =
          std::max({primal_scale, std::abs(primal),
                    std::abs(data.upper_bounds[variable]), std::abs(slack)});
      ++side;
    }
  }

  double dual_scale = 0.0;
  for (int variable = 0; variable < x_dim; ++variable) {
    const double scaling = workspace.dual_residual_scaling[variable];
    const double linear_objective = workspace.scaled_linear_objective[variable];
    const double hessian_product =
        iterate.objective_gradient[variable] - linear_objective;
    const double constraint_dual =
        iterate.dual_residual[variable] - iterate.objective_gradient[variable];
    dual_scale = std::max({dual_scale, std::abs(hessian_product / scaling),
                           std::abs(linear_objective / scaling),
                           std::abs(constraint_dual / scaling)});
  }

  const double primal_residual_relative =
      iterate.max_primal_violation / std::max(1.0, primal_scale);
  const double dual_residual_relative =
      iterate.max_dual_violation / std::max(1.0, dual_scale);
  const bool primal_satisfied =
      iterate.max_primal_violation <
          settings.termination.max_absolute_residual ||
      primal_residual_relative < settings.termination.max_relative_residual;
  const bool dual_satisfied =
      iterate.max_dual_violation < settings.termination.max_absolute_residual ||
      dual_residual_relative < settings.termination.max_relative_residual;
  if (!primal_satisfied || !dual_satisfied) {
    return false;
  }

  recover_solution(data, workspace);
  const ResidualMetrics original_residuals =
      original_coordinate_residuals(data, workspace);
  const bool original_primal_satisfied =
      original_residuals.max_primal_violation <
          settings.termination.max_absolute_residual ||
      original_residuals.max_primal_violation /
              original_residuals.primal_scale <
          settings.termination.max_relative_residual;
  const bool original_stationarity_satisfied =
      original_residuals.max_stationarity_residual <
          settings.termination.max_absolute_residual ||
      original_residuals.max_stationarity_residual /
              original_residuals.stationarity_scale <
          settings.termination.max_relative_residual;
  if (!original_primal_satisfied || !original_stationarity_satisfied) {
    return false;
  }

  return gap_satisfied(
      original_coordinate_gap(data, workspace, original_residuals.hessian_gap),
      settings.termination);
}

void initialize_variables(const Input &input, const Settings &settings,
                          Workspace &workspace) {
  const Data &data = input.data;
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();
  auto &variables = workspace.sip.vars;

  for (int variable = 0; variable < x_dim; ++variable) {
    variables.x[variable] =
        input.initial_primal[variable] / workspace.primal_scaling[variable];
  }
  std::fill_n(variables.y, y_dim, 0.0);

  evaluate_model(data, workspace, variables.x);

  const double initial_mu =
      workspace.objective_scaling * settings.sip.barrier.initial_mu;
  const double slack_floor = std::max(settings.sip.barrier.initial_mu, 1e-8);
  for (int inequality = 0; inequality < s_dim; ++inequality) {
    variables.s[inequality] =
        std::max(-workspace.scaled_model.g[inequality], slack_floor);
    variables.z[inequality] = initial_mu / variables.s[inequality];
  }

  int side = 0;
  for (int variable = 0; variable < x_dim; ++variable) {
    if (std::isfinite(workspace.scaled_lower_bounds[variable])) {
      const double constraint_value =
          workspace.scaled_lower_bounds[variable] - variables.x[variable];
      variables.bound_s[side] = std::max(-constraint_value, slack_floor);
      variables.bound_z[side] = initial_mu / variables.bound_s[side];
      ++side;
    }
    if (std::isfinite(workspace.scaled_upper_bounds[variable])) {
      const double constraint_value =
          variables.x[variable] - workspace.scaled_upper_bounds[variable];
      variables.bound_s[side] = std::max(-constraint_value, slack_floor);
      variables.bound_z[side] = initial_mu / variables.bound_s[side];
      ++side;
    }
  }
}

void recover_solution(const Data &data, Workspace &workspace) {
  for (int variable = 0; variable < data.num_primal_variables(); ++variable) {
    workspace.primal_solution[variable] =
        workspace.primal_scaling[variable] * workspace.sip.vars.x[variable];
  }
  for (int equality = 0; equality < data.num_equalities(); ++equality) {
    workspace.equality_dual_solution[equality] =
        workspace.equality_scaling[equality] * workspace.sip.vars.y[equality];
    workspace.equality_dual_solution[equality] /= workspace.objective_scaling;
  }
  for (int inequality = 0; inequality < data.num_inequalities(); ++inequality) {
    workspace.inequality_dual_solution[inequality] =
        workspace.inequality_scaling[inequality] *
        workspace.sip.vars.z[inequality];
    workspace.inequality_dual_solution[inequality] /=
        workspace.objective_scaling;
  }

  int side = 0;
  for (int variable = 0; variable < data.num_primal_variables(); ++variable) {
    double dual = 0.0;
    double dual_correction = 0.0;
    if (std::isfinite(workspace.scaled_lower_bounds[variable])) {
      const double side_dual = workspace.variable_bound_scaling[variable] *
                               workspace.sip.vars.bound_z[side] /
                               workspace.objective_scaling;
      add_compensated(-side_dual, dual, dual_correction);
      ++side;
    }
    if (std::isfinite(workspace.scaled_upper_bounds[variable])) {
      const double side_dual = workspace.variable_bound_scaling[variable] *
                               workspace.sip.vars.bound_z[side] /
                               workspace.objective_scaling;
      add_compensated(side_dual, dual, dual_correction);
      ++side;
    }
    workspace.variable_bound_dual_solution[variable] = dual + dual_correction;
  }
}

} // namespace

auto solve(const Input &input, const Settings &settings, Workspace &workspace)
    -> Output {
  const Data &data = input.data;
  copy_model(data, workspace);
  compute_scaling(input, settings, workspace);
  apply_scaling(input, workspace);
  initialize_variables(input, settings, workspace);

  const sip_qdldl::Settings qdldl_settings{
      .permute_kkt_system = true,
      .kkt_pinv = input.kkt_ordering.inverse_permutation,
  };
  sip_qdldl::CallbackProvider callback_provider(
      qdldl_settings, workspace.scaled_model, workspace.qdldl);

  const auto factor = [&callback_provider](const double *w, const double *r1,
                                           const double *r2,
                                           const double *r3) -> bool {
    return callback_provider.factor(w, r1, r2, r3);
  };
  const auto linear_solve = [&callback_provider](const double *rhs,
                                                 double *solution) -> void {
    callback_provider.solve(rhs, solution);
  };
  const auto add_kkt_product =
      [&callback_provider](const double *w, const double *r1, const double *r2,
                           const double *r3, const double *x_x,
                           const double *x_y, const double *x_z, double *y_x,
                           double *y_y, double *y_z) -> void {
    callback_provider.add_Kx_to_y(w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y, y_z);
  };
  const auto add_hessian_product = [&callback_provider](const double *x,
                                                        double *y) -> void {
    callback_provider.add_Hx_to_y(x, y);
  };
  const auto add_equality_product = [&callback_provider](const double *x,
                                                         double *y) -> void {
    callback_provider.add_Cx_to_y(x, y);
  };
  const auto add_transposed_equality_product =
      [&callback_provider](const double *x, double *y) -> void {
    callback_provider.add_CTx_to_y(x, y);
  };
  const auto add_inequality_product = [&callback_provider](const double *x,
                                                           double *y) -> void {
    callback_provider.add_Gx_to_y(x, y);
  };
  const auto add_transposed_inequality_product =
      [&callback_provider](const double *x, double *y) -> void {
    callback_provider.add_GTx_to_y(x, y);
  };
  const auto get_objective = [&workspace]() -> double {
    return workspace.scaled_model.f;
  };
  const auto get_gradient = [&workspace]() -> const double * {
    return workspace.scaled_model.gradient_f;
  };
  const auto get_equalities = [&workspace]() -> const double * {
    return workspace.scaled_model.c;
  };
  const auto get_inequalities = [&workspace]() -> const double * {
    return workspace.scaled_model.g;
  };
  const auto model_callback =
      [&data, &workspace](const ::sip::ModelCallbackInput &callback) {
        if (callback.new_x) {
          evaluate_model(data, workspace, callback.x);
        }
      };
  const auto termination_callback =
      [&data, &settings,
       &workspace](const ::sip::TerminationCallbackInput &iterate) {
        return optimality_satisfied(iterate, settings, data, workspace);
      };

  const ::sip::Input sip_input{
      .factor = std::cref(factor),
      .solve = std::cref(linear_solve),
      .add_Kx_to_y = std::cref(add_kkt_product),
      .add_Hx_to_y = std::cref(add_hessian_product),
      .add_Cx_to_y = std::cref(add_equality_product),
      .add_CTx_to_y = std::cref(add_transposed_equality_product),
      .add_Gx_to_y = std::cref(add_inequality_product),
      .add_GTx_to_y = std::cref(add_transposed_inequality_product),
      .get_f = std::cref(get_objective),
      .get_grad_f = std::cref(get_gradient),
      .get_c = std::cref(get_equalities),
      .get_g = std::cref(get_inequalities),
      .model_callback = std::cref(model_callback),
      .termination_callback = std::cref(termination_callback),
      .timeout_callback = std::cref(input.timeout_callback),
      .lower_bounds = workspace.scaled_lower_bounds,
      .upper_bounds = workspace.scaled_upper_bounds,
      .residual_scaling =
          {
              .dual = workspace.dual_residual_scaling,
              .equality = workspace.equality_scaling,
              .inequality = workspace.inequality_scaling,
              .variable_bound = workspace.variable_bound_scaling,
          },
      .dimensions =
          {
              .x_dim = data.num_primal_variables(),
              .s_dim = data.num_inequalities(),
              .y_dim = data.num_equalities(),
          },
  };

  ::sip::Settings sip_settings = settings.sip;
  sip_settings.barrier.initial_mu *= workspace.objective_scaling;
  sip_settings.penalty.initial_penalty_parameter *= workspace.objective_scaling;
  sip_settings.penalty.min_penalty_barrier_product *=
      workspace.objective_scaling * workspace.objective_scaling;
  sip_settings.penalty.max_penalty_parameter *= workspace.objective_scaling;
  sip_settings.line_search.min_merit_slope_to_skip_line_search *=
      workspace.objective_scaling;
  sip_settings.termination.max_constraint_violation =
      settings.termination.max_absolute_residual;
  sip_settings.termination.max_dual_residual =
      settings.termination.max_absolute_residual;
  sip_settings.termination.max_complementarity_gap =
      workspace.objective_scaling *
      std::min(sip_settings.termination.max_complementarity_gap,
               settings.termination.max_absolute_duality_gap);
  sip_settings.barrier.mu_min *= workspace.objective_scaling;
  sip_settings.termination.max_merit_slope *= workspace.objective_scaling;
  const ::sip::Output sip_output =
      ::sip::solve(sip_input, sip_settings, workspace.sip);
  recover_solution(data, workspace);

  return Output{
      .sip = sip_output,
      .primal = workspace.primal_solution,
      .equality_dual = workspace.equality_dual_solution,
      .inequality_dual = workspace.inequality_dual_solution,
      .variable_bound_dual = workspace.variable_bound_dual_solution,
  };
}

} // namespace sip::qp
