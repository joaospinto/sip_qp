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

auto matrix_nnz(const sip_qdldl::ConstSparseMatrix &matrix) -> int {
  return matrix.indptr[matrix.cols];
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
}

void apply_scaling(const Input &input, Workspace &workspace) {
  const Data &data = input.data;
  auto &model = workspace.scaled_model;
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();

  for (int variable = 0; variable < x_dim; ++variable) {
    const double scaling = workspace.primal_scaling[variable];
    workspace.variable_bound_scaling[variable] = 1.0 / scaling;
    workspace.scaled_lower_bounds[variable] =
        data.lower_bounds[variable] / scaling;
    workspace.scaled_upper_bounds[variable] =
        data.upper_bounds[variable] / scaling;
    workspace.scaled_linear_objective[variable] *= scaling;
  }

  auto &hessian = model.upper_hessian_lagrangian;
  for (int column = 0; column < x_dim; ++column) {
    for (int index = hessian.indptr[column]; index < hessian.indptr[column + 1];
         ++index) {
      hessian.data[index] *= workspace.primal_scaling[hessian.ind[index]] *
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

  double objective = data.objective_constant;
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

auto initialize_variables(const Input &input, const Settings &settings,
                          Workspace &workspace,
                          sip_qdldl::CallbackProvider &callback_provider)
    -> bool {
  const Data &data = input.data;
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();
  auto &variables = workspace.sip.vars;

  for (int variable = 0; variable < x_dim; ++variable) {
    variables.x[variable] =
        input.initial_primal[variable] / workspace.primal_scaling[variable];
  }
  evaluate_model(data, workspace, variables.x);

  auto &linear_system = workspace.sip.csd_workspace;
  double *w = linear_system.w;
  double *r1 = linear_system.r1;
  double *r2 = linear_system.r2;
  double *r3 = linear_system.r3;
  double *rhs = linear_system.rhs_block_3x3;
  double *solution = linear_system.sol_block_3x3;
  double *rhs_x = rhs;
  double *rhs_y = rhs_x + x_dim;
  double *rhs_z = rhs_y + y_dim;

  const double dual_regularization =
      1.0 / settings.sip.penalty.initial_penalty_parameter;
  std::fill_n(w, s_dim, 1.0);
  std::fill_n(r1, x_dim, 0.0);
  std::fill_n(r2, y_dim, dual_regularization);
  std::fill_n(r3, s_dim, dual_regularization);

  std::fill_n(rhs_x, x_dim, 0.0);
  callback_provider.add_Hx_to_y(variables.x, rhs_x);
  for (int variable = 0; variable < x_dim; ++variable) {
    rhs_x[variable] -= workspace.scaled_model.gradient_f[variable];
  }

  std::fill_n(rhs_y, y_dim, 0.0);
  callback_provider.add_Cx_to_y(variables.x, rhs_y);
  for (int equality = 0; equality < y_dim; ++equality) {
    rhs_y[equality] -= workspace.scaled_model.c[equality];
  }

  std::fill_n(rhs_z, s_dim, 0.0);
  callback_provider.add_Gx_to_y(variables.x, rhs_z);
  for (int inequality = 0; inequality < s_dim; ++inequality) {
    rhs_z[inequality] -= workspace.scaled_model.g[inequality];
  }

  int side = 0;
  for (int variable = 0; variable < x_dim; ++variable) {
    if (std::isfinite(workspace.scaled_lower_bounds[variable])) {
      constexpr double jacobian = -1.0;
      const double bound_rhs =
          jacobian * workspace.scaled_lower_bounds[variable];
      r1[variable] += 1.0 / (1.0 + dual_regularization);
      rhs_x[variable] += jacobian * bound_rhs / (1.0 + dual_regularization);
      workspace.sip.nrhs.bound_z[side] = bound_rhs;
      ++side;
    }
    if (std::isfinite(workspace.scaled_upper_bounds[variable])) {
      constexpr double jacobian = 1.0;
      const double bound_rhs =
          jacobian * workspace.scaled_upper_bounds[variable];
      r1[variable] += 1.0 / (1.0 + dual_regularization);
      rhs_x[variable] += jacobian * bound_rhs / (1.0 + dual_regularization);
      workspace.sip.nrhs.bound_z[side] = bound_rhs;
      ++side;
    }
  }

  double primal_regularization =
      std::max(settings.sip.regularization.initial,
               settings.sip.regularization.first_positive);
  bool factorization_ok = false;
  for (int attempt = 0; attempt < settings.sip.regularization.max_attempts;
       ++attempt) {
    for (int variable = 0; variable < x_dim; ++variable) {
      r1[variable] += primal_regularization;
      rhs_x[variable] += primal_regularization * variables.x[variable];
    }
    factorization_ok = callback_provider.factor(w, r1, r2, r3);
    for (int variable = 0; variable < x_dim; ++variable) {
      r1[variable] -= primal_regularization;
      rhs_x[variable] -= primal_regularization * variables.x[variable];
    }
    if (factorization_ok) {
      break;
    }
    const double next_regularization =
        primal_regularization < settings.sip.regularization.first_positive
            ? settings.sip.regularization.first_positive
            : std::min(primal_regularization *
                           settings.sip.regularization.increase_factor,
                       settings.sip.regularization.maximum);
    if (next_regularization <= primal_regularization) {
      break;
    }
    primal_regularization = next_regularization;
  }
  if (!factorization_ok) {
    return false;
  }

  for (int variable = 0; variable < x_dim; ++variable) {
    rhs_x[variable] += primal_regularization * variables.x[variable];
  }
  callback_provider.solve(rhs, solution);
  std::copy_n(solution, x_dim, variables.x);
  std::copy_n(solution + x_dim, y_dim, variables.y);
  std::copy_n(solution + x_dim + y_dim, s_dim, variables.z);

  side = 0;
  for (int variable = 0; variable < x_dim; ++variable) {
    if (std::isfinite(workspace.scaled_lower_bounds[variable])) {
      constexpr double jacobian = -1.0;
      variables.bound_z[side] = (jacobian * variables.x[variable] -
                                 workspace.sip.nrhs.bound_z[side]) /
                                (1.0 + dual_regularization);
      ++side;
    }
    if (std::isfinite(workspace.scaled_upper_bounds[variable])) {
      constexpr double jacobian = 1.0;
      variables.bound_z[side] = (jacobian * variables.x[variable] -
                                 workspace.sip.nrhs.bound_z[side]) /
                                (1.0 + dual_regularization);
      ++side;
    }
  }

  const int num_bound_sides = side;
  double slack_shift = 0.0;
  double dual_shift = 0.0;
  const auto update_shifts = [&](double *slack, const double *dual,
                                 const int size) {
    for (int index = 0; index < size; ++index) {
      slack[index] = -dual[index];
      slack_shift = std::max(slack_shift, -slack[index]);
      dual_shift = std::max(dual_shift, -dual[index]);
    }
  };
  update_shifts(variables.s, variables.z, s_dim);
  update_shifts(variables.bound_s, variables.bound_z, num_bound_sides);

  double complementarity_sum = 0.0;
  const auto accumulate_shifted_complementarity =
      [&](const double *slack, const double *dual, const int size) {
        for (int index = 0; index < size; ++index) {
          complementarity_sum +=
              (slack[index] + slack_shift) * (dual[index] + dual_shift);
        }
      };
  accumulate_shifted_complementarity(variables.s, variables.z, s_dim);
  accumulate_shifted_complementarity(variables.bound_s, variables.bound_z,
                                     num_bound_sides);

  const int complementarity_dim = s_dim + num_bound_sides;
  const double target_complementarity =
      complementarity_dim > 0
          ? std::max(complementarity_sum / complementarity_dim, 1e-10)
          : 0.0;
  const auto make_strictly_positive = [&](double *slack, double *dual,
                                          const int size) {
    for (int index = 0; index < size; ++index) {
      const double difference = dual[index];
      const double root =
          std::hypot(difference, 2.0 * std::sqrt(target_complementarity));
      dual[index] = difference >= 0.0
                        ? 0.5 * (difference + root)
                        : 2.0 * target_complementarity / (root - difference);
      slack[index] = dual[index] - difference;
    }
  };
  make_strictly_positive(variables.s, variables.z, s_dim);
  make_strictly_positive(variables.bound_s, variables.bound_z, num_bound_sides);

  evaluate_model(data, workspace, variables.x);
  return true;
}

void recover_solution(const Data &data, Workspace &workspace) {
  for (int variable = 0; variable < data.num_primal_variables(); ++variable) {
    workspace.primal_solution[variable] =
        workspace.primal_scaling[variable] * workspace.sip.vars.x[variable];
  }
  for (int equality = 0; equality < data.num_equalities(); ++equality) {
    workspace.equality_dual_solution[equality] =
        workspace.equality_scaling[equality] * workspace.sip.vars.y[equality];
  }
  for (int inequality = 0; inequality < data.num_inequalities(); ++inequality) {
    workspace.inequality_dual_solution[inequality] =
        workspace.inequality_scaling[inequality] *
        workspace.sip.vars.z[inequality];
  }

  int side = 0;
  for (int variable = 0; variable < data.num_primal_variables(); ++variable) {
    double dual = 0.0;
    if (std::isfinite(workspace.scaled_lower_bounds[variable])) {
      dual -= workspace.variable_bound_scaling[variable] *
              workspace.sip.vars.bound_z[side];
      ++side;
    }
    if (std::isfinite(workspace.scaled_upper_bounds[variable])) {
      dual += workspace.variable_bound_scaling[variable] *
              workspace.sip.vars.bound_z[side];
      ++side;
    }
    workspace.variable_bound_dual_solution[variable] = dual;
  }
}

} // namespace

auto solve(const Input &input, const Settings &settings, Workspace &workspace)
    -> Output {
  const Data &data = input.data;
  copy_model(data, workspace);
  compute_scaling(input, settings, workspace);
  apply_scaling(input, workspace);

  const sip_qdldl::Settings qdldl_settings{
      .permute_kkt_system = true,
      .kkt_pinv = input.kkt_ordering.inverse_permutation,
  };
  sip_qdldl::CallbackProvider callback_provider(
      qdldl_settings, workspace.scaled_model, workspace.qdldl);

  if (!initialize_variables(input, settings, workspace, callback_provider)) {
    return Output{
        .sip =
            {
                .exit_status = ::sip::Status::FACTORIZATION_FAILURE,
                .num_iterations = 0,
                .num_ls_iterations = 0,
                .max_primal_violation =
                    std::numeric_limits<double>::signaling_NaN(),
                .max_dual_violation =
                    std::numeric_limits<double>::signaling_NaN(),
            },
        .primal = workspace.primal_solution,
        .equality_dual = workspace.equality_dual_solution,
        .inequality_dual = workspace.inequality_dual_solution,
        .variable_bound_dual = workspace.variable_bound_dual_solution,
    };
  }

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
      .timeout_callback = std::cref(input.timeout_callback),
      .lower_bounds = workspace.scaled_lower_bounds,
      .upper_bounds = workspace.scaled_upper_bounds,
      .residual_scaling =
          {
              .dual = workspace.primal_scaling,
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

  const ::sip::Output sip_output =
      ::sip::solve(sip_input, settings.sip, workspace.sip);
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
