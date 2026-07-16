#include "sip_qp/types.hpp"

#include <algorithm>
#include <cmath>

namespace sip::qp {
namespace {

auto align_offset(const int offset) -> int {
  constexpr int alignment = alignof(double);
  return ((offset + alignment - 1) / alignment) * alignment;
}

auto hessian_nnz(const Data &data) -> int {
  return data.upper_hessian.indptr[data.upper_hessian.cols];
}

auto equality_jacobian_nnz(const Data &data) -> int {
  return data.transposed_equality_jacobian
      .indptr[data.transposed_equality_jacobian.cols];
}

auto inequality_jacobian_nnz(const Data &data) -> int {
  return data.transposed_inequality_jacobian
      .indptr[data.transposed_inequality_jacobian.cols];
}

auto local_num_doubles(const Data &data) -> int {
  const int kkt_dim = data.num_primal_variables() + data.num_equalities() +
                      data.num_inequalities();
  return 3 * kkt_dim + 5 * data.num_primal_variables();
}

auto assign_double_array(double *&target, const int size, unsigned char *memory,
                         int &offset) -> void {
  target = reinterpret_cast<double *>(memory + offset);
  offset += size * static_cast<int>(sizeof(double));
}

} // namespace

auto Data::num_primal_variables() const -> int { return upper_hessian.cols; }

auto Data::num_equalities() const -> int {
  return transposed_equality_jacobian.cols;
}

auto Data::num_inequalities() const -> int {
  return transposed_inequality_jacobian.cols;
}

auto Data::num_bound_sides() const -> int {
  return ::sip::num_bound_sides(lower_bounds, upper_bounds,
                                num_primal_variables());
}

auto Data::kkt_nnz() const -> int {
  int result = equality_jacobian_nnz(*this) + inequality_jacobian_nnz(*this) +
               num_equalities() + num_inequalities();
  for (int column = 0; column < num_primal_variables(); ++column) {
    bool has_diagonal = false;
    for (int index = upper_hessian.indptr[column];
         index < upper_hessian.indptr[column + 1]; ++index) {
      if (upper_hessian.ind[index] <= column) {
        ++result;
        has_diagonal = has_diagonal || upper_hessian.ind[index] == column;
      }
    }
    result += has_diagonal ? 0 : 1;
  }
  return result;
}

Settings::Settings()
    : sip{
          .mode = ::sip::Mode::REGULARIZED_IPM,
          .max_iterations = 1000,
          .num_iterative_refinement_steps = 0,
          .barrier =
              {
                  .initial_mu = 1e-2,
                  .mu_update_factor = 0.2,
                  .mu_min = 1e-16,
                  .mu_update_kappa = 100.0,
              },
          .penalty =
              {
                  .initial_penalty_parameter = 1.0,
                  .penalty_parameter_increase_factor = 1.35,
                  .max_penalty_parameter = 1e9,
              },
          .termination =
              {
                  .max_dual_residual = 1e-6,
                  .max_constraint_violation = 1e-6,
                  .max_complementarity_gap = 1e-6,
                  .max_merit_slope = 1e-24,
              },
          .regularization =
              {
                  .initial = 3e-5,
                  .first_positive = 1e-8,
                  .maximum = 1e12,
                  .max_attempts = 24,
                  .increase_factor = 4.0,
                  .decrease_factor = 0.15,
              },
          .line_search =
              {
                  .max_iterations = 5000,
                  .tau = 0.995,
                  .skip_line_search = false,
                  .enable_line_search_failures = false,
              },
          .logging =
              {
                  .print_logs = false,
                  .print_line_search_logs = false,
                  .print_search_direction_logs = false,
                  .print_derivative_check_logs = false,
              },
      },
      scaling{} {}

auto default_settings() -> Settings { return Settings{}; }

void Workspace::reserve(const Input &input, const Settings &settings) {
  const Data &data = input.data;
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();
  const int kkt_dim = x_dim + y_dim + s_dim;

  scaled_model.reserve(x_dim, s_dim, y_dim, hessian_nnz(data),
                       equality_jacobian_nnz(data),
                       inequality_jacobian_nnz(data), true, true);
  sip.reserve(x_dim, s_dim, y_dim, data.num_bound_sides(), settings.sip);
  qdldl.reserve(kkt_dim, data.kkt_nnz(), input.kkt_ordering.factor_nnz);

  primal_scaling = new double[x_dim];
  equality_scaling = new double[y_dim];
  inequality_scaling = new double[s_dim];
  variable_bound_scaling = new double[x_dim];
  scaling_norms = new double[kkt_dim];
  scaled_lower_bounds = new double[x_dim];
  scaled_upper_bounds = new double[x_dim];
  scaled_linear_objective = new double[x_dim];
  primal_solution = new double[x_dim];
  equality_dual_solution = new double[y_dim];
  inequality_dual_solution = new double[s_dim];
  variable_bound_dual_solution = new double[x_dim];
}

void Workspace::free() {
  scaled_model.free();
  sip.free();
  qdldl.free();

  delete[] primal_scaling;
  delete[] equality_scaling;
  delete[] inequality_scaling;
  delete[] variable_bound_scaling;
  delete[] scaling_norms;
  delete[] scaled_lower_bounds;
  delete[] scaled_upper_bounds;
  delete[] scaled_linear_objective;
  delete[] primal_solution;
  delete[] equality_dual_solution;
  delete[] inequality_dual_solution;
  delete[] variable_bound_dual_solution;
}

auto Workspace::mem_assign(const Input &input, const Settings &settings,
                           unsigned char *memory) -> int {
  const Data &data = input.data;
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();
  const int kkt_dim = x_dim + y_dim + s_dim;
  int offset = 0;

  assign_double_array(primal_scaling, x_dim, memory, offset);
  assign_double_array(equality_scaling, y_dim, memory, offset);
  assign_double_array(inequality_scaling, s_dim, memory, offset);
  assign_double_array(variable_bound_scaling, x_dim, memory, offset);
  assign_double_array(scaling_norms, kkt_dim, memory, offset);
  assign_double_array(scaled_lower_bounds, x_dim, memory, offset);
  assign_double_array(scaled_upper_bounds, x_dim, memory, offset);
  assign_double_array(scaled_linear_objective, x_dim, memory, offset);
  assign_double_array(primal_solution, x_dim, memory, offset);
  assign_double_array(equality_dual_solution, y_dim, memory, offset);
  assign_double_array(inequality_dual_solution, s_dim, memory, offset);
  assign_double_array(variable_bound_dual_solution, x_dim, memory, offset);

  offset = align_offset(offset);
  offset += scaled_model.mem_assign(
      x_dim, s_dim, y_dim, hessian_nnz(data), equality_jacobian_nnz(data),
      inequality_jacobian_nnz(data), true, true, memory + offset);
  offset = align_offset(offset);
  offset += sip.mem_assign(x_dim, s_dim, y_dim, data.num_bound_sides(),
                           settings.sip, memory + offset);
  offset = align_offset(offset);
  offset += qdldl.mem_assign(kkt_dim, data.kkt_nnz(),
                             input.kkt_ordering.factor_nnz, memory + offset);

  return offset;
}

auto Workspace::num_bytes(const Input &input, const Settings &settings) -> int {
  const Data &data = input.data;
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();
  const int kkt_dim = x_dim + y_dim + s_dim;
  int result = local_num_doubles(data) * static_cast<int>(sizeof(double));

  result = align_offset(result);
  result += sip_qdldl::ModelCallbackOutput::num_bytes(
      x_dim, s_dim, y_dim, hessian_nnz(data), equality_jacobian_nnz(data),
      inequality_jacobian_nnz(data), true, true);
  result = align_offset(result);
  result += ::sip::Workspace::num_bytes(x_dim, s_dim, y_dim,
                                        data.num_bound_sides(), settings.sip);
  result = align_offset(result);
  result += sip_qdldl::Workspace::num_bytes(kkt_dim, data.kkt_nnz(),
                                            input.kkt_ordering.factor_nnz);
  return result;
}

} // namespace sip::qp
