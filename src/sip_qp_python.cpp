#include "sip_qp/sip_qp.hpp"

#include "sip/types.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <Eigen/SparseCore>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace nb = nanobind;

namespace sip_qp_python {
namespace {

using ColumnMajorMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using RowMajorMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;
using Array = nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>;
using IndexArray = nb::ndarray<nb::numpy, const int, nb::ndim<1>, nb::c_contig>;

auto copy_array(const Array &array, const int expected_size, const char *name)
    -> std::vector<double> {
  if (static_cast<int>(array.size()) != expected_size) {
    throw nb::value_error(name);
  }
  return std::vector<double>(array.data(), array.data() + array.size());
}

auto const_matrix(const ColumnMajorMatrix &matrix)
    -> sip_qdldl::ConstSparseMatrix {
  return sip_qdldl::ConstSparseMatrix(
      matrix.rows(), matrix.cols(), matrix.innerIndexPtr(),
      matrix.outerIndexPtr(), matrix.valuePtr(), false);
}

auto const_transposed_matrix(const RowMajorMatrix &matrix)
    -> sip_qdldl::ConstSparseMatrix {
  return sip_qdldl::ConstSparseMatrix(
      matrix.cols(), matrix.rows(), matrix.innerIndexPtr(),
      matrix.outerIndexPtr(), matrix.valuePtr(), true);
}

auto is_symmetric(const ColumnMajorMatrix &matrix) -> bool {
  ColumnMajorMatrix difference = matrix - ColumnMajorMatrix(matrix.transpose());
  for (int column = 0; column < difference.outerSize(); ++column) {
    for (ColumnMajorMatrix::InnerIterator entry(difference, column); entry;
         ++entry) {
      if (entry.value() != 0.0) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

struct Result {
  sip::Output info;
  Vector x;
  Vector y;
  Vector z;
  Vector z_box;
};

class Solver {
public:
  Solver(const ColumnMajorMatrix &hessian, const Array &linear_objective,
         const RowMajorMatrix &inequality_jacobian,
         const Array &inequality_upper_bound,
         const RowMajorMatrix &equality_jacobian, const Array &equality_target,
         const Array &lower_bounds, const Array &upper_bounds,
         const IndexArray &kkt_inverse_permutation,
         const sip::qp::Settings &settings, const double time_limit_seconds)
      : hessian_(hessian), inequality_jacobian_(inequality_jacobian),
        equality_jacobian_(equality_jacobian), settings_(settings),
        time_limit_seconds_(time_limit_seconds) {
    const int x_dim = hessian_.rows();
    if (x_dim <= 0 || hessian_.cols() != x_dim ||
        inequality_jacobian_.cols() != x_dim ||
        equality_jacobian_.cols() != x_dim) {
      throw nb::value_error("inconsistent QP matrix dimensions");
    }
    if (!is_symmetric(hessian_)) {
      throw nb::value_error("P must be symmetric");
    }

    const ColumnMajorMatrix upper_hessian =
        hessian_.triangularView<Eigen::Upper>();
    hessian_ = upper_hessian;
    hessian_.makeCompressed();
    inequality_jacobian_.makeCompressed();
    equality_jacobian_.makeCompressed();

    linear_objective_ = copy_array(linear_objective, x_dim, "invalid q size");
    lower_bounds_ = copy_array(lower_bounds, x_dim, "invalid lb size");
    upper_bounds_ = copy_array(upper_bounds, x_dim, "invalid ub size");
    for (int variable = 0; variable < x_dim; ++variable) {
      if (std::isnan(lower_bounds_[variable]) ||
          std::isnan(upper_bounds_[variable]) ||
          lower_bounds_[variable] > upper_bounds_[variable]) {
        throw nb::value_error("invalid variable bounds");
      }
    }

    const auto inequality_bound = copy_array(
        inequality_upper_bound, inequality_jacobian_.rows(), "invalid h size");
    inequality_offsets_.resize(inequality_bound.size());
    std::transform(inequality_bound.begin(), inequality_bound.end(),
                   inequality_offsets_.begin(),
                   [](const double value) { return -value; });

    const auto target = copy_array(equality_target, equality_jacobian_.rows(),
                                   "invalid b size");
    equality_offsets_.resize(target.size());
    std::transform(target.begin(), target.end(), equality_offsets_.begin(),
                   [](const double value) { return -value; });

    const sip::qp::Data data = make_data();
    inverse_permutation_.resize(data.num_primal_variables() +
                                data.num_equalities() +
                                data.num_inequalities());
    if (kkt_inverse_permutation.size() != inverse_permutation_.size()) {
      throw nb::value_error("invalid inverse KKT permutation size");
    }
    std::copy_n(kkt_inverse_permutation.data(), kkt_inverse_permutation.size(),
                inverse_permutation_.data());
    sip::qp::KktOrderingWorkspace ordering_workspace;
    auto ordering_memory = std::make_unique<unsigned char[]>(
        sip::qp::KktOrderingWorkspace::num_bytes(data));
    ordering_workspace.mem_assign(data, ordering_memory.get());
    factor_nnz_ = sip::qp::analyze_kkt_ordering(
                      data, inverse_permutation_.data(), ordering_workspace)
                      .factor_nnz;
    timeout_callback_ = [this] {
      if (std::isinf(time_limit_seconds_)) {
        return false;
      }
      return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                           solve_start_)
                 .count() >= time_limit_seconds_;
    };

    const std::vector<double> initial(x_dim, 0.0);
    const sip::qp::Input input = make_input(data, initial.data());
    workspace_.reserve(input, settings_);
    workspace_reserved_ = true;
  }

  ~Solver() {
    if (workspace_reserved_) {
      workspace_.free();
    }
  }

  Solver(const Solver &) = delete;
  auto operator=(const Solver &) -> Solver & = delete;
  Solver(Solver &&) = delete;
  auto operator=(Solver &&) -> Solver & = delete;

  auto solve(const Array &initial_primal) -> Result {
    const int x_dim = hessian_.rows();
    if (static_cast<int>(initial_primal.size()) != x_dim) {
      throw nb::value_error("invalid initial primal size");
    }

    const sip::qp::Data data = make_data();
    const sip::qp::Input input = make_input(data, initial_primal.data());
    solve_start_ = std::chrono::steady_clock::now();
    const sip::qp::Output output = sip::qp::solve(input, settings_, workspace_);

    return Result{
        .info = output.sip,
        .x = Eigen::Map<const Vector>(output.primal, x_dim),
        .y = Eigen::Map<const Vector>(output.equality_dual,
                                      data.num_equalities()),
        .z = Eigen::Map<const Vector>(output.inequality_dual,
                                      data.num_inequalities()),
        .z_box = Eigen::Map<const Vector>(output.variable_bound_dual, x_dim),
    };
  }

private:
  auto make_data() const -> sip::qp::Data {
    return sip::qp::Data{
        .objective_constant = 0.0,
        .linear_objective = linear_objective_.data(),
        .upper_hessian = const_matrix(hessian_),
        .equality_offsets = equality_offsets_.data(),
        .transposed_equality_jacobian =
            const_transposed_matrix(equality_jacobian_),
        .inequality_offsets = inequality_offsets_.data(),
        .transposed_inequality_jacobian =
            const_transposed_matrix(inequality_jacobian_),
        .lower_bounds = lower_bounds_.data(),
        .upper_bounds = upper_bounds_.data(),
    };
  }

  auto make_input(const sip::qp::Data &data, const double *initial_primal) const
      -> sip::qp::Input {
    return sip::qp::Input{
        .data = data,
        .initial_primal = initial_primal,
        .kkt_ordering =
            {
                .inverse_permutation = inverse_permutation_.data(),
                .factor_nnz = factor_nnz_,
            },
        .timeout_callback = std::cref(timeout_callback_),
    };
  }

  ColumnMajorMatrix hessian_;
  RowMajorMatrix inequality_jacobian_;
  RowMajorMatrix equality_jacobian_;
  std::vector<double> linear_objective_;
  std::vector<double> inequality_offsets_;
  std::vector<double> equality_offsets_;
  std::vector<double> lower_bounds_;
  std::vector<double> upper_bounds_;
  std::vector<int> inverse_permutation_;
  int factor_nnz_{0};
  sip::qp::Settings settings_;
  double time_limit_seconds_;
  std::chrono::steady_clock::time_point solve_start_;
  std::function<bool()> timeout_callback_;
  sip::qp::Workspace workspace_;
  bool workspace_reserved_{false};
};

} // namespace sip_qp_python

NB_MODULE(sip_qp_python_ext, module) {
  module.doc() = "Python bindings for the SIP QP front-end.";

  nb::enum_<sip::Status>(module, "Status")
      .value("SOLVED", sip::Status::SOLVED)
      .value("SUBOPTIMAL", sip::Status::SUBOPTIMAL)
      .value("LOCALLY_INFEASIBLE", sip::Status::LOCALLY_INFEASIBLE)
      .value("ITERATION_LIMIT", sip::Status::ITERATION_LIMIT)
      .value("LINE_SEARCH_ITERATION_LIMIT",
             sip::Status::LINE_SEARCH_ITERATION_LIMIT)
      .value("LINE_SEARCH_FAILURE", sip::Status::LINE_SEARCH_FAILURE)
      .value("TIMEOUT", sip::Status::TIMEOUT)
      .value("FAILED_CHECK", sip::Status::FAILED_CHECK)
      .value("FACTORIZATION_FAILURE", sip::Status::FACTORIZATION_FAILURE)
      .export_values();

  nb::enum_<sip::Mode>(module, "Mode")
      .value("REGULARIZED_IPM", sip::Mode::REGULARIZED_IPM)
      .value("PRIMAL_PROXIMAL_IPM", sip::Mode::PRIMAL_PROXIMAL_IPM)
      .value("PRIMAL_DUAL_PROXIMAL_IPM", sip::Mode::PRIMAL_DUAL_PROXIMAL_IPM);

  nb::class_<sip::RegularizationSettings>(module, "RegularizationSettings")
      .def(nb::init<>())
      .def_rw("initial", &sip::RegularizationSettings::initial)
      .def_rw("first_positive", &sip::RegularizationSettings::first_positive)
      .def_rw("maximum", &sip::RegularizationSettings::maximum)
      .def_rw("max_attempts", &sip::RegularizationSettings::max_attempts)
      .def_rw("increase_factor", &sip::RegularizationSettings::increase_factor)
      .def_rw("decrease_factor", &sip::RegularizationSettings::decrease_factor);

  nb::class_<sip::BarrierSettings>(module, "BarrierSettings")
      .def(nb::init<>())
      .def_rw("initial_mu", &sip::BarrierSettings::initial_mu)
      .def_rw("mu_update_factor", &sip::BarrierSettings::mu_update_factor)
      .def_rw("mu_min", &sip::BarrierSettings::mu_min)
      .def_rw("mu_update_kappa", &sip::BarrierSettings::mu_update_kappa);

  nb::class_<sip::PenaltySettings>(module, "PenaltySettings")
      .def(nb::init<>())
      .def_rw("initial_penalty_parameter",
              &sip::PenaltySettings::initial_penalty_parameter)
      .def_rw("warm_start_penalties",
              &sip::PenaltySettings::warm_start_penalties)
      .def_rw("min_acceptable_constraint_violation_ratio",
              &sip::PenaltySettings::min_acceptable_constraint_violation_ratio)
      .def_rw("penalty_parameter_increase_factor",
              &sip::PenaltySettings::penalty_parameter_increase_factor)
      .def_rw("penalty_parameter_decrease_factor",
              &sip::PenaltySettings::penalty_parameter_decrease_factor)
      .def_rw("max_penalty_parameter",
              &sip::PenaltySettings::max_penalty_parameter);

  nb::class_<sip::TerminationSettings>(module, "SipTerminationSettings")
      .def(nb::init<>())
      .def_rw("max_dual_residual", &sip::TerminationSettings::max_dual_residual)
      .def_rw("max_constraint_violation",
              &sip::TerminationSettings::max_constraint_violation)
      .def_rw("max_complementarity_gap",
              &sip::TerminationSettings::max_complementarity_gap)
      .def_rw("max_suboptimal_constraint_violation",
              &sip::TerminationSettings::max_suboptimal_constraint_violation)
      .def_rw("max_merit_slope", &sip::TerminationSettings::max_merit_slope)
      .def_rw("num_consecutive_stalled_iterations_before_termination",
              &sip::TerminationSettings::
                  num_consecutive_stalled_iterations_before_termination);

  nb::class_<sip::LineSearchSettings>(module, "LineSearchSettings")
      .def(nb::init<>())
      .def_rw("max_iterations", &sip::LineSearchSettings::max_iterations)
      .def_rw("tau", &sip::LineSearchSettings::tau)
      .def_rw("start_ls_with_alpha_s_max",
              &sip::LineSearchSettings::start_ls_with_alpha_s_max)
      .def_rw("armijo_factor", &sip::LineSearchSettings::armijo_factor)
      .def_rw("line_search_factor",
              &sip::LineSearchSettings::line_search_factor)
      .def_rw("line_search_min_step_size",
              &sip::LineSearchSettings::line_search_min_step_size)
      .def_rw("min_merit_slope_to_skip_line_search",
              &sip::LineSearchSettings::min_merit_slope_to_skip_line_search)
      .def_rw("skip_line_search", &sip::LineSearchSettings::skip_line_search)
      .def_rw("use_filter_line_search",
              &sip::LineSearchSettings::use_filter_line_search)
      .def_rw("filter_gamma_theta",
              &sip::LineSearchSettings::filter_gamma_theta)
      .def_rw("filter_gamma_f", &sip::LineSearchSettings::filter_gamma_f)
      .def_rw("filter_min_total_line_search_iterations",
              &sip::LineSearchSettings::filter_min_total_line_search_iterations)
      .def_rw("enable_line_search_failures",
              &sip::LineSearchSettings::enable_line_search_failures);

  nb::class_<sip::LoggingSettings>(module, "LoggingSettings")
      .def(nb::init<>())
      .def_rw("print_logs", &sip::LoggingSettings::print_logs)
      .def_rw("print_line_search_logs",
              &sip::LoggingSettings::print_line_search_logs)
      .def_rw("print_search_direction_logs",
              &sip::LoggingSettings::print_search_direction_logs)
      .def_rw("print_derivative_check_logs",
              &sip::LoggingSettings::print_derivative_check_logs)
      .def_rw("only_check_search_direction_slope",
              &sip::LoggingSettings::only_check_search_direction_slope);

  nb::class_<sip::Settings>(module, "SipSettings")
      .def(nb::init<>())
      .def_rw("mode", &sip::Settings::mode)
      .def_rw("max_iterations", &sip::Settings::max_iterations)
      .def_rw("num_iterative_refinement_steps",
              &sip::Settings::num_iterative_refinement_steps)
      .def_rw("assert_checks_pass", &sip::Settings::assert_checks_pass)
      .def_rw("barrier", &sip::Settings::barrier)
      .def_rw("penalty", &sip::Settings::penalty)
      .def_rw("termination", &sip::Settings::termination)
      .def_rw("regularization", &sip::Settings::regularization)
      .def_rw("line_search", &sip::Settings::line_search)
      .def_rw("logging", &sip::Settings::logging);

  nb::class_<sip::qp::ScalingSettings>(module, "ScalingSettings")
      .def(nb::init<>())
      .def_rw("max_iterations", &sip::qp::ScalingSettings::max_iterations)
      .def_rw("min_norm", &sip::qp::ScalingSettings::min_norm)
      .def_rw("max_norm", &sip::qp::ScalingSettings::max_norm)
      .def_rw("convergence_tolerance",
              &sip::qp::ScalingSettings::convergence_tolerance);

  nb::class_<sip::qp::TerminationSettings>(module, "TerminationSettings")
      .def(nb::init<>())
      .def_rw("max_absolute_residual",
              &sip::qp::TerminationSettings::max_absolute_residual)
      .def_rw("max_relative_residual",
              &sip::qp::TerminationSettings::max_relative_residual)
      .def_rw("max_absolute_duality_gap",
              &sip::qp::TerminationSettings::max_absolute_duality_gap)
      .def_rw("max_relative_duality_gap",
              &sip::qp::TerminationSettings::max_relative_duality_gap);

  nb::class_<sip::qp::Settings>(module, "Settings")
      .def(nb::init<>())
      .def_rw("sip", &sip::qp::Settings::sip)
      .def_rw("scaling", &sip::qp::Settings::scaling)
      .def_rw("termination", &sip::qp::Settings::termination);

  nb::class_<sip::Output>(module, "SolverInfo")
      .def_ro("exit_status", &sip::Output::exit_status)
      .def_ro("num_iterations", &sip::Output::num_iterations)
      .def_ro("num_ls_iterations", &sip::Output::num_ls_iterations)
      .def_ro("max_primal_violation", &sip::Output::max_primal_violation)
      .def_ro("max_dual_violation", &sip::Output::max_dual_violation);

  nb::class_<sip_qp_python::Result>(module, "Result")
      .def_ro("info", &sip_qp_python::Result::info)
      .def_ro("x", &sip_qp_python::Result::x)
      .def_ro("y", &sip_qp_python::Result::y)
      .def_ro("z", &sip_qp_python::Result::z)
      .def_ro("z_box", &sip_qp_python::Result::z_box);

  nb::class_<sip_qp_python::Solver>(module, "Solver")
      .def(nb::init<const sip_qp_python::ColumnMajorMatrix &,
                    const sip_qp_python::Array &,
                    const sip_qp_python::RowMajorMatrix &,
                    const sip_qp_python::Array &,
                    const sip_qp_python::RowMajorMatrix &,
                    const sip_qp_python::Array &, const sip_qp_python::Array &,
                    const sip_qp_python::Array &,
                    const sip_qp_python::IndexArray &,
                    const sip::qp::Settings &, double>(),
           nb::arg("P"), nb::arg("q"), nb::arg("G"), nb::arg("h"), nb::arg("A"),
           nb::arg("b"), nb::arg("lb"), nb::arg("ub"),
           nb::arg("kkt_inverse_permutation"),
           nb::arg("settings") = sip::qp::Settings{},
           nb::arg("time_limit_s") = std::numeric_limits<double>::infinity())
      .def("solve", &sip_qp_python::Solver::solve, nb::arg("initial_primal"));
}
