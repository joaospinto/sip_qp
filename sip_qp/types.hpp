#pragma once

#include "sip/types.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

namespace sip::qp {

struct Data {
  double objective_constant;
  const double *linear_objective;
  sip_qdldl::ConstSparseMatrix upper_hessian;

  const double *equality_offsets;
  sip_qdldl::ConstSparseMatrix transposed_equality_jacobian;

  const double *inequality_offsets;
  sip_qdldl::ConstSparseMatrix transposed_inequality_jacobian;

  const double *lower_bounds;
  const double *upper_bounds;

  auto num_primal_variables() const -> int;
  auto num_equalities() const -> int;
  auto num_inequalities() const -> int;
  auto num_bound_sides() const -> int;
  auto kkt_nnz() const -> int;
};

struct KktOrdering {
  const int *inverse_permutation;
  int factor_nnz;
};

struct Input {
  const Data &data;
  const double *initial_primal;
  KktOrdering kkt_ordering;
  ::sip::Input::TimeoutCallback timeout_callback;
};

struct ScalingSettings {
  int max_iterations = 10;
  double min_norm = 1e-4;
  double max_norm = 1e4;
  double convergence_tolerance = 1e-3;
};

struct Settings {
  Settings();

  ::sip::Settings sip;
  ScalingSettings scaling;
};

auto default_settings() -> Settings;

struct Output {
  ::sip::Output sip;
  const double *primal;
  const double *equality_dual;
  const double *inequality_dual;
  const double *variable_bound_dual;
};

struct Workspace {
  ::sip::Workspace sip;
  sip_qdldl::Workspace qdldl;
  sip_qdldl::ModelCallbackOutput scaled_model;

  double *primal_scaling;
  double *equality_scaling;
  double *inequality_scaling;
  double *variable_bound_scaling;
  double *scaling_norms;
  double *scaled_lower_bounds;
  double *scaled_upper_bounds;
  double *scaled_linear_objective;
  double *primal_solution;
  double *equality_dual_solution;
  double *inequality_dual_solution;
  double *variable_bound_dual_solution;

  void reserve(const Input &input, const Settings &settings);
  void free();
  auto mem_assign(const Input &input, const Settings &settings,
                  unsigned char *memory) -> int;
  static auto num_bytes(const Input &input, const Settings &settings) -> int;
};

} // namespace sip::qp
