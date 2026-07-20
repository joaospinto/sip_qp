#include "sip_qp/sip_qp.hpp"
#include "tests/allocation_counter.hpp"

#include "sip/types.hpp"
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <functional>
#include <vector>

namespace sip::qp {
namespace {

struct EqualityConstrainedQp {
  std::array<int, 3> hessian_indptr{0, 1, 3};
  std::array<int, 3> hessian_indices{0, 0, 1};
  std::array<double, 3> hessian_data{4.0, 1.0, 2.0};
  std::array<double, 2> linear_objective{1.0, 1.0};

  std::array<int, 2> equality_indptr{0, 2};
  std::array<int, 2> equality_indices{0, 1};
  std::array<double, 2> equality_data{1.0, 1.0};
  std::array<double, 1> equality_offsets{-1.0};

  std::array<int, 1> inequality_indptr{0};
  std::array<double, 2> lower_bounds{0.0, 0.0};
  std::array<double, 2> upper_bounds{0.7, 0.7};
  std::array<double, 2> initial_primal{0.0, 0.0};
  std::array<int, 3> inverse_permutation{0, 1, 2};

  Data data{
      .objective_constant = 0.0,
      .linear_objective = linear_objective.data(),
      .upper_hessian = {2, 2, hessian_indices.data(), hessian_indptr.data(),
                        hessian_data.data(), false},
      .equality_offsets = equality_offsets.data(),
      .transposed_equality_jacobian = {2, 1, equality_indices.data(),
                                       equality_indptr.data(),
                                       equality_data.data(), true},
      .inequality_offsets = nullptr,
      .transposed_inequality_jacobian = {2, 0, nullptr,
                                         inequality_indptr.data(), nullptr,
                                         true},
      .lower_bounds = lower_bounds.data(),
      .upper_bounds = upper_bounds.data(),
  };

  KktOrderingWorkspace ordering_workspace{};
  KktOrdering kkt_ordering{};
  std::function<bool()> timeout = [] { return false; };
  Input input{
      .data = data,
      .initial_primal = initial_primal.data(),
      .kkt_ordering = {},
      .timeout_callback = std::cref(timeout),
  };

  EqualityConstrainedQp() {
    ordering_workspace.reserve(data);
    kkt_ordering = analyze_kkt_ordering(data, inverse_permutation.data(),
                                        ordering_workspace);
    input.kkt_ordering = kkt_ordering;
  }

  ~EqualityConstrainedQp() { ordering_workspace.free(); }
};

void expect_equality_constrained_solution(const Output &output) {
  EXPECT_EQ(output.sip.exit_status, ::sip::Status::SOLVED);
  EXPECT_NEAR(output.primal[0], 0.3, 1e-5);
  EXPECT_NEAR(output.primal[1], 0.7, 1e-5);
  EXPECT_LT(output.sip.max_primal_violation, 1e-6);
  EXPECT_LT(output.sip.max_dual_violation, 1e-6);
  EXPECT_GT(output.variable_bound_dual[1], 0.0);
}

TEST(SipQpTest, SolvesWithReservedWorkspace) {
  EqualityConstrainedQp problem;
  const Settings settings = default_settings();
  Workspace workspace;
  workspace.reserve(problem.input, settings);

  expect_equality_constrained_solution(
      solve(problem.input, settings, workspace));
  EXPECT_DOUBLE_EQ(workspace.objective_scaling, 1.0);

  workspace.free();
}

TEST(SipQpTest, AnalyzesUserDefinedKktOrdering) {
  EqualityConstrainedQp problem;
  const std::array<int, 3> inverse_permutation{0, 1, 2};
  const int num_bytes = KktOrderingWorkspace::num_bytes(problem.data);
  std::vector<unsigned char> memory(num_bytes);
  KktOrderingWorkspace workspace;
  EXPECT_EQ(workspace.mem_assign(problem.data, memory.data()), num_bytes);

  allocation_counter::start();
  const KktOrdering ordering =
      analyze_kkt_ordering(problem.data, inverse_permutation.data(), workspace);
  const std::size_t num_allocations = allocation_counter::stop();

  EXPECT_EQ(ordering.inverse_permutation, inverse_permutation.data());
  EXPECT_GT(ordering.factor_nnz, 0);
  EXPECT_EQ(num_allocations, 0);
}

TEST(SipQpTest, SolvesWithAssignedWorkspace) {
  EqualityConstrainedQp problem;
  const Settings settings = default_settings();
  const int num_bytes = Workspace::num_bytes(problem.input, settings);
  std::vector<unsigned char> memory(num_bytes);
  Workspace workspace;

  EXPECT_EQ(workspace.mem_assign(problem.input, settings, memory.data()),
            num_bytes);
  allocation_counter::start();
  const Output output = solve(problem.input, settings, workspace);
  const std::size_t num_allocations = allocation_counter::stop();

  expect_equality_constrained_solution(output);
  EXPECT_EQ(num_allocations, 0);
}

TEST(SipQpTest, ObjectiveScalingPreservesOriginalCoordinates) {
  EqualityConstrainedQp problem;
  problem.hessian_data = {0.04, 0.01, 0.02};
  problem.linear_objective = {0.0, 0.0};
  Settings settings = default_settings();
  settings.scaling.scale_homogeneous_objective = true;

  Workspace workspace;
  workspace.reserve(problem.input, settings);
  const Output base_output = solve(problem.input, settings, workspace);
  const std::array<double, 2> base_primal{base_output.primal[0],
                                          base_output.primal[1]};
  const double base_equality_dual = base_output.equality_dual[0];
  const double base_bound_dual = base_output.variable_bound_dual[1];
  const double base_objective_scaling = workspace.objective_scaling;
  const ::sip::Status base_status = base_output.sip.exit_status;
  workspace.free();

  constexpr double objective_multiplier = 1e6;
  for (double &entry : problem.hessian_data) {
    entry *= objective_multiplier;
  }
  workspace.reserve(problem.input, settings);
  const Output scaled_output = solve(problem.input, settings, workspace);

  EXPECT_EQ(base_status, ::sip::Status::SOLVED);
  EXPECT_EQ(scaled_output.sip.exit_status, ::sip::Status::SOLVED);
  EXPECT_NE(base_objective_scaling, 1.0);
  EXPECT_TRUE(std::isfinite(base_objective_scaling));
  EXPECT_TRUE(std::isfinite(workspace.objective_scaling));
  EXPECT_NEAR(scaled_output.primal[0], base_primal[0], 2e-7);
  EXPECT_NEAR(scaled_output.primal[1], base_primal[1], 2e-7);
  EXPECT_NEAR(scaled_output.equality_dual[0],
              objective_multiplier * base_equality_dual,
              1e-6 * std::abs(objective_multiplier * base_equality_dual));
  EXPECT_NEAR(scaled_output.variable_bound_dual[1],
              objective_multiplier * base_bound_dual,
              3e-6 * std::abs(objective_multiplier * base_bound_dual));

  workspace.free();
}

TEST(SipQpTest, ObjectiveScalingRemainsFiniteAtExtremeMagnitudes) {
  EqualityConstrainedQp problem;
  problem.hessian_data = {1e300, 2e299, 5e299};
  problem.linear_objective = {0.0, 0.0};
  problem.timeout = [] { return true; };
  const Settings settings = default_settings();

  Workspace workspace;
  workspace.reserve(problem.input, settings);
  solve(problem.input, settings, workspace);

  EXPECT_TRUE(std::isfinite(workspace.objective_scaling));
  EXPECT_GE(workspace.objective_scaling, 1.0 / settings.scaling.max_norm);
  EXPECT_LE(workspace.objective_scaling, 1.0 / settings.scaling.min_norm);
  const auto &scaled_hessian = workspace.scaled_model.upper_hessian_lagrangian;
  for (int index = 0; index < scaled_hessian.indptr[scaled_hessian.cols];
       ++index) {
    EXPECT_TRUE(std::isfinite(scaled_hessian.data[index]));
  }

  workspace.free();
}

TEST(SipQpTest, EquilibratesLargeConstraintCoefficients) {
  const std::array<int, 2> hessian_indptr{0, 1};
  const std::array<int, 1> hessian_indices{0};
  const std::array<double, 1> hessian_data{1.0};
  const std::array<double, 1> linear_objective{0.0};
  const std::array<int, 1> equality_indptr{0};
  const std::array<int, 2> inequality_indptr{0, 1};
  const std::array<int, 1> inequality_indices{0};
  const std::array<double, 1> inequality_data{-1e6};
  const std::array<double, 1> inequality_offsets{1e6};
  const std::array<double, 1> lower_bounds{-INFINITY};
  const std::array<double, 1> upper_bounds{INFINITY};
  const std::array<double, 1> initial_primal{0.0};
  std::array<int, 2> inverse_permutation{0, 1};

  const Data data{
      .objective_constant = 0.0,
      .linear_objective = linear_objective.data(),
      .upper_hessian = {1, 1, hessian_indices.data(), hessian_indptr.data(),
                        hessian_data.data(), false},
      .equality_offsets = nullptr,
      .transposed_equality_jacobian = {1, 0, nullptr, equality_indptr.data(),
                                       nullptr, true},
      .inequality_offsets = inequality_offsets.data(),
      .transposed_inequality_jacobian = {1, 1, inequality_indices.data(),
                                         inequality_indptr.data(),
                                         inequality_data.data(), true},
      .lower_bounds = lower_bounds.data(),
      .upper_bounds = upper_bounds.data(),
  };
  const std::function<bool()> timeout = [] { return false; };
  KktOrderingWorkspace ordering_workspace;
  ordering_workspace.reserve(data);
  const KktOrdering kkt_ordering = analyze_kkt_ordering(
      data, inverse_permutation.data(), ordering_workspace);
  const Input input{
      .data = data,
      .initial_primal = initial_primal.data(),
      .kkt_ordering = kkt_ordering,
      .timeout_callback = std::cref(timeout),
  };
  const Settings settings = default_settings();
  Workspace workspace;
  workspace.reserve(input, settings);

  const Output output = solve(input, settings, workspace);

  EXPECT_EQ(output.sip.exit_status, ::sip::Status::SOLVED);
  EXPECT_NEAR(output.primal[0], 1.0, 1e-5);
  EXPECT_LT(output.sip.max_primal_violation, 1e-6);
  EXPECT_LT(output.sip.max_dual_violation, 1e-6);
  workspace.free();
  ordering_workspace.free();
}

TEST(SipQpTest, IncludesVariableBoundRowsInScaling) {
  EqualityConstrainedQp problem;
  problem.hessian_data = {1e-3, 0.0, 1e-3};
  problem.equality_data = {0.0, 0.0};
  problem.timeout = [] { return true; };
  const Settings settings = default_settings();
  Workspace workspace;
  workspace.reserve(problem.input, settings);

  solve(problem.input, settings, workspace);

  EXPECT_DOUBLE_EQ(workspace.primal_scaling[0], 1.0);
  EXPECT_DOUBLE_EQ(workspace.primal_scaling[1], 1.0);
  workspace.free();
}

} // namespace
} // namespace sip::qp
