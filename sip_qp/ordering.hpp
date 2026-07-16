#pragma once

#include "sip_qp/types.hpp"

namespace sip::qp {

struct KktOrderingWorkspace {
  int *permuted_indptr;
  int *permuted_indices;
  int *counts;
  int *cursor;
  int *elimination_tree;
  int *column_counts;
  int *qdldl_workspace;

  void reserve(const Data &data);
  void free();
  auto mem_assign(const Data &data, unsigned char *memory) -> int;
  static auto num_bytes(const Data &data) -> int;
};

auto analyze_kkt_ordering(const Data &data, const int *inverse_permutation,
                          KktOrderingWorkspace &workspace) -> KktOrdering;

} // namespace sip::qp
