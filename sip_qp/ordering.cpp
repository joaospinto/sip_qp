#include "sip_qp/ordering.hpp"

#include <qdldl.h>

#include <algorithm>
#include <stdexcept>

namespace sip::qp {
namespace {

template <typename Callback>
void for_each_kkt_entry(const Data &data, Callback &&callback) {
  const int x_dim = data.num_primal_variables();
  const int y_dim = data.num_equalities();
  const int s_dim = data.num_inequalities();

  for (int column = 0; column < x_dim; ++column) {
    bool has_diagonal = false;
    for (int index = data.upper_hessian.indptr[column];
         index < data.upper_hessian.indptr[column + 1]; ++index) {
      const int row = data.upper_hessian.ind[index];
      if (row <= column) {
        callback(row, column);
        has_diagonal = has_diagonal || row == column;
      }
    }
    if (!has_diagonal) {
      callback(column, column);
    }
  }
  for (int equality = 0; equality < y_dim; ++equality) {
    const int column = x_dim + equality;
    for (int index = data.transposed_equality_jacobian.indptr[equality];
         index < data.transposed_equality_jacobian.indptr[equality + 1];
         ++index) {
      callback(data.transposed_equality_jacobian.ind[index], column);
    }
    callback(column, column);
  }
  for (int inequality = 0; inequality < s_dim; ++inequality) {
    const int column = x_dim + y_dim + inequality;
    for (int index = data.transposed_inequality_jacobian.indptr[inequality];
         index < data.transposed_inequality_jacobian.indptr[inequality + 1];
         ++index) {
      callback(data.transposed_inequality_jacobian.ind[index], column);
    }
    callback(column, column);
  }
}

auto kkt_dimension(const Data &data) -> int {
  return data.num_primal_variables() + data.num_equalities() +
         data.num_inequalities();
}

void validate_inverse_permutation(const int *inverse_permutation,
                                  const int size, int *seen) {
  std::fill_n(seen, size, 0);
  for (int index = 0; index < size; ++index) {
    const int value = inverse_permutation[index];
    if (value < 0 || value >= size || seen[value] != 0) {
      throw std::invalid_argument("invalid inverse KKT permutation");
    }
    seen[value] = 1;
  }
}

void assign_int_array(int *&target, const int size, unsigned char *memory,
                      int &offset) {
  target = reinterpret_cast<int *>(memory + offset);
  offset += size * static_cast<int>(sizeof(int));
}

} // namespace

void KktOrderingWorkspace::reserve(const Data &data) {
  const int dimension = kkt_dimension(data);
  const int nnz = data.kkt_nnz();
  permuted_indptr = new int[dimension + 1];
  permuted_indices = new int[nnz];
  counts = new int[dimension];
  cursor = new int[dimension];
  elimination_tree = new int[dimension];
  column_counts = new int[dimension];
  qdldl_workspace = new int[dimension];
}

void KktOrderingWorkspace::free() {
  delete[] permuted_indptr;
  delete[] permuted_indices;
  delete[] counts;
  delete[] cursor;
  delete[] elimination_tree;
  delete[] column_counts;
  delete[] qdldl_workspace;
}

auto KktOrderingWorkspace::mem_assign(const Data &data, unsigned char *memory)
    -> int {
  const int dimension = kkt_dimension(data);
  int offset = 0;
  assign_int_array(permuted_indptr, dimension + 1, memory, offset);
  assign_int_array(permuted_indices, data.kkt_nnz(), memory, offset);
  assign_int_array(counts, dimension, memory, offset);
  assign_int_array(cursor, dimension, memory, offset);
  assign_int_array(elimination_tree, dimension, memory, offset);
  assign_int_array(column_counts, dimension, memory, offset);
  assign_int_array(qdldl_workspace, dimension, memory, offset);
  return offset;
}

auto KktOrderingWorkspace::num_bytes(const Data &data) -> int {
  return (data.kkt_nnz() + 6 * kkt_dimension(data) + 1) *
         static_cast<int>(sizeof(int));
}

auto analyze_kkt_ordering(const Data &data, const int *inverse_permutation,
                          KktOrderingWorkspace &workspace) -> KktOrdering {
  const int dimension = kkt_dimension(data);
  validate_inverse_permutation(inverse_permutation, dimension,
                               workspace.counts);

  std::fill_n(workspace.counts, dimension, 0);
  for_each_kkt_entry(data, [&](const int row, const int column) {
    const int permuted_row = inverse_permutation[row];
    const int permuted_column = inverse_permutation[column];
    ++workspace.counts[std::max(permuted_row, permuted_column)];
  });

  workspace.permuted_indptr[0] = 0;
  for (int column = 0; column < dimension; ++column) {
    workspace.permuted_indptr[column + 1] =
        workspace.permuted_indptr[column] + workspace.counts[column];
    workspace.cursor[column] = workspace.permuted_indptr[column];
  }

  for_each_kkt_entry(data, [&](const int row, const int column) {
    const int permuted_row = inverse_permutation[row];
    const int permuted_column = inverse_permutation[column];
    const int output_column = std::max(permuted_row, permuted_column);
    workspace.permuted_indices[workspace.cursor[output_column]++] =
        std::min(permuted_row, permuted_column);
  });

  for (int column = 0; column < dimension; ++column) {
    std::sort(workspace.permuted_indices + workspace.permuted_indptr[column],
              workspace.permuted_indices +
                  workspace.permuted_indptr[column + 1]);
  }

  const int factor_nnz =
      QDLDL_etree(dimension, workspace.permuted_indptr,
                  workspace.permuted_indices, workspace.qdldl_workspace,
                  workspace.column_counts, workspace.elimination_tree);
  if (factor_nnz < 0) {
    throw std::runtime_error("QDLDL rejected the KKT sparsity pattern");
  }
  return KktOrdering{
      .inverse_permutation = inverse_permutation,
      .factor_nnz = factor_nnz,
  };
}

} // namespace sip::qp
