// smooth_feedback: Control theory on Lie groups
// https://github.com/pettni/smooth_feedback
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SMOOTH__FEEDBACK__UTILS__SPARSE_HPP_
#define SMOOTH__FEEDBACK__UTILS__SPARSE_HPP_

/**
 * @file
 * @brief Sparse matrix utilities.
 */

#include <Eigen/Sparse>

#include <numeric>
#include <optional>

namespace smooth::feedback {

template<typename T>
concept SparseMat = (std::is_base_of_v<Eigen::SparseMatrixBase<T>, T>);

/**
 * @brief Block sparse matrix construction.
 *
 * @param blocks list of lists {{b00, b01}, {b10, b11 ...}} of (optional) sparse matrix blocks
 * @return the blocks as a single sparse matrix
 *
 * Non-present (std::nullopt-valued) blocks are considered zeros.
 *
 * @warning The block sizes must be consistent, i.e. all blocks in the same block-column must have
 * the same number of columns, and similarly for row-columns and rows.
 */
inline Eigen::SparseMatrix<double> sparse_block_matrix(
  const std::initializer_list<std::initializer_list<std::optional<Eigen::SparseMatrix<double>>>> &
    blocks)
{
  const auto n_rows = blocks.size();
  const auto n_cols = std::begin(blocks)->size();

  Eigen::VectorXi dims_rows = Eigen::VectorXi::Constant(n_rows, -1);
  Eigen::VectorXi dims_cols = Eigen::VectorXi::Constant(n_cols, -1);

  // figure block row and col dimensions
  for (auto krow = 0u; const auto & row : blocks) {
    for (auto kcol = 0u; const auto & item : row) {
      if (item.has_value()) {
        if (dims_cols(kcol) == -1) {
          dims_cols(kcol) = item->cols();
        } else {
          assert(dims_cols(kcol) == item->cols());
        }
        if (dims_rows(krow) == -1) {
          dims_rows(krow) = item->rows();
        } else {
          assert(dims_rows(krow) == item->rows());
        }
      }
      ++kcol;
    }
    ++krow;
  }

  // check that all dimensions are defined by input args
  assert(dims_rows.minCoeff() > -1);
  assert(dims_cols.minCoeff() > -1);

  // figure starting indices
  const auto n_row = std::accumulate(std::cbegin(dims_rows), std::cend(dims_rows), 0u);
  const auto n_col = std::accumulate(std::cbegin(dims_cols), std::cend(dims_cols), 0u);

  Eigen::SparseMatrix<double> ret(n_row, n_col);

  // allocate pattern
  Eigen::Matrix<decltype(ret)::StorageIndex, -1, 1> pattern(n_col);
  pattern.setZero();
  for (const auto & row : blocks) {
    for (auto kcol = 0u, col0 = 0u; const auto &item : row) {
      if (item.has_value()) {
        for (auto col = 0; col < dims_cols(kcol); ++col) {
          pattern(col0 + col) += item->outerIndexPtr()[col + 1] - item->outerIndexPtr()[col];
        }
      }
      col0 += dims_cols(kcol++);
    }
  }

  ret.reserve(pattern);

  // insert values
  for (auto krow = 0u, row0 = 0u; const auto &row : blocks) {
    for (auto kcol = 0u, col0 = 0u; const auto &item : row) {
      if (item.has_value()) {
        for (auto col = 0; col < dims_cols(kcol); ++col) {
          for (Eigen::InnerIterator it(*item, col); it; ++it) {
            ret.insert(row0 + it.index(), col0 + col) = it.value();
          }
        }
      }
      col0 += dims_cols(kcol++);
    }
    row0 += dims_rows(krow++);
  }

  ret.makeCompressed();

  return ret;
}

/**
 * @brief nxn sparse identity matrix
 *
 * @param n matrix square dimension
 */
inline Eigen::SparseMatrix<double> sparse_identity(std::size_t n)
{
  Eigen::SparseMatrix<double> ret(n, n);
  ret.reserve(Eigen::VectorXi::Ones(n));
  for (auto i = 0u; i < n; ++i) { ret.insert(i, i) = 1; }
  return ret;
}

/**
 * @brief Compute X ⊗ In where X is sparse.
 *
 * @param X sparse matrix in compressed format
 * @param n identity matrix dimension
 *
 * The result has the same storage order as X.
 */
template<SparseMat Mat>
inline auto kron_identity(const Mat & X, std::size_t n)
{
  Eigen::SparseMatrix<typename Mat::Scalar, Mat::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor>
    ret(X.rows() * n, X.cols() * n);

  Eigen::Matrix<int, -1, 1> pattern(X.outerSize() * n);

  for (auto i0 = 0u, i = 0u; i < X.outerSize(); ++i) {
    auto nnz_i = X.outerIndexPtr()[i + 1] - X.outerIndexPtr()[i];
    pattern.segment(i0, n).setConstant(nnz_i);
    i0 += n;
  }

  ret.reserve(pattern);

  for (auto i0 = 0u; i0 < X.outerSize(); ++i0) {
    for (Eigen::InnerIterator it(X, i0); it; ++it) {
      for (auto diag = 0u; diag < n; ++diag) {
        ret.insert(n * it.row() + diag, n * it.col() + diag) = it.value();
      }
    }
  }

  ret.makeCompressed();

  return ret;
}

/**
 * @brief Compute X ⊗ In where X is dense.
 *
 * @param X sparse matrix in compressed format
 * @param n identity matrix dimension
 *
 * The result is column-major.
 */
template<typename Derived>
inline auto kron_identity(const Eigen::MatrixBase<Derived> & X, std::size_t n)
{
  Eigen::SparseMatrix<typename Derived::Scalar> ret(X.rows() * n, X.cols() * n);

  ret.reserve(Eigen::Matrix<int, -1, 1>::Constant(ret.cols(), n * X.rows()));

  for (auto row = 0u; row < X.rows(); ++row) {
    for (auto col = 0u; col < X.cols(); ++col) {
      for (auto diag = 0u; diag < n; ++diag) {
        ret.insert(n * row + diag, n * col + diag) = X(row, col);
      }
    }
  }

  ret.makeCompressed();

  return ret;
}

/**
 * @brief Write block into a sparse matrix (sparse source).
 *
 * After this function the output variable dest is s.t.
 *
 * dest[row0 + r, col0 + c] += scale * source[r, c]
 *
 * @param dest destination
 * @param row0 starting row for block
 * @param col0 starting column for block
 * @param source block values
 * @param scale scaling parameter
 * @param upper_only only add into upper triangular part
 *
 * @note Values are accessed with coeffRef().
 */
template<typename SpMat, int Options>
  requires(std::is_base_of_v<Eigen::SparseMatrixBase<SpMat>, SpMat>)
void block_add(
  Eigen::SparseMatrix<double, Options> & dest,
  Eigen::Index row0,
  Eigen::Index col0,
  const SpMat & source,
  double scale      = 1,
  double upper_only = false)
{
  for (auto c = 0; c < source.outerSize(); ++c) {
    for (Eigen::InnerIterator it(source, c); it; ++it) {
      if (!upper_only || row0 + it.row() <= col0 + it.col()) {
        dest.coeffRef(row0 + it.row(), col0 + it.col()) += scale * it.value();
      }
    }
  }
}

/**
 * @brief Write block into a sparse matrix (dense source).
 *
 * After this function the output variable dest is s.t.
 *
 * dest[row0 + r, col0 + c] += scale * source[r, c]
 *
 * @param dest destination
 * @param row0 starting row for block
 * @param col0 starting column for block
 * @param source block values
 * @param scale scaling parameter
 * @param upper_only only add into upper triangular part
 *
 * @note Values are accessed with coeffRef().
 */
template<typename Derived, int Options>
void block_add(
  Eigen::SparseMatrix<double, Options> & dest,
  Eigen::Index row0,
  Eigen::Index col0,
  const Eigen::MatrixBase<Derived> & source,
  double scale      = 1,
  double upper_only = false)
{
  for (auto c = 0; c < source.cols(); ++c) {
    for (auto r = 0; r < source.rows(); ++r) {
      if (upper_only && row0 + r > col0 + c) { break; }
      dest.coeffRef(row0 + r, col0 + c) += scale * source(r, c);
    }
  }
}

/**
 * @brief Write identity matrix a sparse matrix.
 *
 * After this function the output variable dest is s.t.
 *
 * dest[row0 + k, col0 + k] += scale, k = 0...n
 *
 * @param dest destination
 * @param row0 starting row for block
 * @param col0 starting column for block
 * @param n size of identity matrix
 * @param scale scaling parameter
 *
 * @note Values are accessed with coeffRef().
 */
template<int Options>
void block_add_identity(
  Eigen::SparseMatrix<double, Options> & dest,
  Eigen::Index row0,
  Eigen::Index col0,
  Eigen::Index n,
  double scale = 1)
{
  for (auto k = 0u; k < n; ++k) { dest.coeffRef(row0 + k, col0 + k) += scale; }
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__UTILS__SPARSE_HPP_
