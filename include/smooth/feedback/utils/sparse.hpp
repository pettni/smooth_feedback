
#ifndef SMOOTH__FEEDBACK__UTILS__SPARSE_HPP_
#define SMOOTH__FEEDBACK__UTILS__SPARSE_HPP_

#include <Eigen/Sparse>

#include <numeric>
#include <optional>

namespace smooth::feedback {

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
    l)
{
  const auto n_rows = l.size();
  const auto n_cols = std::begin(l)->size();

  Eigen::VectorXi dims_rows = Eigen::VectorXi::Constant(n_rows, -1);
  Eigen::VectorXi dims_cols = Eigen::VectorXi::Constant(n_cols, -1);

  // figure block row and col dimensions
  for (auto krow = 0u; const auto & row : l) {
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
  for (const auto & row : l) {
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
  for (auto krow = 0u, row0 = 0u; const auto &row : l) {
    for (auto kcol = 0u, col0 = 0u; const auto &item : row) {
      if (item.has_value()) {
        for (auto col = 0; col < dims_cols(kcol); ++col) {
          for (typename std::decay_t<decltype(*item)>::InnerIterator it(*item, col); it; ++it) {
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
  ret.reserve(Eigen::Matrix<int, -1, 1>::Ones(n));
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
template<typename Derived>
inline auto kron_identity(const Eigen::SparseCompressedBase<Derived> & X, std::size_t n)
{
  Eigen::
    SparseMatrix<typename Derived::Scalar, Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor>
      ret(X.rows() * n, X.cols() * n);

  Eigen::Matrix<int, -1, 1> pattern(X.outerSize() * n);

  for (auto i0 = 0u, i = 0u; i < X.outerSize(); ++i) {
    auto nnz_i = X.outerIndexPtr()[i + 1] - X.outerIndexPtr()[i];
    pattern.segment(i0, n).setConstant(nnz_i);
    i0 += n;
  }

  ret.reserve(pattern);

  for (auto i0 = 0u; i0 < X.outerSize(); ++i0) {
    for (typename std::decay_t<decltype(X)>::InnerIterator it(X, i0); it; ++it) {
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

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__UTILS__SPARSE_HPP_
