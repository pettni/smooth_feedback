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

#ifndef SMOOOTH__FEEDBACK__INTERNAL__LDLT_SPARSE_HPP
#define SMOOOTH__FEEDBACK__INTERNAL__LDLT_SPARSE_HPP

#include <Eigen/Sparse>

/**
 * @file
 * @brief Sparse LDL' for non-singular matrices
 */

namespace smooth::feedback::detail {

template<typename Scalar, int Layout = Eigen::ColMajor>
using It = typename Eigen::SparseMatrix<Scalar, Layout>::InnerIterator;

/**
 * Compute the nonzero pattern of L in LDL' factorization of A
 */
template<typename Scalar, int Layout>
inline auto ldlt_nnz(const Eigen::SparseMatrix<Scalar, Layout> & A)
{
  Eigen::Index n = A.cols();

  // nnz(k) = # nonzeros in row k of L
  Eigen::Matrix<int, -1, 1> nnz =
    Eigen::Matrix<int, -1, 1>::Ones(n);  // account for ones on diagonal
  Eigen::Matrix<int, -1, 1> nnz_row =
    Eigen::Matrix<int, -1, 1>::Ones(n);  // account for ones on diagonal

  // elimination tree: tree(i) = min { j : j > i, l_{ji} != 0 }
  // elimination tree is s.t. a non-zero at A_ki causes fill-in in row k for tree successors of i
  Eigen::Matrix<int, -1, 1> tree(n);
  Eigen::Matrix<int, -1, 1> visited(n);

  // traverse each row of L
  for (auto row = 0u; row != n; ++row) {
    tree(row)    = -1;   // no parent yet
    visited(row) = row;  // mark

    // traverse non-zeros in corresponding row/column of symmetric A (up to diagonal)
    for (It<Scalar, Layout> it(A, row); it && it.index() < row; ++it) {
      // traverse elimination tree to determine fill-in from non-zero A_{col, row}
      for (int col = it.index(); visited(col) != row; col = tree(col)) {
        // found new non-zero
        if (tree(col) == -1) { tree(col) = row; }  // first non-zero in this column
        ++nnz(col);                                // increment count of column non-zeros
        ++nnz_row(row);                            // increment count of column non-zeros
        visited(col) = row;
      }
    }
  }

  return std::make_tuple(nnz, nnz_row, tree);
}

/**
 * @brief Bare-bones LDLt factorization of sparse matrices.
 */
template<typename Scalar>
class LDLTSparse
{
public:
  /**
   * @brief Factorize symmetric \f$ A \f$ to enable solving \f$ A x = b \f$.
   *
   * \p A is factorized as \f$ A = L D L^T \f$ where \f$ L \f$ is lower triangular
   * and \f$ D \f$ is diagonal.
   *
   * @param A sparse symmetric matrix to factorize in column-major format.
   *
   * @note Only the upper triangular part of \f$ A \f$ is accessed.
   */
  inline LDLTSparse(const Eigen::SparseMatrix<Scalar> & A)
  {
    auto n = A.cols();

    const auto [nnz_col, nnz_row, tree] = ldlt_nnz(A);
    L_.resize(n, n);
    L_.reserve(nnz_col);

    d_inv_.resize(n);

    const Scalar d_new = A.coeff(0, 0);
    if (d_new == 0) {
      info_ = 1;
      return;
    }
    d_inv_(0)       = Scalar(1.) / d_new;
    L_.insert(0, 0) = Scalar(1);

    Eigen::Matrix<int, -1, 1> visited(n);

    for (auto row = 1u; row != n; ++row) {
      // this solves the triangular sparse system
      //  L[:k, :k] y = A[:k, k] w.r.t. y
      Eigen::SparseMatrix<Scalar, Eigen::ColMajor> y(row, 1);
      y.reserve(nnz_row(row) - 1);
      visited[row] = row;

      // pass one: insert elements (some zero) to establish y pattern
      for (It<Scalar> it(A, row); it && it.index() < row; ++it) {
        y.coeffRef(it.index(), 0) += it.value();  // set y[i] = A(row, i)
        for (int node = tree(it.index()); visited[node] != row; node = tree(node)) {
          y.coeffRef(node, 0) += 0.;  // mark as nonzero (inefficient, may not be ordered)
          visited[node] = row;
        }
      }

      // pass two: iterate over columns k of L and perform subtractions y_k -= l_{k, i} y_i
      // we can loop over y since pattern is established
      for (It<Scalar> it_y(y, 0); it_y; ++it_y) {
        It<Scalar> it_l(L_, it_y.index());
        ++it_l;  // step past one on diagonal
        for (; it_l && it_l.index() < row; ++it_l) {
          y.coeffRef(it_l.index(), 0) -= it_y.value() * it_l.value();
        }
      }

      // set L[row, :] = Dinv[:row, :row] * y
      // and calculate d_new = A[row, row] - y' D y
      Scalar d_new = A.coeff(row, row);
      for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(y, 0); it; ++it) {
        L_.insert(row, it.index()) = it.value() * d_inv_(it.index());
        d_new -= it.value() * it.value() * d_inv_(it.index());
      }

      if (d_new == 0) {
        info_ = row + 1;
        return;
      }
      d_inv_(row)         = Scalar(1) / d_new;
      L_.insert(row, row) = Scalar(1);
    }

    L_.makeCompressed();

    info_ = 0;
  }

  /// Default copy constructor
  LDLTSparse(const LDLTSparse &) = default;
  /// Default copy assignment
  LDLTSparse & operator=(const LDLTSparse &) = default;
  /// Default move constructor
  LDLTSparse(LDLTSparse &&) = default;
  /// Default move assignment
  LDLTSparse & operator=(LDLTSparse &&) = default;

  /**
   * @brief Factorization status
   *
   * @return integer \f$ i \f$
   *
   * * 0: successful exit
   * * \f$ i > 0 \f$: input matrix is singular s.t. \f$ D(i-1, i-1) = 0 \f$.
   */
  inline int info() const { return info_; }

  /**
   * @brief Solve linear symmetric system of equations.
   *
   * @param[in, out] b in: right-hand side in \f$ A x = b \f$, out: solution \f$ x \f$.
   */
  inline void solve_inplace(Eigen::Matrix<Scalar, -1, 1> & b) const
  {
    L_.template triangularView<Eigen::Lower>().solveInPlace(b);
    b.applyOnTheLeft(d_inv_.asDiagonal());
    L_.template triangularView<Eigen::Lower>().transpose().solveInPlace(b);
  }

  /**
   * @brief Solve linear symmetric system of equations.
   *
   * @param b right-hand side in \f$ A x = b \f$.
   * @return solution \f$ x \f$.
   */
  inline Eigen::Matrix<Scalar, -1, 1> solve(const Eigen::Matrix<Scalar, -1, 1> & b) const
  {
    Eigen::Matrix<Scalar, -1, 1> x(b);
    solve_inplace(x);
    return x;
  }

private:
  int info_;
  Eigen::SparseMatrix<Scalar, Eigen::ColMajor> L_;
  Eigen::Matrix<Scalar, -1, 1> d_inv_;
};

}  // namespace smooth::feedback::detail

#endif  // SMOOOTH__FEEDBACK__INTERNAL__LDLT_SPARSE_HPP
