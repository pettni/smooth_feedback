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

// \cond
template<typename Scalar, int Layout = Eigen::ColMajor>
using It = typename Eigen::SparseMatrix<Scalar, Layout>::InnerIterator;
// \endcond

/**
 * @brief Compute the column-wise nonzero pattern of L in LDL' factorization of A.
 */
template<typename Scalar, int Layout>
inline auto ldlt_nnz(const Eigen::SparseMatrix<Scalar, Layout> & A)
{
  Eigen::Index n = A.cols();

  // nnz(k) = # nonzeros in row k of L
  Eigen::Matrix<int, -1, 1> nnz = Eigen::Matrix<int, -1, 1>::Ones(n);  // account for diagonal ones

  // elimination tree: tree(i) = min { j : j > i, l_{ji} != 0 }
  // elimination tree is s.t. a non-zero at A_ki causes fill-in in row k for tree successors of i
  Eigen::Matrix<int, -1, 1> tree(n);
  Eigen::Matrix<int, -1, 1> visited(n);

  // traverse each row of L
  for (Eigen::Index row = 0; row != n; ++row) {
    tree(row)    = -1;   // no parent yet
    visited(row) = row;  // mark

    // traverse non-zeros in corresponding row/column of symmetric A (up to diagonal)
    for (It<Scalar, Layout> it(A, row); it && it.index() < row; ++it) {
      // traverse elimination tree to determine fill-in from non-zero A_{col, row}
      for (int col = it.index(); visited(col) != row; col = tree(col)) {
        // found new non-zero
        if (tree(col) == -1) { tree(col) = row; }  // first non-zero in this column
        ++nnz(col);                                // increment count of column non-zeros
        visited(col) = row;
      }
    }
  }

  return std::make_tuple(nnz, tree);
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

    // working memory
    Eigen::Matrix<int, -1, 1> visited(n);
    Eigen::Matrix<int, -1, 1> Yidx(n), bfr(n);  // nonzero indices of Y and temporary buffer
    Eigen::Matrix<Scalar, -1, 1> Yval(n);       // values of Y stored at nonzero indices

    const auto [nnz_col, tree] = ldlt_nnz(A);
    L_.resize(n, n);
    L_.reserve(nnz_col);
    d_inv_.resize(n);

    Scalar d_new = A.coeff(0, 0);
    if (d_new == 0) {
      info_ = 1;
      return;
    }
    d_inv_(0)       = Scalar(1.) / d_new;
    L_.insert(0, 0) = Scalar(1);

    // Fill in L row-wise
    for (Eigen::Index row = 1; row != n; ++row) {
      // solve the triangular sparse system L[:k, :k] y = A[:k, k] w.r.t. y

      // find Yidx -- the non-zero indices of y
      visited[row] = row;
      int Ynnz     = 0;
      Yval(row)    = 0.;
      for (It<Scalar> it(A, row); it && it.index() < row; ++it) {
        int branch_nnz = 0;
        for (int node = it.index(); visited[node] != row; node = tree(node)) {
          visited[node]     = row;
          bfr(branch_nnz++) = node;
        }
        while (branch_nnz > 0) {
          Yidx[Ynnz++] = bfr[--branch_nnz];  // store branch in reverse order
        }
      }

      for (It<Scalar> it(A, row); it && it.index() < row; ++it) {
        Yval(it.index()) = it.value();  // set y[i] = A(row, i)
      }

      // pass two: iterate over columns k of L and perform subtractions y_k -= l_{k, i} y_i
      for (Eigen::Index i = 0; i != Ynnz; ++i) {
        const int col = Yidx[Ynnz - i - 1];  // traverse reverse branch-wise
        It<Scalar> it_l(L_, col);
        ++it_l;  // step past one on diagonal
        for (; it_l && it_l.index(); ++it_l) { Yval(it_l.index()) -= Yval(col) * it_l.value(); }
      }

      // Now y defined by Ynnz, Yidx and Yval solves system above

      // set L[row, :] = Dinv[:row, :row] * y
      // and calculate d_new = A[row, row] - y' Dinv y
      Scalar d_new = A.coeff(row, row);
      for (Eigen::Index i = 0; i != Ynnz; ++i) {
        const int yi        = Yidx[Ynnz - i - 1];
        const Scalar yval_i = Yval[yi];
        Yval[yi]            = 0;                    // reset for next iteration
        L_.insert(row, yi)  = yval_i * d_inv_(yi);  // columns of L are filled in order
        d_new -= yval_i * yval_i * d_inv_(yi);
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
