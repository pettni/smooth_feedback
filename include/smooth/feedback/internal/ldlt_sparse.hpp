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

/**
 * @brief Wrapper for suitesparse ldl_XXX routines for factorizing and solving
 * sparse symmetric linear systems of equations.
 */
template<typename Scalar>
class LDLTSparse
{
public:
  /**
   * @brief Factorize symmetric \f$ A \f$ to enable solving \f$ A x = b \f$.
   *
   * \p A is factorized as \f$ A = U D U^T \f$ where \f$ U \f$ is upper triangular
   * and \f$ D \f$ is block-diagonal.
   *
   * @param A sparse symmetric matrix to factorize.
   *
   * @note Only the upper triangular part of \f$ A \f$ is accessed.
   */
  inline LDLTSparse(const Eigen::SparseMatrix<Scalar> & A)
  {
    auto n = A.cols();

    L_.resize(n, n);
    L_.setIdentity();

    // TODO determine sparsity pattern of L via graph method...

    d_inv_.resize(n);

    const Scalar d_new = A.coeff(0, 0);
    if (d_new == 0) {
      info_ = 1;
      return;
    }
    d_inv_(0) = Scalar(1.) / d_new;

    for (auto k = 1u; k != n; ++k) {
      // TODO this copy must be avoided...
      Eigen::SparseMatrix<Scalar> Lkk(L_.topLeftCorner(k, k));

      // solve (k-1) x (k-1) lower triangular sparse system L[:k, :k] y = A[:k, k]
      Eigen::SparseMatrix<Scalar, Eigen::ColMajor> y = A.col(k).head(k);
      Lkk.template triangularView<Eigen::Lower>().solveInPlace(y);

      // set L[k, :] = Dinv[:k, :k] * y
      // and calculate d_new = A[k, k] - y' D y
      Scalar d_new = A.coeff(k, k);
      for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(y, 0); it; ++it) {
        L_.insert(k, it.index()) = it.value() * d_inv_(it.index());
        d_new -= it.value() * it.value() * d_inv_(it.index());
      }

      if (d_new == 0) {
        info_ = k + 1;
        return;
      }
      d_inv_(k) = Scalar(1.) / d_new;
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
   * @param[in, out] b in: right-hand side in \f$ A x = b \f$, out: solution x.
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
   * @return solution x.
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
