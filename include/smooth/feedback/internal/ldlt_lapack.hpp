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

#ifndef SMOOOTH__FEEDBACK__INTERNAL__LDLT_LAPACK_HPP_
#define SMOOOTH__FEEDBACK__INTERNAL__LDLT_LAPACK_HPP_

/**
 * @file
 * @brief Wrapper for lapack ldlt routines
 */

#include <Eigen/Core>

using lapack_int = int;

extern "C" void ssytrf_(char const * uplo,
  lapack_int const * n,
  float * A,
  lapack_int const * lda,
  lapack_int * ipiv,
  float * work,
  lapack_int const * lwork,
  lapack_int * info);

extern "C" void ssytrs_(char const * uplo,
  lapack_int const * n,
  lapack_int const * nrhs,
  float const * A,
  lapack_int const * lda,
  lapack_int const * ipiv,
  float * B,
  lapack_int const * ldb,
  lapack_int * info);

extern "C" void dsytrf_(char const * uplo,
  lapack_int const * n,
  double * A,
  lapack_int const * lda,
  lapack_int * ipiv,
  double * work,
  lapack_int const * lwork,
  lapack_int * info);

extern "C" void dsytrs_(char const * uplo,
  lapack_int const * n,
  lapack_int const * nrhs,
  double const * A,
  lapack_int const * lda,
  lapack_int const * ipiv,
  double * B,
  lapack_int const * ldb,
  lapack_int * info);

namespace smooth::feedback::detail {

// \cond
template<typename Scalar>
struct lapack_xsytr
{
  static constexpr auto factor = &ssytrf_;
  static constexpr auto solve  = &ssytrs_;
};

template<>
struct lapack_xsytr<double>
{
  static constexpr auto factor = &dsytrf_;
  static constexpr auto solve  = &dsytrs_;
};
// \endcond

/**
 * @brief Wrapper for LAPACKE xSYTRF / xSYTRS routines for factorizing and solving
 * symmetric linear systems of equations.
 */
template<typename Scalar, Eigen::Index N>
  requires std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double> class LDLTLapack
{
public:
  /**
   * @brief Factorize symmetric \f$ A \f$ to enable solving \f$ A x = b \f$.
   *
   * \p A is factorized as \f$ A = U D U^T \f$ where \f$ U \f$ is upper triangular
   * and \f$ D \f$ is block-diagonal.
   *
   * @param A symmetric matrix to factorize.
   *
   * @note Only the upper triangular part of \f$ A \f$ is accessed.
   */
  inline LDLTLapack(Eigen::Matrix<Scalar, N, N, Eigen::ColMajor> && A)
      : AF_(std::move(A)), IPIV_(AF_.cols())
  {
    static constexpr lapack_int LWORK = N == -1 ? -1 : 3 * N;
    Eigen::Matrix<Scalar, LWORK, 1> work(3 * AF_.cols());

    static constexpr char U = 'U';
    const lapack_int n      = AF_.cols();
    const lapack_int lda    = AF_.rows();
    const lapack_int lwork  = work.size();

    (*lapack_xsytr<Scalar>::factor)(
      &U, &n, AF_.data(), &lda, IPIV_.data(), work.data(), &lwork, &info_);
  }

  /// Default copy constructor
  LDLTLapack(const LDLTLapack &) = default;
  /// Default copy assignment
  LDLTLapack & operator=(const LDLTLapack &) = default;
  /// Default move constructor
  LDLTLapack(LDLTLapack &&) = default;
  /// Default move assignment
  LDLTLapack & operator=(LDLTLapack &&) = default;

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
   * @brief Factorize symmetric \f$ A \f$ to enable solving \f$ A x = b \f$.
   *
   * \p A is factorized as \f$ A = U D U^T \f$ where \f$ U \f$ is upper triangular
   * and \f$ D \f$ is block-diagonal.
   *
   * @param A symmetric matrix to factorize
   *
   * @note Only the upper triangular part of \f$ A \f$ is accessed.
   */
  template<typename Derived>
  inline LDLTLapack(const Eigen::MatrixBase<Derived> & A)
      : LDLTLapack(Eigen::Matrix<Scalar, N, N>(A))
  {}

  /**
   * @brief Solve linear symmetric system of equations in-place.
   *
   * @param[in, out] b in: right-hand side in \f$ A x = b \f$, out: solution \f$ x \f$
   */
  inline void solve_inplace(Eigen::Matrix<Scalar, N, 1> & b)
  {
    static constexpr char U         = 'U';
    static constexpr lapack_int one = 1;
    const lapack_int n              = AF_.cols();
    const lapack_int lda            = AF_.rows();

    (*lapack_xsytr<Scalar>::solve)(
      &U, &n, &one, AF_.data(), &lda, IPIV_.data(), b.data(), &lda, &info_);
  }

  /**
   * @brief Solve linear symmetric system of equations.
   *
   * @param b right-hand side in \f$ A x = b \f$.
   *
   * @return solution \f$ x \f$.
   */
  inline Eigen::Matrix<Scalar, N, 1> solve(const Eigen::Matrix<Scalar, N, 1> & b)
  {
    Eigen::Matrix<Scalar, N, 1> x = b;
    solve_inplace(x);
    return x;
  }

private:
  lapack_int info_;
  Eigen::Matrix<Scalar, N, N, Eigen::ColMajor> AF_;
  Eigen::Matrix<lapack_int, N, 1> IPIV_;
};

}  // namespace smooth::feedback::detail

#endif  // SMOOOTH__FEEDBACK__INTERNAL__LDLT_LAPACK_HPP_
