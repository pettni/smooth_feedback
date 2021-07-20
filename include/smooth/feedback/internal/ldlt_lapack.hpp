
#include <Eigen/Core>

#include <lapacke.h>

/**
 * @file
 * @brief Wrapper for lapack ldlt routinges
 */

namespace smooth::feedback::detail {

// \cond
template<typename Scalar>
struct lapack_ldlt_fcn
{
  static constexpr auto factor = &LAPACKE_ssytrf_work;
  static constexpr auto solve  = &LAPACKE_ssytrs_work;
};

template<>
struct lapack_ldlt_fcn<double>
{
  static constexpr auto factor = &LAPACKE_dsytrf_work;
  static constexpr auto solve  = &LAPACKE_dsytrs_work;
};
// \endcond

/**
 * @brief Wrapper for LAPACKE xSYTRF / xSYTRS routines for factorizing and solving
 * symmetric linear systems of equations.
 */
template<typename Scalar, Eigen::Index N>
  requires std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>
class LDLTLapack
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
  inline LDLTLapack(Eigen::Matrix<Scalar, N, N> && A) : n_(A.cols()), AF_(std::move(A)), IPIV_(n_)
  {
    static constexpr lapack_int LWORK = N == -1 ? -1 : 2 * N;
    Eigen::Matrix<Scalar, LWORK, 1> work(3 * N);

    info_ = (*lapack_ldlt_fcn<Scalar>::factor)(LAPACK_COL_MAJOR,
      'U',           // UPLO
      n_,            // N
      AF_.data(),    // A
      n_,            // LDA
      IPIV_.data(),  // IPIV
      work.data(),
      work.size());
  }

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
   * @brief Factorization status
   *
   * @return integer \f$ i \f$
   *
   * * 0: successful exit
   * * \f$ i > 0 \f$: input matrix is singular s.t. \f$ D(i, i) = 0 \f$.
   */
  inline lapack_int info() const { return info_; }

  /**
   * @brief Solve linear symmetric system of equations in-place.
   *
   * @param[in, out] b in: right-hand side in \f$ A x = b \f$, out: solution \f$ x \f$
   */
  inline void solve_inplace(Eigen::Matrix<Scalar, N, 1> & b)
  {
    info_ = (*lapack_ldlt_fcn<Scalar>::solve)(LAPACK_COL_MAJOR,
      'U',           // UPLO
      n_,            // N
      1,             // NRHS
      AF_.data(),    // A
      n_,            // LDA
      IPIV_.data(),  // IPIV
      b.data(),      // B
      n_);           // LDB
  }

  /**
   * @brief Solve linear symmetric system of equations.
   *
   * @param b right-hand side in \f$ A x = b \f$.
   */
  inline Eigen::Matrix<Scalar, N, 1> solve(const Eigen::Matrix<Scalar, N, 1> & b)
  {
    Eigen::Matrix<Scalar, N, 1> x = b;
    solve_inplace(x);
    return x;
  }

private:
  const lapack_int n_;
  Eigen::Matrix<Scalar, N, N, Eigen::ColMajor> AF_;
  Eigen::Matrix<lapack_int, N, 1> IPIV_;
  lapack_int info_;
};

}  // namespace smooth::feedback::detail
