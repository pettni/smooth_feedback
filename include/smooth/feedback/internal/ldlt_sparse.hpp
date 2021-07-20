#include <Eigen/Sparse>

/**
 * @file
 * @brief Wrapper for suitesparse ldl routines
 */

namespace smooth::feedback::detail {

/**
 * @brief Wrapper for suitesparse ldl_XXX routines for factorizing and solving
 * sparse symmetric linear systems of equations.
 */
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
  inline LDLTSparse(const Eigen::SparseMatrix<double> & A)
  {
    auto n = A.cols();

    L_.resize(n, n);
    L_.setIdentity();
    diag_.resize(n);

    diag_(0) = A.coeff(0, 0);

    for (auto k = 1u; k != n; ++k) {
      // TODO this copy must be avoided...
      Eigen::SparseMatrix<double> Lkk(L_.topLeftCorner(k, k));

      Eigen::SparseMatrix<double, Eigen::ColMajor> y = A.col(k).head(k);
      Lkk.triangularView<Eigen::Lower>().solveInPlace(y);

      for (Eigen::SparseMatrix<double>::InnerIterator it(y, 0); it; ++it) {
        L_.coeffRef(k, it.index()) = it.value() / diag_(it.index());
      }

      double Lk_dot_y = 0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(y, 0); it; ++it) {
        Lk_dot_y += it.value() * L_.coeff(k, it.index());
      }
      diag_(k) = A.coeff(k, k) - Lk_dot_y;
    }

    L_.makeCompressed();
  }
  /**
   * @brief Solve linear symmetric system of equations.
   *
   * @param b right-hand side in \f$ A x = b \f$.
   */
  inline Eigen::VectorXd solve(const Eigen::VectorXd & b) const
  {
    Eigen::VectorXd x = b;

    L_.triangularView<Eigen::Lower>().solveInPlace(x);
    x = diag_.cwiseInverse().cwiseProduct(x);
    L_.triangularView<Eigen::Lower>().transpose().solveInPlace(x);

    return x;
  }

private:
  Eigen::VectorXd diag_;
  Eigen::SparseMatrix<double, Eigen::ColMajor> L_;
};

}  // namespace smooth::feedback::detail
