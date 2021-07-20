#include <Eigen/Sparse>

#define LDL_LONG

#include <suitesparse/ldl.h>

/**
 * @file
 * @brief Wrapper for suitesparse ldl routines
 */

namespace smooth::feedback::detail {

/**
 * @brief Wrapper for suitesparse ldl_XXX routines for factorizing and solving
 * sparse symmetric linear systems of equations.
 */
class LDLTSuiteSparse
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
inline LDLT_sparse(Eigen::SparseMatrix<double, Eigen::ColMajor, LDL_int> & A)
  {
    A.makeCompressed();
    LDL_int n = A.cols();

    LDL_int * Ap    = A.outerIndexPtr();
    LDL_int * Ai    = A.innerIndexPtr();
    double * Ax = A.valuePtr();

    LDL_int * Lp     = new LDL_int[n + 1];
    LDL_int * Parent = new LDL_int[n];
    LDL_int * Lnz    = new LDL_int[n];
    LDL_int * Flag   = new LDL_int[n];

    LDL_symbolic(n, Ap, Ai, Lp, Parent, Lnz, Flag, NULL, NULL);

    LDL_int * Li      = new LDL_int[(unsigned long)Lnz];
    double * Lx   = new double[(unsigned long)Lnz];
    double * Y    = new double[n];
    LDL_int * Pattern = new LDL_int[n];

    // LDL_numeric(n, Ap, Ai, Ax, Lp, Parent, Lnz, Li, Lx, D_.data(), Y, Pattern, Flag, NULL, NULL);

    Eigen::Index Lnz_tot = 0;
    for (LDL_int i = 0; i != n; ++i) {
      Lnz_tot += Lnz[i];
    }

    Eigen::Map<Eigen::SparseMatrix<double, Eigen::ColMajor, LDL_int>> lM(n, n, Lnz_tot, Lp, Li, Lx);

    delete[] Lp;
    delete[] Parent;
    delete[] Lnz;
    delete[] Flag;
    delete[] Li;
    delete[] Lx;
    delete[] Y;
    delete[] Pattern;
  }

  /**
   * @brief Factorization status
   *
   * @return integer \f$ i \f$
   *
   * * 0: successful exit
   * * \f$ i > 0 \f$: input matrix is singular s.t. \f$ D(i, i) = 0 \f$.
   */
  // inline lapack_int info() const { return info_; }

  /**
   * @brief Solve linear symmetric system of equations in-place.
   *
   * @param[in, out] b in: right-hand side in \f$ A x = b \f$, out: solution \f$ x \f$
   */
  // inline void solve_inplace(Eigen::Matrix<Scalar, N, 1> & b)

  /**
   * @brief Solve linear symmetric system of equations.
   *
   * @param b right-hand side in \f$ A x = b \f$.
   */
  inline Eigen::VectorXd solve(const Eigen::VectorXd & b)
  {
    Eigen::VectorXd x = b;
    return x;
  }

private:
  Eigen::SparseMatrix<double, Eigen::ColMajor, LDL_int> L_;
  Eigen::VectorXd D_;
};

}  // namespace smooth::feedback::detail
