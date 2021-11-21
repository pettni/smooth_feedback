/**
 * @file
 * @brief Nonlinear programming
 */

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <smooth/diff.hpp>

#include <limits>

#include "collocation.hpp"

namespace smooth::feedback {

/**
 * @brief Optimal control problem on the interval \f$ t \in [0, t_f] \f$.
 * \f[
 * \begin{cases}
 *  \min        & \theta(t_f, x_0, x_f, q)                                              \\
 *  \text{s.t.} & x(0) = x_0                                                            \\
 *              & x(t_f) = x_f                                                          \\
 *              & \dot x(t) = f(t, x(t), u(t))                                          \\
 *              & q = \int_{0}^{t_f} g(t, x(t), u(t)) \mathrm{d}t                       \\
 *              & c_{rl} \leq c_r(t, x(t), u(t)) \leq c_{ru} \quad t \in [0, t_f]       \\
 *              & c_{el} \leq c_e(t_f, x_0, x_f, q) \leq c_{eu}
 * \end{cases}
 * \f]
 *
 * The optimal control problem depends on arbitrary functions \f$ \theta, f, g, c_r, c_e \f$.
 * The type of those functions are template pararamters in this structure.
 *
 * @note To enable automatic differentiation \f$ \theta, g, c_r, c_e \f$ must be templated over the
 * scalar type.
 */
template<typename Theta, typename F, typename G, typename CR, typename CE>
struct OCP
{
  /// @brief State dimension \f$ n_{x} \f$
  std::size_t nx;
  /// @brief Input dimension \f$ n_{u} \f$
  std::size_t nu;
  /// @brief Number of integrals \f$ n_{q} \f$
  std::size_t nq;
  /// @brief Number of running constraints \f$ n_{cr} \f$
  std::size_t ncr;
  /// @brief Number of end constraints \f$ n_{ce} \f$
  std::size_t nce;

  /// @brief Objective function
  Theta theta;
  /// @brief System dynamics \f$ f : R \times R^{n_x} \times R^{n_u} \rightarrow R^{n_x} \f$
  F f;
  /// @brief Integrals \f$ g : R \times R^{n_x} \times R^{n_u} \rightarrow R^{n_q} \f$
  G g;
  /// @brief Integrals \f$ c_r : R \times R^{n_x} \times R^{n_u} \rightarrow R^{n_{cr}} \f$
  CR cr;
  /// @brief Running constraint lower bound \f$ c_{rl} \in R^{n_{cr}} \f$
  Eigen::VectorXd crl;
  /// @brief Running constraint upper bound \f$ c_{ru} \in R^{n_{cr}} \f$
  Eigen::VectorXd cru;
  /// @brief End constraint \f$ c_e : R \times R^{n_x} \times R^{n_x} \times R^{n_q} \rightarrow
  /// R^{n_{ce}} \f$
  CE ce;
  /// @brief End constraint lower bound \f$ c_{el} \in R^{n_{ce}} \f$
  Eigen::VectorXd cel;
  /// @brief End constraint upper bound \f$ c_{eu} \in R^{n_{ce}} \f$
  Eigen::VectorXd ceu;
};

/**
 * @brief Nonlinear Programming Problem
 * \f[
 *  \begin{cases}
 *   \min_{x}    & f(x)                    \\
 *   \text{s.t.} & x_l \leq x \leq x_u     \\
 *               & g_l \leq g(x) \leq g_u
 *  \end{cases}
 * \f]
 * for \f$ f : \mathbb{R}^n \rightarrow \mathbb{R} \f$ and
 * \f$ g : \mathbb{R}^n \rightarrow \mathbb{R}^m \f$.
 */
struct NLP
{
  /// @brief Number of variables
  std::size_t n;

  /// @brief Number of constraints
  std::size_t m;

  /// @brief Objective function
  std::function<double(Eigen::VectorXd)> f;
  //
  /// @brief Variable bounds
  Eigen::VectorXd xl, xu;

  /// @brief Constraint function
  std::function<Eigen::VectorXd(Eigen::VectorXd)> g;

  /// @brief Constaint bounds
  Eigen::VectorXd gl, gu;

  /// @brief Jacobian of objective function (R^n -> R^{n x n})
  std::function<Eigen::SparseMatrix<double>(Eigen::VectorXd)> df_dx;

  /// @brief Jacobian of constraint function (R^n -> R^{m x n})
  std::function<Eigen::SparseMatrix<double>(Eigen::VectorXd)> dg_dx;

  /// @brief Hessian of objective function (R^n -> R^{n x n}) [optional]
  std::optional<std::function<Eigen::SparseMatrix<double>(Eigen::VectorXd, Eigen::VectorXd)>>
    d2f_dx2 = std::nullopt;

  /**
   * @brief Projected Hessian of constraint function (R^m, R^n -> R^{n x n}) [optional]
   *
   * Returns the derivative
   * \f[
   *  H_g(\lambda, x) = \nabla^2_x \lambda^T g(x), \quad \lambda \in \mathbb{R}^m, x \in
   * \mathbb{R}^n \f]
   */
  std::optional<std::function<Eigen::SparseMatrix<double>(Eigen::VectorXd, Eigen::VectorXd)>>
    d2g_dx2 = std::nullopt;
};

/**
 * @brief Formulate an OCP as a NLP using collocation on a mesh.
 *
 * @param ocp Optimal control problem definition
 * @param mesh Mesh that describes collocation point structure
 */
template<typename Theta,
  typename F,
  typename G,
  typename CR,
  typename CE,
  std::size_t Kmin,
  std::size_t Kmax>
auto ocp_to_nlp(const OCP<Theta, F, G, CR, CE> & ocp, const Mesh<Kmin, Kmax> & mesh)
{
  std::size_t N = mesh.N_colloc();

  // variable layout
  std::array<std::size_t, 4> var_len{
    1,                 // tf
    ocp.nq,            // integrals
    ocp.nx * (N + 1),  // states
    ocp.nu * N,        // inputs
  };

  // constraint layout
  std::array<std::size_t, 4> con_len{
    ocp.nx * N,   // derivatives
    ocp.nq,       // other integrals
    ocp.ncr * N,  // running constraints
    ocp.nce,      // end constraints
  };

  std::array<std::size_t, 5> var_beg{0};
  std::partial_sum(var_len.begin(), var_len.end(), var_beg.begin() + 1);

  std::array<std::size_t, 5> con_beg{0};
  std::partial_sum(con_len.begin(), con_len.end(), con_beg.begin() + 1);

  const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
  const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

  const auto [dcon_B, qcon_B, crcon_B, cecon_B, m] = con_beg;
  const auto [dcon_L, qcon_L, crcon_L, cecon_L]    = con_len;

  // OBJECTIVE FUNCTION

  auto f = [var_beg, var_len, ocp = ocp](const Eigen::VectorXd & x) -> double {
    const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
    const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

    assert(std::size_t(x.size()) == n);

    const double tf          = x(tfvar_B);
    const Eigen::VectorXd q  = x.segment(qvar_B, qvar_L);
    const Eigen::VectorXd x0 = x.segment(xvar_B, ocp.nx);
    const Eigen::VectorXd xf = x.segment(xvar_B + xvar_L - ocp.nx, ocp.nx);

    return ocp.theta.template operator()<double>(tf, x0, xf, q);
  };

  // OBJECTIVE JACOBIAN

  auto df_dx = [var_beg, var_len, ocp = ocp](
                 const Eigen::VectorXd & x) -> Eigen::SparseMatrix<double> {
    const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
    const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

    const double tf          = x(tfvar_B);
    const Eigen::VectorXd q  = x.segment(qvar_B, qvar_L);
    const Eigen::VectorXd x0 = x.segment(xvar_B, ocp.nx);
    const Eigen::VectorXd xf = x.segment(xvar_B + xvar_L - ocp.nx, ocp.nx);

    auto [val, J] = diff::dr(ocp.theta, wrt(tf, x0, xf, q));

    Eigen::SparseMatrix<double> ret(1, n);
    ret.reserve(1 + 2 * ocp.nx + ocp.nq);

    ret.insert(0, tfvar_B) = J(0);
    for (auto i = 0u; i < ocp.nx; ++i) { ret.insert(0, xvar_B + i) = J(1 + i); }
    for (auto i = 0u; i < ocp.nx; ++i) {
      ret.insert(0, xvar_B + xvar_L - ocp.nx + i) = J(1 + ocp.nx + i);
    }
    for (auto i = 0u; i < ocp.nq; ++i) { ret.insert(0, qvar_B + i) = J(1 + 2 * ocp.nx + i); }

    ret.makeCompressed();

    return ret;
  };

  // VARIABLE BOUNDS

  Eigen::VectorXd xl = Eigen::VectorXd::Constant(n, -std::numeric_limits<double>::infinity());
  Eigen::VectorXd xu = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());

  xl.segment(tfvar_B, tfvar_L).setZero();  // tf lower bounded by zero

  // CONSTRAINT FUNCTION

  auto g = [var_beg, var_len, con_beg, con_len, mesh, ocp = ocp](
             const Eigen::VectorXd & x) -> Eigen::VectorXd {
    const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
    const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

    const auto [dcon_B, qcon_B, crcon_B, cecon_B, m] = con_beg;
    const auto [dcon_L, qcon_L, crcon_L, cecon_L]    = con_len;

    assert(std::size_t(x.size()) == n);

    const double t0          = 0;
    const double tf          = x(tfvar_B);
    const Eigen::VectorXd Qm = x.segment(qvar_B, qvar_L);
    const Eigen::MatrixXd Xm = x.segment(xvar_B, xvar_L).reshaped(ocp.nx, xvar_L / ocp.nx);
    const Eigen::MatrixXd Um = x.segment(uvar_B, uvar_L).reshaped(ocp.nu, uvar_L / ocp.nu);

    const Eigen::VectorXd x0 = Xm.leftCols(1);
    const Eigen::VectorXd xf = Xm.rightCols(1);

    // dynamics constraint
    const auto Fval = dynamics_constraint<false>(ocp.nx, ocp.f, mesh, t0, tf, Xm, Um);

    // integral constraint
    const auto Gval = integral_constraint<false>(ocp.nq, ocp.g, mesh, t0, tf, Qm, Xm, Um);

    // running constraints
    const auto CRval = colloc_eval<false>(ocp.ncr, ocp.cr, mesh, t0, tf, Xm, Um);

    // end constraints
    const auto CEval = endpoint_eval<false>(ocp.nce, ocp.nx, ocp.ce, t0, tf, Xm, Qm);

    Eigen::VectorXd ret(m);
    ret.segment(dcon_B, dcon_L)   = Fval.reshaped();
    ret.segment(qcon_B, qcon_L)   = Gval.reshaped();
    ret.segment(crcon_B, crcon_L) = CRval.reshaped();
    ret.segment(cecon_B, cecon_L) = CEval.reshaped();
    return ret;
  };

  // CONSTRAINT JACOBIAN
  auto dg_dx = [var_beg, var_len, mesh, ocp = ocp](const Eigen::VectorXd & x) {
    const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
    const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

    assert(std::size_t(x.size()) == n);

    const double t0          = 0;
    const double tf          = x(tfvar_B);
    const Eigen::VectorXd Qm = x.segment(qvar_B, qvar_L);
    const Eigen::MatrixXd Xm = x.segment(xvar_B, xvar_L).reshaped(ocp.nx, xvar_L / ocp.nx);
    const Eigen::MatrixXd Um = x.segment(uvar_B, uvar_L).reshaped(ocp.nu, uvar_L / ocp.nu);

    // dynamics constraint
    const auto [Fval, dF_dt0, dF_dtf, dF_dX, dF_dU] =
      dynamics_constraint<true>(ocp.nx, ocp.f, mesh, t0, tf, Xm, Um);

    // integral constraint
    const auto [Gval, dG_dt0, dG_dtf, dG_dQ, dG_dX, dG_dU] =
      integral_constraint<true>(ocp.nq, ocp.g, mesh, t0, tf, Qm, Xm, Um);

    // running constraints
    const auto [CRval, dCR_dt0, dCR_dtf, dCR_dX, dCR_dU] =
      colloc_eval<true>(ocp.ncr, ocp.cr, mesh, t0, tf, Xm, Um);

    // end constraints
    const auto [CEval, dCE_dt0, dCE_dtf, dCE_dX, dCE_dQ] =
      endpoint_eval<true>(ocp.nce, ocp.nx, ocp.ce, t0, tf, Xm, Qm);

    return sparse_block_matrix({
      // clang-format off
      {dF_dtf,      {}, dF_dX,   dF_dU},
      {dG_dtf,   dG_dQ, dG_dX,   dG_dU},
      {dCR_dtf,     {}, dCR_dX, dCR_dU},
      {dCE_dtf, dCE_dQ, dCE_dX,     {}},
      // clang-format on
    });
  };

  // CONSTRAINT BOUNDS

  Eigen::VectorXd gl(m);
  Eigen::VectorXd gu(m);

  // derivative constraints are equalities
  gl.segment(dcon_B, dcon_L).setZero();
  gu.segment(dcon_B, dcon_L).setZero();

  // integral constraints are equalities
  gl.segment(qcon_B, qcon_L).setZero();
  gu.segment(qcon_B, qcon_L).setZero();

  // running constraints
  gl.segment(crcon_B, crcon_L) = ocp.crl.replicate(N, 1);
  gu.segment(crcon_B, crcon_L) = ocp.cru.replicate(N, 1);

  // end constraints
  gl.segment(cecon_B, cecon_L) = ocp.cel;
  gu.segment(cecon_B, cecon_L) = ocp.ceu;

  return NLP{
    .n       = n,
    .m       = m,
    .f       = std::move(f),
    .xl      = std::move(xl),
    .xu      = std::move(xu),
    .g       = std::move(g),
    .gl      = std::move(gl),
    .gu      = std::move(gu),
    .df_dx   = std::move(df_dx),
    .dg_dx   = std::move(dg_dx),
    .d2f_dx2 = {},
    .d2g_dx2 = {},
  };
}

}  // namespace smooth::feedback

// solve_ocp_ipopt: convert inf to 2e19...
