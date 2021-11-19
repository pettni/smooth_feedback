/**
 * @file
 * @brief Nonlinear programming
 */

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <limits>

#include "collocation.hpp"

/**
 * @brief Optimal control problem on the interval \f$ t \in [0, t_f] \f$.
 * \f[
 * \begin{cases}
 *  \min        & \int_{0}^{t_f} \phi(t, x(t), u(t)) \mathrm{d}t + \theta(t_f, x_0, x_f)  \\
 *  \text{s.t.} & x(0) = x_0                                                            \\
 *              & x(t_f) = x_f                                                          \\
 *              & \dot x(t) = f(t, x(t), u(t))                                          \\
 *              & q = \int_{0}^{t_f} g(t, x(t), u(t)) \mathrm{d}t                       \\
 *              & c_{rl} \leq c_r(t, x(t), u(t)) \leq c_{ru} \quad t \in [0, t_f]       \\
 *              & c_{el} \leq c_e(t_f, x_0, x_f, q) \leq c_{eu}                         \\
 * \end{cases}
 * \f]
 *
 * TODO make this into a concept...
 */
template<typename Phi, typename Theta, typename F, typename G, typename CR, typename CE>
struct OCP
{
  /// @brief State dimension
  std::size_t nx;
  /// @brief Input dimension
  std::size_t nu;
  /// @brief Number of additional integrals
  std::size_t nq;
  /// @brief Number of running constraints
  std::size_t ncr;
  /// @brief Number of end constraints
  std::size_t nce;

  /// @brief Objective integrand
  Phi phi;
  /// @brief Objective end function
  Theta theta;
  /// @brief Function defining system dynamics
  F f;
  /// @brief Integrals function
  G g;
  /// @brief Running constraint function
  CR cr;
  /// @brief Running constraint lower bound
  Eigen::VectorXd crl;
  /// @brief Running constraint upper bound
  Eigen::VectorXd cru;
  /// @brief End constraint function
  CE ce;
  /// @brief End constraint lower
  Eigen::VectorXd cel;
  /// @brief End constraint upper bound
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

  /// @brief Hessian of objective function (R^n -> R^{n x n})
  std::optional<std::function<Eigen::SparseMatrix<double>(Eigen::VectorXd, Eigen::VectorXd)>>
    d2f_dx2 = std::nullopt;

  /**
   * @brief Projected Hessian of constraint function (R^m, R^n -> R^{n x n})
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
 */
template<typename OCP, typename MeshType>
auto ocp_to_nlp(OCP && ocp, const MeshType & mesh)
{
  std::size_t N = mesh.N_colloc();

  // variable layout
  std::array<std::size_t, 5> var_len{
    1,                 // tf
    1,                 // objective integral
    ocp.nq,            // other integrals
    ocp.nx * (N + 1),  // states
    ocp.nu * N,        // inputs
  };

  // constraint layout
  std::array<std::size_t, 5> con_len{
    ocp.nx * N,   // derivatives
    1,            // objective integral
    ocp.nq,       // other integrals
    ocp.ncr * N,  // running constraints
    ocp.nce,      // end constraints
  };

  std::array<std::size_t, 6> var_beg;
  var_beg[0] = 0;
  std::partial_sum(var_len.begin(), var_len.end(), var_beg.begin() + 1);

  std::array<std::size_t, 6> con_beg;
  con_beg[0] = 0;
  std::partial_sum(con_len.begin(), con_len.end(), con_beg.begin() + 1);

  const auto [tfvar_B, oqvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
  const auto [tfvar_L, oqvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

  const auto [dcon_B, oqcon_B, qcon_B, crcon_B, cecon_B, m] = con_beg;
  const auto [dcon_L, oqcon_L, qcon_L, crcon_L, cecon_L]    = con_len;

  // OBJECTIVE FUNCTION

  std::cout << "objective function" << std::endl;

  auto f = [var_beg, var_len, nx = ocp.nx, theta = ocp.theta](const Eigen::VectorXd & x) -> double {
    const auto [tfvar_B, oqvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
    const auto [tfvar_L, oqvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

    assert(x.size() == n);

    const double OQ          = x(oqvar_B);
    const double tf          = x(tfvar_B);
    const Eigen::VectorXd x0 = x.segment(xvar_B, nx);
    const Eigen::VectorXd xf = x.segment(xvar_B + xvar_L - nx, nx);

    return OQ + theta.template operator()<double>(tf, x0, xf);
  };

  // VARIABLE BOUNDS

  std::cout << "variable bounds" << std::endl;

  Eigen::VectorXd xl = Eigen::VectorXd::Constant(n, -std::numeric_limits<double>::infinity());
  Eigen::VectorXd xu = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());

  xl.segment(tfvar_B, tfvar_L).setZero();  // tf lower bounded by zero

  // CONSTRAINT FUNCTION

  std::cout << "constraint function" << std::endl;

  auto g = [var_beg, var_len, con_beg, con_len, mesh, ocp](
             const Eigen::VectorXd & x) -> Eigen::VectorXd {
    const auto [tfvar_B, oqvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
    const auto [tfvar_L, oqvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

    const auto [dcon_B, oqcon_B, qcon_B, crcon_B, cecon_B, m] = con_beg;
    const auto [dcon_L, oqcon_L, qcon_L, crcon_L, cecon_L]    = con_len;

    assert(x.size() == n);

    std::cout << "extract vars" << std::endl;

    const double t0           = 0;
    const double tf           = x(tfvar_B);
    const Eigen::VectorXd OQm = x.segment(oqvar_B, oqvar_L);
    const Eigen::VectorXd Qm  = x.segment(qvar_B, qvar_L);
    const Eigen::MatrixXd Xm  = x.segment(xvar_B, xvar_L).reshaped(ocp.nx, xvar_L / ocp.nx);
    const Eigen::MatrixXd Um  = x.segment(uvar_B, uvar_L).reshaped(ocp.nu, uvar_L / ocp.nu);

    const Eigen::VectorXd x0 = Xm.leftCols(1);
    const Eigen::VectorXd xf = Xm.rightCols(1);

    // dynamics constraint
    std::cout << "call dyn" << std::endl;
    const auto [Fval, dF_dt0, dF_dtf, dF_dX, dF_dU] =
      dynamics_constraint(ocp.f, mesh, t0, tf, Xm, Um);

    // objective integral constraint
    std::cout << "call integral" << std::endl;
    auto phi_wrap = [&ocp]<typename T>(const auto... vars) -> Eigen::VectorX<T> {
      const T val = ocp.phi.template operator()<T>(vars...);
      return Eigen::VectorX<T>::Constant(1, val);
    };
    const auto [G1val, dG1_dt0, dG1_dtf, dG1_dI, dG1_dX, dG1_dU] =
      integral_constraint(phi_wrap, mesh, t0, tf, OQm, Xm, Um);

    // other integral constraint
    std::cout << "call integral again" << std::endl;
    const auto [G2val, dG2_dt0, dG2_dtf, dG2_dI, dG2_dX, dG2_dU] =
      integral_constraint(ocp.g, mesh, t0, tf, Qm, Xm, Um);

    // running constraints
    std::cout << "call running" << std::endl;
    const auto [CRval, dCR_dt0, dCR_dtf, dCR_dX, dCR_dU] =
      colloc_eval(ocp.ncr, ocp.cr, mesh, t0, tf, Xm, Um);

    // end constraints
    std::cout << "call ceval" << std::endl;
    const Eigen::VectorXd CEval = ocp.ce.template operator()<double>(tf, x0, xf, Qm);

    std::cout << "fill" << std::endl;
    Eigen::VectorXd ret(m);
    ret.segment(dcon_B, dcon_L)   = Fval.reshaped();
    ret.segment(oqcon_B, oqcon_L) = G1val.reshaped();
    ret.segment(qcon_B, qcon_L)   = G2val.reshaped();
    ret.segment(crcon_B, crcon_L) = CRval.reshaped();
    ret.segment(cecon_B, cecon_L) = CEval.reshaped();
    return ret;
  };

  // CONSTRAINT BOUNDS

  std::cout << "constraint bounds" << std::endl;

  Eigen::VectorXd gl(m);
  Eigen::VectorXd gu(m);

  // derivative constraints are equality constraints
  gl.segment(dcon_B, dcon_L).setZero();
  gu.segment(dcon_B, dcon_L).setZero();

  // objective integral is an equality constraint
  gl.segment(oqcon_B, oqcon_L).setZero();
  gu.segment(oqcon_B, oqcon_L).setZero();

  // ... so are remaining integral constraints
  gl.segment(qcon_B, qcon_L).setZero();
  gu.segment(qcon_B, qcon_L).setZero();

  // running constraints
  std::cout << "running" << std::endl;
  gl.segment(crcon_B, crcon_L) = ocp.crl.replicate(N, 1);
  gu.segment(crcon_B, crcon_L) = ocp.cru.replicate(N, 1);
  std::cout << "/running" << std::endl;

  // end constraints
  std::cout << "end" << std::endl;
  gl.segment(cecon_B, cecon_L) = ocp.cel;
  gu.segment(cecon_B, cecon_L) = ocp.ceu;
  std::cout << "/end" << std::endl;

  // OBJECTIVE JACOBIAN
  auto df_dx = [n = n](const Eigen::VectorXd &) -> Eigen::SparseMatrix<double> {
    Eigen::SparseMatrix<double> ret(n, n);
    return ret;
  };

  // CONSTRAINT JACOBIAN
  auto dg_dx = [n = n, m = m](const Eigen::VectorXd &) -> Eigen::SparseMatrix<double> {
    Eigen::SparseMatrix<double> ret(m, n);
    return ret;
  };

  std::cout << "returning NLP" << std::endl;

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

// solve_ocp_ipopt: convert inf to 2e19...
