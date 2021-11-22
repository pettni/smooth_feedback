// smooth_feedback: Control theory on Lie groups
// https://github.com/pettni/smooth_feedback
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson, John B. Mains
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

#ifndef SMOOTH__FEEDBACK__OCP_HPP_
#define SMOOTH__FEEDBACK__OCP_HPP_

/**
 * @file
 * @brief Optimal control problem definition and conversion to QP.
 */

#include <Eigen/Core>

#include "collocation.hpp"
#include "nlp.hpp"

namespace smooth::feedback {

/**
 * @brief Rn optimal control problem on the interval \f$ t \in [0, t_f] \f$.
 * \f[
 * \begin{cases}
 *  \min_{t_f, u(\cdot), x(\cdot)} & \theta(t_f, x_0, x_f, q)                                        \\
 *  \text{s.t.}                    & x(0) = x_0                                                      \\
 *                                 & x(t_f) = x_f                                                    \\
 *                                 & \dot x(t) = f(t, x(t), u(t))                                    \\
 *                                 & q = \int_{0}^{t_f} g(t, x(t), u(t)) \mathrm{d}t                 \\
 *                                 & c_{rl} \leq c_r(t, x(t), u(t)) \leq c_{ru} \quad t \in [0, t_f] \\
 *                                 & c_{el} \leq c_e(t_f, x_0, x_f, q) \leq c_{eu}
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

  /// @brief Objective function \f$ \theta : R \times R^{n_x} \times R^{n_x} \times R^{n_q}
  /// \rightarrow R \f$
  Theta theta;

  /// @brief System dynamics \f$ f : R \times R^{n_x} \times R^{n_u} \rightarrow R^{n_x} \f$
  F f;
  /// @brief Integrals \f$ g : R \times R^{n_x} \times R^{n_u} \rightarrow R^{n_q} \f$
  G g;

  /// @brief Running constraint \f$ c_r : R \times R^{n_x} \times R^{n_u} \rightarrow R^{n_{cr}} \f$
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
 * @brief Solution to OCP problem.
 */
struct OCPSolution
{
  double t0;
  double tf;

  Eigen::VectorXd Q;
  Eigen::MatrixXd U;
  Eigen::MatrixXd X;
};

namespace detail {

/// @brief Variable and constraint structure of an OCP NLP
template<typename Theta,
  typename F,
  typename G,
  typename CR,
  typename CE,
  std::size_t Kmin,
  std::size_t Kmax>
auto ocp_nlp_structure(const OCP<Theta, F, G, CR, CE> & ocp, const Mesh<Kmin, Kmax> & mesh)
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

  return std::make_tuple(var_beg, var_len, con_beg, con_len);
}

}  // namespace detail

/**
 * @brief Formulate an OCP as a NLP using collocation on a mesh.
 *
 * @param ocp Optimal control problem definition
 * @param mesh Mesh that describes collocation point structure
 * @param NLP encoding of ocp as a nonlinear program
 */
template<typename Theta,
  typename F,
  typename G,
  typename CR,
  typename CE,
  std::size_t Kmin,
  std::size_t Kmax>
NLP ocp_to_nlp(const OCP<Theta, F, G, CR, CE> & ocp, const Mesh<Kmin, Kmax> & mesh)
{
  const auto [var_beg, var_len, con_beg, con_len] = detail::ocp_nlp_structure(ocp, mesh);

  // OBJECTIVE FUNCTION

  auto f = [var_beg = var_beg, var_len = var_len, ocp = ocp](const Eigen::VectorXd & x) -> double {
    const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
    const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

    assert(std::size_t(x.size()) == n);

    const double t0         = 0;
    const double tf         = x(tfvar_B);
    const Eigen::VectorXd Q = x.segment(qvar_B, qvar_L);
    const Eigen::MatrixXd X = x.segment(xvar_B, xvar_L).reshaped(ocp.nx, xvar_L / ocp.nx);

    return colloc_eval_endpt<false>(1, ocp.nx, ocp.theta, t0, tf, X, Q);
  };

  // OBJECTIVE JACOBIAN

  auto df_dx = [var_beg = var_beg, var_len = var_len, ocp = ocp](
                 const Eigen::VectorXd & x) -> Eigen::SparseMatrix<double> {
    const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
    const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

    const double t0         = 0;
    const double tf         = x(tfvar_B);
    const Eigen::VectorXd Q = x.segment(qvar_B, qvar_L);
    const Eigen::MatrixXd X = x.segment(xvar_B, xvar_L).reshaped(ocp.nx, xvar_L / ocp.nx);

    auto [fval, df_dt0, df_dtf, df_dvecX, df_dQ] =
      colloc_eval_endpt<true>(1, ocp.nx, ocp.theta, t0, tf, X, Q);

    return sparse_block_matrix({{df_dtf, df_dQ, df_dvecX, Eigen::SparseMatrix<double>(1, uvar_L)}});
  };

  // CONSTRAINT FUNCTION

  auto g = [var_beg  = var_beg,
             var_len = var_len,
             con_beg = con_beg,
             con_len = con_len,
             mesh    = mesh,
             ocp     = ocp](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
    const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

    const auto [dcon_B, qcon_B, crcon_B, cecon_B, m] = con_beg;
    const auto [dcon_L, qcon_L, crcon_L, cecon_L]    = con_len;

    assert(std::size_t(x.size()) == n);

    const double t0         = 0;
    const double tf         = x(tfvar_B);
    const Eigen::VectorXd Q = x.segment(qvar_B, qvar_L);
    const Eigen::MatrixXd X = x.segment(xvar_B, xvar_L).reshaped(ocp.nx, xvar_L / ocp.nx);
    const Eigen::MatrixXd U = x.segment(uvar_B, uvar_L).reshaped(ocp.nu, uvar_L / ocp.nu);

    Eigen::VectorXd ret(m);
    // clang-format off
    ret.segment(dcon_B, dcon_L)  = colloc_dyn<false>(ocp.nx, ocp.f, mesh, t0, tf, X, U);
    ret.segment(qcon_B, qcon_L)  = colloc_int<false>(ocp.nq, ocp.g, mesh, t0, tf, Q, X, U).reshaped();
    ret.segment(crcon_B, crcon_L) = colloc_eval<false>(ocp.ncr, ocp.cr, mesh, t0, tf, X, U).reshaped();
    ret.segment(cecon_B, cecon_L) = colloc_eval_endpt<false>(ocp.nce, ocp.nx, ocp.ce, t0, tf, X, Q).reshaped();
    // clang-format on
    return ret;
  };

  // CONSTRAINT JACOBIAN
  auto dg_dx = [var_beg = var_beg, var_len = var_len, mesh = mesh, ocp = ocp](
                 const Eigen::VectorXd & x) {
    const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
    const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

    assert(std::size_t(x.size()) == n);

    const double t0          = 0;
    const double tf          = x(tfvar_B);
    const Eigen::VectorXd Qm = x.segment(qvar_B, qvar_L);
    const Eigen::MatrixXd Xm = x.segment(xvar_B, xvar_L).reshaped(ocp.nx, xvar_L / ocp.nx);
    const Eigen::MatrixXd Um = x.segment(uvar_B, uvar_L).reshaped(ocp.nu, uvar_L / ocp.nu);

    // clang-format off
    const auto [Fval, dF_dt0, dF_dtf, dF_dX, dF_dU]        = colloc_dyn<true>(ocp.nx, ocp.f, mesh, t0, tf, Xm, Um);
    const auto [Gval, dG_dt0, dG_dtf, dG_dQ, dG_dX, dG_dU] = colloc_int<true>(ocp.nq, ocp.g, mesh, t0, tf, Qm, Xm, Um);
    const auto [CRval, dCR_dt0, dCR_dtf, dCR_dX, dCR_dU]   = colloc_eval<true>(ocp.ncr, ocp.cr, mesh, t0, tf, Xm, Um);
    const auto [CEval, dCE_dt0, dCE_dtf, dCE_dX, dCE_dQ]   = colloc_eval_endpt<true>(ocp.nce, ocp.nx, ocp.ce, t0, tf, Xm, Qm);
    // clang-format on

    return sparse_block_matrix({
      // clang-format off
      {dF_dtf,      {},  dF_dX,  dF_dU},
      {dG_dtf,   dG_dQ,  dG_dX,  dG_dU},
      {dCR_dtf,     {}, dCR_dX, dCR_dU},
      {dCE_dtf, dCE_dQ, dCE_dX,     {}},
      // clang-format on
    });
  };

  const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
  const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

  const auto [dcon_B, qcon_B, crcon_B, cecon_B, m] = con_beg;
  const auto [dcon_L, qcon_L, crcon_L, cecon_L]    = con_len;

  // VARIABLE BOUNDS

  Eigen::VectorXd xl = Eigen::VectorXd::Constant(n, -std::numeric_limits<double>::infinity());
  Eigen::VectorXd xu = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());

  xl.segment(tfvar_B, tfvar_L).setZero();  // tf lower bounded by zero

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
  gl.segment(crcon_B, crcon_L) = ocp.crl.replicate(mesh.N_colloc(), 1);
  gu.segment(crcon_B, crcon_L) = ocp.cru.replicate(mesh.N_colloc(), 1);

  // end constraints
  gl.segment(cecon_B, cecon_L) = ocp.cel;
  gu.segment(cecon_B, cecon_L) = ocp.ceu;

  return {
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

template<typename Theta,
  typename F,
  typename G,
  typename CR,
  typename CE,
  std::size_t Kmin,
  std::size_t Kmax>
OCPSolution nlp_sol_to_ocp_sol(
  const OCP<Theta, F, G, CR, CE> & ocp, const Mesh<Kmin, Kmax> & mesh, const NLPSolution & nlp_sol)
{
  const auto [var_beg, var_len, con_beg, con_len] = detail::ocp_nlp_structure(ocp, mesh);

  const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
  const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

  OCPSolution ret;
  ret.t0 = 0;
  ret.tf = nlp_sol.x(tfvar_B);

  ret.Q = nlp_sol.x.segment(qvar_B, qvar_L);
  ret.X = nlp_sol.x.segment(xvar_B, xvar_L).reshaped(ocp.nx, xvar_L / ocp.nx);
  ret.U = nlp_sol.x.segment(uvar_B, uvar_L).reshaped(ocp.nu, uvar_L / ocp.nu);

  return ret;
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__OCP_HPP_
