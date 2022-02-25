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

#ifndef SMOOTH__FEEDBACK__OCP_TO_QP_HPP_
#define SMOOTH__FEEDBACK__OCP_TO_QP_HPP_

/**
 * @file
 * @brief Formulate optimal control problem as a quadratic program
 */

#include <Eigen/Core>
#include <smooth/diff.hpp>

#include "collocation/mesh.hpp"
#include "collocation/mesh_function.hpp"
#include "ocp.hpp"
#include "qp.hpp"

namespace smooth::feedback {

namespace detail {

/**
 * @brief Working memory for ocp_to_qp
 */
struct OcpToQpWorkmemory
{
  MeshValue<1> cr_out;   /// @brief output of mesh_eval
  MeshValue<2> int_out;  /// @brief output of mesh_integrate
};

/**
 * @brief Allocate a qp for ocp_to_qp_update()
 *
 * @param[out] qp quadratic program to allocate
 * @param[out] work working memory to allocate
 * @param[in] ocp input problem
 * @param[in] mesh time discretization
 */
template<diff::Type DT = diff::Type::Default>
void ocp_to_qp_allocate(
  QuadraticProgramSparse<double> & qp,
  OcpToQpWorkmemory & work,
  OCPType auto & ocp,
  const MeshType auto & mesh)
{
  using ocp_t = typename std::decay_t<decltype(ocp)>;

  using X = typename ocp_t::X;
  using U = typename ocp_t::U;

  static constexpr auto Nx = ocp_t::Nx;
  static constexpr auto Nu = ocp_t::Nu;

  /////////////////////////
  //// VARIABLE LAYOUT ////
  /////////////////////////

  // [x0 x1 ... xN u0 u1 ... uN-1]

  const auto N = mesh.N_colloc();

  const auto xvar_L = Nx * (N + 1);
  const auto uvar_L = Nu * N;

  const auto dcon_L  = Nx * N;
  const auto crcon_L = ocp_t::Ncr * N;
  const auto cecon_L = ocp_t::Nce;

  const auto dcon_B  = 0u;
  const auto crcon_B = dcon_L;
  const auto cecon_B = crcon_B + crcon_L;

  const auto Nvar = xvar_L + uvar_L;
  const auto Ncon = dcon_L + crcon_L + cecon_L;

  // resize qp
  qp.P.resize(Nvar, Nvar);
  qp.q.resize(Nvar);
  qp.A.resize(Ncon, Nvar);
  qp.l.resize(Ncon);
  qp.u.resize(Ncon);

  // sparsity pattern of A
  Eigen::VectorXi A_pattern = Eigen::VectorXi::Zero(Ncon);
  for (auto ival = 0u, I0 = 0u; ival < mesh.N_ivals(); I0 += mesh.N_colloc_ival(ival), ++ival) {
    const auto Ki = mesh.N_colloc_ival(ival);  // number of nodes in interval
    A_pattern.segment(dcon_B + I0 * Nx, Ki * Nx) +=
      Eigen::VectorXi::Constant(Ki * Nx, Nx + Ki + Nu);
  }
  A_pattern.segment(crcon_B, crcon_L).setConstant(Nx + Nu);
  A_pattern.segment(cecon_B, cecon_L).setConstant(2 * Nx);
  qp.A.reserve(A_pattern);

  // sparsity pattern of P
  Eigen::VectorXi P_pattern = Eigen::VectorXi::Zero(Nvar);
  for (auto i = 0u; i < N; ++i) {
    for (auto j = 0u; j < Nx; ++j) { P_pattern(Nx * i + j) = j + 1; }
  }
  for (auto j = 0u; j < Nx; ++j) { P_pattern(Nx * N + j) = Nx + j + 1; }
  for (auto i = 0u; i < N; ++i) {
    for (auto j = 0u; j < Nu; ++j) { P_pattern(Nx * (N + 1) + Nu * i + j) = Nx + (j + 1); }
  }
  qp.P.reserve(P_pattern);

  // compute work stuff once to allocate pattern
  const double tf = 1.;
  auto xslin      = mesh.all_nodes() | transform([&](double) { return Identity<X>(); });
  auto uslin      = mesh.all_nodes() | transform([&](double) { return Identity<U>(); });

  work.int_out.lambda.setConstant(1, 1);
  mesh_eval<1, DT>(work.cr_out, mesh, ocp.cr, 0, tf, xslin, uslin);       // allocates work.cr_out
  mesh_integrate<2, DT>(work.int_out, mesh, ocp.g, 0, tf, xslin, uslin);  // allocates work.int_out
}

/**
 * @brief Formulate an optimal control problem as a quadratic program via linearization.
 *
 * @tparam DT differentiation method to use for derivatives of the ocp
 * @tparam DT_xl differentiation method to use for derivatives of xl_fun
 *
 * @param[in, out] qp pre-allocated quadratic program
 * @param[in, out] work workin memory
 * @param[in] ocp input problem
 * @param[in] mesh time discretization
 * @param[in] tf time horizon
 * @param[in] xl_fun state linearization (must be differentiable w.r.t. time)
 * @param[in] ul_fun input linearization
 *
 * @return sparse quadratic program for a flattened formulation of ocp.
 *
 * @see ocp_to_qp_allocate() qpsol_to_ocpsol()
 */
template<diff::Type DT = diff::Type::Default>
void ocp_to_qp_update(
  QuadraticProgramSparse<double> & qp,
  OcpToQpWorkmemory & work,
  OCPType auto & ocp,
  const MeshType auto & mesh,
  double tf,
  auto && xl_fun,
  auto && ul_fun)
{
  using utils::zip;
  using namespace std::views;

  using ocp_t = typename std::decay_t<decltype(ocp)>;

  using X = typename ocp_t::X;

  static constexpr auto Nx = ocp_t::Nx;
  static constexpr auto Nu = ocp_t::Nu;
  static constexpr auto Nq = ocp_t::Nq;

  const double t0 = 0.;

  const X xl0 = xl_fun(0.);
  const X xlf = xl_fun(tf);
  const Eigen::Matrix<double, 1, 1> ql(0);

  static_assert(ocp_t::Nq == 1, "exactly one integral supported in ocp_to_qp");

  ////////////////////////
  //// ZERO VARIABLES ////
  ////////////////////////

  set_zero(qp.A);
  set_zero(qp.P);

  qp.l.setConstant(-std::numeric_limits<double>::infinity());
  qp.u.setConstant(std::numeric_limits<double>::infinity());
  qp.q.setZero();

  /////////////////////////
  //// VARIABLE LAYOUT ////
  /////////////////////////

  // [x0 x1 ... xN u0 u1 ... uN-1]

  const auto N = mesh.N_colloc();

  const auto xvar_L = Nx * (N + 1);
  const auto uvar_L = Nu * N;

  const auto xvar_B = 0u;
  const auto uvar_B = xvar_L;

  const auto dcon_L  = Nx * N;
  const auto crcon_L = ocp_t::Ncr * N;
  const auto cecon_L = ocp_t::Nce;

  const auto dcon_B  = 0u;
  const auto crcon_B = dcon_L;
  const auto cecon_B = crcon_B + crcon_L;

  /////////////////////////////////
  //// OBJECTIVE LINEARIZATION ////
  /////////////////////////////////

  const auto & [th, dth, d2th] = diff::dr<2, DT>(ocp.theta, wrt(tf, xl0, xlf, ql));

  const Eigen::Vector<double, Nx> qo_x0 = dth.middleCols(1, Nx).transpose();
  const Eigen::Vector<double, Nx> qo_xf = dth.middleCols(1 + Nx, Nx).transpose();
  const Eigen::Vector<double, Nq> qo_q  = dth.middleCols(1 + 2 * Nx, Nq).transpose();

  const Eigen::Matrix<double, Nx, Nx> Qo_x0  = d2th.block(1, 1, Nx, Nx) / 2;
  const Eigen::Matrix<double, Nx, Nx> Qo_x0f = d2th.block(1, 1 + Nx, Nx, Nx) / 2;
  const Eigen::Matrix<double, Nx, Nx> Qo_xf  = d2th.block(1 + Nx, 1 + Nx, Nx, Nx) / 2;

  /////////////////////////////////
  //// COLLOCATION CONSTRAINTS ////
  /////////////////////////////////

  for (auto ival = 0u, M = 0u; ival < mesh.N_ivals(); M += mesh.N_colloc_ival(ival), ++ival) {
    const auto Ki = mesh.N_colloc_ival(ival);  // number of nodes in interval

    const auto [alpha, Dus] = mesh.interval_diffmat_unscaled(ival);

    // in each interval the collocation constraint is
    //
    // [A0 x0 ... Ak-1 xk-1 0]  + [B0 u0 ... Bk-1 uk-1] + [E0 ... Ek-1] = alpha * X Dus

    for (const auto & [i, tau_i] : zip(iota(0u, Ki), mesh.interval_nodes(ival))) {
      const auto t_i             = tf * tau_i;                         // unscaled time
      const auto & [xl_i, dxl_i] = diff::dr<1, DT>(xl_fun, wrt(t_i));  // x-lin
      const auto ul_i            = ul_fun(t_i);                        // u-lin

      // LINEARIZE DYNAMICS
      const auto & [f_i, df_i] = diff::dr<1, DT>(ocp.f, wrt(t_i, xl_i, ul_i));

      const Eigen::Matrix<double, Nx, Nx> A =
        tf * (-0.5 * ad<X>(f_i) - 0.5 * ad<X>(dxl_i) + df_i.template middleCols<Nx>(1));
      const Eigen::Matrix<double, Nx, Nu> B = tf * df_i.template middleCols<Nu>(1 + Nx);
      const Eigen::Vector<double, Nx> E     = tf * (f_i - dxl_i);

      // insert new constraint A xi + B ui + E = [x0 ... XNi] di

      for (auto j = 0u; j < Ki + 1; ++j) {
        for (auto diag = 0u; diag < Nx; ++diag) {
          qp.A.coeffRef(dcon_B + (M + i) * Nx + diag, (M + j) * Nx + diag) -= alpha * Dus(j, i);
        }
      }

      for (auto row = 0u; row < Nx; ++row) {
        for (auto col = 0u; col < Nx; ++col) {
          qp.A.coeffRef(dcon_B + (M + i) * Nx + row, (M + i) * Nx + col) += A(row, col);
        }
        for (auto col = 0u; col < Nu; ++col) {
          qp.A.coeffRef(dcon_B + (M + i) * Nx + row, uvar_B + (M + i) * Nu + col) = B(row, col);
        }
      }

      qp.l.segment(dcon_B + (M + i) * Nx, Nx) = -E;
      qp.u.segment(dcon_B + (M + i) * Nx, Nx) = -E;
    }
  }

  auto xslin = mesh.all_nodes() | transform([&](double t) { return xl_fun(t0 + (tf - t0) * t); });
  auto uslin = mesh.all_nodes() | transform([&](double t) { return ul_fun(t0 + (tf - t0) * t); });

  const Eigen::Vector<double, 1> qlin{1.};

  /////////////////////////////
  //// RUNNING CONSTRAINTS ////
  /////////////////////////////

  mesh_eval<1, DT>(work.cr_out, mesh, ocp.cr, 0, tf, xslin, uslin);

  block_add(qp.A, crcon_B, 0, work.cr_out.dF.middleCols(2, xvar_L + uvar_L));
  qp.l.segment(crcon_B, crcon_L) = ocp.crl.replicate(N, 1) - work.cr_out.F;
  qp.u.segment(crcon_B, crcon_L) = ocp.cru.replicate(N, 1) - work.cr_out.F;

  /////////////////////////
  //// END CONSTRAINTS ////
  /////////////////////////

  const auto & [ceval, dceval] = diff::dr<1, DT>(ocp.ce, wrt(tf, xl0, xlf, qlin));

  // integral constraints not supported
  // assert(dceval.middleCols(1 + 2 * Nx, Nq).cwiseAbs().maxCoeff() < 1e-9);

  block_add(qp.A, cecon_B, xvar_B, dceval.middleCols(1, Nx));                     // dce / dx0
  block_add(qp.A, cecon_B, xvar_B + xvar_L - Nx, dceval.middleCols(1 + Nx, Nx));  // dce / dxf

  qp.l.segment(cecon_B, cecon_L) = ocp.cel - ceval;
  qp.u.segment(cecon_B, cecon_L) = ocp.ceu - ceval;

  ///////////////////////
  //// INTEGRAL COST ////
  ///////////////////////

  mesh_integrate<2, DT>(work.int_out, mesh, ocp.g, 0, tf, xslin, uslin);

  block_add(
    qp.P, 0, 0, work.int_out.d2F.block(2, 2, xvar_L + uvar_L, xvar_L + uvar_L), qo_q.x(), true);
  qp.q.segment(xvar_B, xvar_L) = qo_q.x() * work.int_out.dF.middleCols(2, xvar_L).transpose();
  qp.q.segment(uvar_B, uvar_L) =
    qo_q.x() * work.int_out.dF.middleCols(2 + xvar_L, uvar_L).transpose();

  ///////////////////////
  //// ENDPOINT COST ////
  ///////////////////////

  block_add(qp.P, 0, 0, Qo_x0, 1., true);            // d2q / dx0x0
  block_add(qp.P, 0, Nx * N, Qo_x0f, 1., true);      // d2q / dx0xf
  block_add(qp.P, Nx * N, Nx * N, Qo_xf, 1., true);  // d2q / dxfxf
  qp.q.segment(0, Nx) += qo_x0;                      // dq / dx0
  qp.q.segment(Nx * N, Nx) += qo_xf;                 // dq / dxf
}

}  // namespace detail

/**
 * @brief Formulate an optimal control problem as a quadratic program via linearization.
 *
 * @param ocp input problem
 * @param mesh time discretization
 * @param tf time horizon
 * @param xl_fun state linearization (must be differentiable w.r.t. time)
 * @param ul_fun input linearization
 *
 * @return sparse quadratic program for a flattened formulation of ocp.
 *
 * @note allocates memory for each call. To reduce memory allocation for repeated calls, see
 * ocp_to_qp_allocate() and ocp_to_qp_update().
 *
 * @see qpsol_to_ocpsol()
 */
template<diff::Type DT = diff::Type::Default>
QuadraticProgramSparse<double> ocp_to_qp(
  const OCPType auto & ocp, const MeshType auto & mesh, double tf, auto && xl_fun, auto && ul_fun)
{
  QuadraticProgramSparse<double> qp;
  detail::OcpToQpWorkmemory work;

  detail::ocp_to_qp_allocate<DT>(qp, work, ocp, mesh);
  detail::ocp_to_qp_update<DT>(qp, work, ocp, mesh, tf, xl_fun, ul_fun);

  qp.A.makeCompressed();
  qp.P.makeCompressed();

  return qp;
}

/**
 * @brief Convert QP solution to OCP solution
 *
 * If qp_sol solves a QP obtained via ocp_to_qp(), then qpsol_to_ocpsol(qp_sol)
 * is the corrensponding OCP solution.
 *
 * @param ocp optimal control problem
 * @param mesh discretization mesh
 * @param qpsol solution to quadratic program obtained via ocp_to_qp()
 * @param tf final time used in ocp_to_qp()
 * @param xl_fun state linearization trajectory used in ocp_to_qp()
 * @param ul_fun input linearization trajectory used in ocp_to_qp()
 *
 * @see ocp_to_qp()
 */
auto qpsol_to_ocpsol(
  const OCPType auto & ocp,
  const MeshType auto & mesh,
  const QPSolution<-1, -1, double> & qpsol,
  double tf,
  auto && xl_fun,
  auto && ul_fun)
{
  using ocp_t = std::decay_t<decltype(ocp)>;

  using X = typename ocp_t::X;
  using U = typename ocp_t::U;

  static constexpr int Nx  = ocp_t::Nx;
  static constexpr int Nu  = ocp_t::Nu;
  static constexpr int Nq  = ocp_t::Nq;
  static constexpr int Ncr = ocp_t::Ncr;
  static constexpr int Nce = ocp_t::Nce;

  const auto N = mesh.N_colloc();

  const auto xvar_L = Nx * (N + 1);
  const auto uvar_L = Nu * N;

  const auto xvar_B    = 0u;
  const auto uvar_B    = xvar_L;
  Eigen::MatrixXd Xmat = qpsol.primal.segment(xvar_B, xvar_L).reshaped(Nx, N + 1);
  Eigen::MatrixXd Umat = qpsol.primal.segment(uvar_B, uvar_L).reshaped(Nu, N);

  auto xfun = [t0     = 0.,
               tf     = tf,
               mesh   = mesh,
               Xmat   = std::move(Xmat),
               xl_fun = std::forward<decltype(xl_fun)>(xl_fun)](double t) -> X {
    const auto tngnt =
      mesh.template eval<Eigen::Vector<double, Nx>>((t - t0) / (tf - t0), Xmat.colwise(), 0, true);
    return rplus(xl_fun(t), tngnt);
  };

  auto ufun = [t0     = 0.,
               tf     = tf,
               mesh   = mesh,
               Umat   = std::move(Umat),
               ul_fun = std::forward<decltype(ul_fun)>(ul_fun)](double t) -> U {
    const auto tngnt =
      mesh.template eval<Eigen::Vector<double, Nu>>((t - t0) / (tf - t0), Umat.colwise(), 0, false);
    return rplus(ul_fun(t), tngnt);
  };

  return OCPSolution<X, U, Nq, Nce, Ncr>{
    .t0 = 0.,
    .tf = tf,
    .u  = std::move(ufun),
    .x  = std::move(xfun),
  };
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__OCP_TO_QP_HPP_
