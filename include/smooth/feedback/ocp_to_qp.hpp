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

#ifndef SMOOTH__FEEDBACK__LINEAR_OCP_HPP_
#define SMOOTH__FEEDBACK__LINEAR_OCP_HPP_

#include <Eigen/Core>

#include <smooth/compat/autodiff.hpp>
#include <smooth/diff.hpp>
#include <smooth/lie_group.hpp>

#include "ocp.hpp"
#include "qp.hpp"

namespace smooth::feedback {

inline QuadraticProgramSparse<double> ocp_to_qp(
  const OCPType auto & ocp, const MeshType auto & mesh, double tf, auto && xl_fun, auto && ul_fun)
{
  using X = typename std::decay_t<decltype(ocp)>::X;
  using U = typename std::decay_t<decltype(ocp)>::U;

  using Mat = Eigen::MatrixXd;
  using Vec = Eigen::VectorXd;

  const X xl0 = xl_fun(0.);
  const X xlf = xl_fun(tf);
  const Vec ql(ocp.nq);

  assert(ocp.nq == 1);

  // objective function linearization

  const auto [th, dth, d2th] = diff::dr<2>(ocp.theta, wrt(tf, xl0, xlf, ql));

  [[maybe_unused]] const Vec qo_tf = dth.segment(0, 1);
  const Vec qo_x0xf                = dth.segment(1, 2 * ocp.nx);
  const Vec qo_q                   = dth.segment(1 + 2 * ocp.nx, ocp.nq);

  [[maybe_unused]] const Mat Qo_tf = d2th.block(0, 0, 1, 1) / 2;
  const Mat Qo_x0xf                = d2th.block(1, 1, 2 * ocp.nx, 2 * ocp.nx) / 2;
  [[maybe_unused]] const Mat Qo_ql = d2th.block(1 + 2 * ocp.nx, 1 + 2 * ocp.nx, ocp.nq, ocp.nq) / 2;

  std::cout << "qo_tf: " << qo_tf.transpose() << std::endl;
  std::cout << "qo_x0xf: " << qo_x0xf.transpose() << std::endl;
  std::cout << "qo_q: " << qo_q.transpose() << std::endl;

  std::cout << "qo_tf: " << std::endl << Qo_tf << std::endl;
  std::cout << "qo_x0xf: " << std::endl << Qo_x0xf << std::endl;
  std::cout << "qo_q: " << std::endl << Qo_ql << std::endl;

  const auto N = mesh.N_colloc();

  /////////////////////////
  //// VARIABLE LAYOUT ////
  /////////////////////////

  // [x0 x1 ... xN u0 u1 ... uN-1]

  const auto xvar_L = ocp.nx * (N + 1);
  const auto uvar_L = ocp.nu * N;

  const auto xvar_B = 0u;
  const auto uvar_B = xvar_L;

  const auto dcon_L  = ocp.nx * N;
  const auto crcon_L = ocp.ncr * N;
  const auto cecon_L = ocp.nce;

  const auto dcon_B  = 0u;
  const auto crcon_B = dcon_L;
  const auto cecon_B = crcon_B + crcon_L;

  const auto Nvar = xvar_L + uvar_L;
  const auto Ncon = dcon_L + crcon_L + cecon_L;

  /////////////////////////////////
  //// COLLOCATION CONSTRAINTS ////
  /////////////////////////////////

  Eigen::SparseMatrix<double> Adyn_X(dcon_L, xvar_L);
  Eigen::SparseMatrix<double> Adyn_U(dcon_L, uvar_L);
  Eigen::VectorXd bdyn(dcon_L);

  for (auto ival = 0u, M = 0u; ival < mesh.N_ivals(); M += mesh.N_colloc_ival(ival), ++ival) {
    const auto Nival        = mesh.N_colloc_ival(ival);               // number of nodes in interval
    const auto [tau_s, w_s] = mesh.interval_nodes_and_weights(ival);  // node values (plus 1)

    const Mat D = mesh.interval_diffmat(ival);

    // in each interval the collocation constraint is
    //
    // [A0 x0 ... Ak-1 xk-1 0]  + [B0 u0 ... Bk-1 uk-1] + [E0 ... Ek-1] = X D

    for (auto i = 0u; i < Nival; ++i) {
      const auto tau_i         = tau_s[i];                       // scaled time
      const auto t_i           = tf * tau_i;                     // unscaled time
      const auto [xl_i, dxl_i] = diff::dr<1>(xl_fun, wrt(t_i));  // linearization trajectory
      const auto ul_i          = ul_fun(t_i);                    // linearization input

      // LINEARIZE DYNAMICS
      const auto [f_i, df_i] = diff::dr<1>(ocp.f, wrt(t_i, xl_i, ul_i));

      const Mat A = -0.5 * ad<X>(f_i) - 0.5 * ad<X>(dxl_i) + df_i.middleCols(1, ocp.nx);
      const Mat B = df_i.middleCols(1 + ocp.nx, ocp.nu);
      const Vec E = f_i - dxl_i;

      // insert new constraint A xi + B ui + E = [x0 ... XNi] di

      for (auto j = 0u; j < Nival + 1; ++j) {
        for (auto diag = 0u; diag < ocp.nx; ++diag) {
          Adyn_X.coeffRef((M + i) * ocp.nx + diag, (M + j) * ocp.nx + diag) -= D(j, i);
        }
      }

      for (auto row = 0u; row < ocp.nx; ++row) {
        for (auto col = 0u; col < ocp.nx; ++col) {
          Adyn_X.coeffRef((M + i) * ocp.nx + row, (M + i) * ocp.nx + col) = A(row, col);
        }
        for (auto col = 0u; col < ocp.nu; ++col) {
          Adyn_U.coeffRef((M + i) * ocp.nx + row, (M + i) * ocp.nu + col) = B(row, col);
        }
      }
      bdyn.segment((M + i) * ocp.nx, ocp.nx) = -E;
    }
  }

  const auto [nodes, weights] = mesh.all_nodes_and_weights();

  auto xslin = nodes | std::views::transform([&xl_fun](double t) { return xl_fun(t); });
  auto uslin =
    nodes | std::views::take(N) | std::views::transform([&ul_fun](double t) { return ul_fun(t); });

  const Eigen::VectorXd q_dummy{{1.}};

  //////////////////////
  //// RUNNING COST ////
  //////////////////////

  const auto [gval, dg_dt0, dg_dtf, dg_dI, dg_dx, dg_du] =
    colloc_int<true>(1, ocp.g, mesh, 0., tf, q_dummy, xslin, uslin);

  // TODO need second derivative (id for now)
  Eigen::SparseMatrix<double> d2g_dx2 = sparse_identity(ocp.nx * (N + 1));
  Eigen::SparseMatrix<double> d2g_dxdu(ocp.nx * (N + 1), ocp.nu * N);
  Eigen::SparseMatrix<double> d2g_du2 = sparse_identity(ocp.nu * N);

  /////////////////////////////
  //// RUNNING CONSTRAINTS ////
  /////////////////////////////

  const auto [crlin, dcrlin_dt0, dcrlin_dtf, dcrlin_dx, dcrlin_du] =
    colloc_eval<true>(ocp.ncr, ocp.cr, mesh, 0., tf, xslin, uslin);

  /////////////////////////
  //// END CONSTRAINTS ////
  /////////////////////////

  const auto [celin, dcelin_dt0, dcelin_dtf, dcelin_dx, dcelin_dq] =
    colloc_eval_endpt<true>(ocp.nce, ocp.nx, ocp.ce, 0., tf, xslin, q_dummy);

  /////////////////////
  //// ASSEMBLE QP ////
  /////////////////////

  Eigen::SparseMatrix<double> P = sparse_block_matrix({
    {d2g_dx2, d2g_dxdu},
    {d2g_dxdu.transpose(), d2g_du2},
  });

  Eigen::VectorXd q(Nvar);
  q.segment(xvar_B, xvar_L) = Eigen::RowVectorXd(dg_dx).transpose();
  q.segment(uvar_B, uvar_L) = Eigen::RowVectorXd(dg_du).transpose();

  Eigen::SparseMatrix<double> A = sparse_block_matrix({
    {Adyn_X, Adyn_U},
    {dcrlin_dx, dcrlin_du},
    {dcelin_dx, {}},
  });

  Eigen::VectorXd l(Ncon), u(Ncon);

  l.segment(dcon_B, dcon_L)   = bdyn;
  l.segment(crcon_B, crcon_L) = ocp.crl.replicate(N, 1) - crlin.reshaped();
  l.segment(cecon_B, cecon_L) = ocp.cel - celin;

  u.segment(dcon_B, dcon_L)   = bdyn;
  u.segment(crcon_B, crcon_L) = ocp.cru.replicate(N, 1) - crlin.reshaped();
  u.segment(cecon_B, cecon_L) = ocp.ceu - celin;

  return QuadraticProgramSparse<double>{
    .P = std::move(P),
    .q = std::move(q),
    .A = std::move(A),
    .l = std::move(l),
    .u = std::move(u),
  };
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__LINEAR_OCP_HPP_
