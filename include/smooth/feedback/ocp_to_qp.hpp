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
  const Eigen::Matrix<double, 1, 1> ql(0);

  assert(ocp.nq == 1);

  // objective function linearization

  const auto [th, dth, d2th] = diff::dr<2>(ocp.theta, wrt(tf, xl0, xlf, ql));

  [[maybe_unused]] const Vec qo_tf = dth.segment(0, 1);
  const Vec qo_x0                  = dth.segment(1, ocp.nx);
  const Vec qo_xf                  = dth.segment(1 + ocp.nx, ocp.nx);
  const Vec qo_q                   = dth.segment(1 + 2 * ocp.nx, ocp.nq);

  [[maybe_unused]] const Mat Qo_tf = d2th.block(0, 0, 1, 1) / 2;
  const Mat Qo_x0                  = d2th.block(1, 1, ocp.nx, ocp.nx) / 2;
  const Mat Qo_x0f                 = d2th.block(1, 1 + ocp.nx, ocp.nx, ocp.nx) / 2;
  const Mat Qo_xf                  = d2th.block(1 + ocp.nx, 1 + ocp.nx, ocp.nx, ocp.nx) / 2;
  [[maybe_unused]] const Mat Qo_ql = d2th.block(1 + 2 * ocp.nx, 1 + 2 * ocp.nx, ocp.nq, ocp.nq) / 2;

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

  std::cout << "Collocation constraints\n\n";

  Eigen::SparseMatrix<double, Eigen::RowMajor> Adyn_X(dcon_L, xvar_L);
  Eigen::SparseMatrix<double, Eigen::RowMajor> Adyn_U(dcon_L, uvar_L);
  Eigen::VectorXd bdyn(dcon_L);

  // TODO reserve nonzeros...

  for (auto ival = 0u, M = 0u; ival < mesh.N_ivals(); M += mesh.N_colloc_ival(ival), ++ival) {
    const auto Nival        = mesh.N_colloc_ival(ival);               // number of nodes in interval
    const auto [tau_s, w_s] = mesh.interval_nodes_and_weights(ival);  // node values (plus 1)

    const Mat D = mesh.interval_diffmat(ival);

    // in each interval the collocation constraint is
    //
    // [A0 x0 ... Ak-1 xk-1 0]  + [B0 u0 ... Bk-1 uk-1] + [E0 ... Ek-1] = X D

    for (const auto & [i, tau_i] :
         smooth::utils::zip(std::views::iota(0u, Nival), mesh.interval_nodes_range(ival))) {
      const auto t_i           = tf * tau_i;                     // unscaled time
      const auto [xl_i, dxl_i] = diff::dr<1>(xl_fun, wrt(t_i));  // linearization trajectory
      const auto ul_i          = ul_fun(t_i);                    // linearization input

      // LINEARIZE DYNAMICS
      const auto [f_i, df_i] = diff::dr<1>(ocp.f, wrt(t_i, xl_i, ul_i));

      const Mat A = tf * (-0.5 * ad<X>(f_i) - 0.5 * ad<X>(dxl_i) + df_i.middleCols(1, ocp.nx));
      const Mat B = tf * df_i.middleCols(1 + ocp.nx, ocp.nu);
      const Vec E = tf * (f_i - dxl_i);

      // insert new constraint A xi + B ui + E = [x0 ... XNi] di

      for (auto j = 0u; j < Nival + 1; ++j) {
        for (auto diag = 0u; diag < ocp.nx; ++diag) {
          Adyn_X.coeffRef((M + i) * ocp.nx + diag, (M + j) * ocp.nx + diag) -= D(j, i);
        }
      }

      for (auto row = 0u; row < ocp.nx; ++row) {
        for (auto col = 0u; col < ocp.nx; ++col) {
          Adyn_X.coeffRef((M + i) * ocp.nx + row, (M + i) * ocp.nx + col) += A(row, col);
        }
        for (auto col = 0u; col < ocp.nu; ++col) {
          Adyn_U.coeffRef((M + i) * ocp.nx + row, (M + i) * ocp.nu + col) = B(row, col);
        }
      }

      bdyn.segment((M + i) * ocp.nx, ocp.nx) = -E;
    }
  }

  auto xslin =
    mesh.all_nodes_range() | std::views::transform([&xl_fun](double t) { return xl_fun(t); });
  auto uslin = mesh.all_nodes_range() | std::views::take(int64_t(N))
             | std::views::transform([&ul_fun](double t) { return ul_fun(t); });

  const Eigen::Vector<double, 1> q_dummy{1.};

  //////////////////
  //// INTEGRAL ////
  //////////////////

  std::cout << "Running cost\n";

  CollocEvalReduceResult g_res(1, ocp.nx, ocp.nu, N);
  colloc_integrate<2>(g_res, ocp.g, mesh, 0., tf, xslin, uslin);

  /////////////////////////////
  //// RUNNING CONSTRAINTS ////
  /////////////////////////////

  std::cout << "Running constr\n";

  CollocEvalResult CRres(ocp.ncr, ocp.nx, ocp.nu, N);
  colloc_eval<true>(CRres, ocp.cr, mesh, 0., tf, xslin, uslin);

  /////////////////////////
  //// END CONSTRAINTS ////
  /////////////////////////

  std::cout << "End constr\n";

  const auto [celin, dcelin_dt0, dcelin_dtf, dcelin_dx, dcelin_dq] =
    colloc_eval_endpt<true>(ocp.nce, ocp.nx, ocp.ce, 0., tf, xslin, q_dummy);

  // integral constraints not supported
  assert(Eigen::VectorXd(dcelin_dq).cwiseAbs().maxCoeff() < 1e-9);

  /////////////////////
  //// ASSEMBLE QP ////
  /////////////////////

  std::cout << "Assemble\n";

  // part from integrals
  Eigen::SparseMatrix<double> P = sparse_block_matrix({
    {g_res.d2F_dXX, g_res.d2F_dXU},
    {{}, g_res.d2F_dUU},
  });
  P *= qo_q.x() / 2;

  Eigen::VectorXd q(Nvar);
  q.segment(xvar_B, xvar_L) = qo_q.x() * g_res.dF_dX.transpose();
  q.segment(uvar_B, uvar_L) = qo_q.x() * g_res.dF_dU.transpose();

  // weights from x0 and xf
  q.segment(0, ocp.nx) += qo_x0;
  q.segment(ocp.nx * N, ocp.nx) += qo_xf;

  for (auto row = 0u; row < ocp.nx; ++row) {
    for (auto col = 0u; col < ocp.nx; ++col) {
      P.coeffRef(row, col) += Qo_x0(row, col);
      P.coeffRef(row, ocp.nx * N + col) += Qo_x0f(row, col);  // upper-triangular
      P.coeffRef(ocp.nx * N + row, ocp.nx * N + col) += Qo_xf(row, col);
    }
  }

  // TODO: this should be row-major...
  Eigen::SparseMatrix<double> A = sparse_block_matrix({
    {Adyn_X, Adyn_U},
    {CRres.dF_dX, CRres.dF_dU},
    {dcelin_dx, {}},
  });

  Eigen::VectorXd l(Ncon), u(Ncon);

  l.segment(dcon_B, dcon_L)   = bdyn;
  l.segment(crcon_B, crcon_L) = ocp.crl.replicate(N, 1) - CRres.F.reshaped();
  l.segment(cecon_B, cecon_L) = ocp.cel - celin;

  u.segment(dcon_B, dcon_L)   = bdyn;
  u.segment(crcon_B, crcon_L) = ocp.cru.replicate(N, 1) - CRres.F.reshaped();
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
