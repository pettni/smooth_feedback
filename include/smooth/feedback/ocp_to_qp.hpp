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

  // end constraint linearization

  const auto [ce, dce] = diff::dr<1>(ocp.ce, wrt(tf, xl0, xlf, ql));

  [[maybe_unused]] const Mat Ace_tf = dce.middleCols(0, 1);
  const Mat Ace_x0                  = dce.middleCols(1, ocp.nx);
  const Mat Ace_xf                  = dce.middleCols(1 + ocp.nx, ocp.nx);
  [[maybe_unused]] const Mat Ace_q  = dce.middleCols(1 + 2 * ocp.nx, ocp.nq);

  std::cout << "Ace_x0" << std::endl << Ace_x0 << std::endl;
  std::cout << "Ace_xf" << std::endl << Ace_xf << std::endl;

  // TODO add end constraints to QP

  // variable layout: [x0 x1 ... xN u0 u1 ... uN-1]

  const auto N_ivals = mesh.N_ivals();
  const auto N       = mesh.N_colloc();

  const auto Nx = ocp.nx * (N + 1);
  const auto Nu = ocp.nu * N;

  const auto Nvar = Nx + Nu;

  const auto Ncon = ocp.nce      // end constraints (inequality)
                  + N * ocp.ncr  // running constraints (inequality)
                  + N * ocp.nx;  // collocation (equality)

  QuadraticProgramSparse<double> ret;
  ret.P.resize(Nvar, Nvar);
  ret.q.resize(Nvar);

  ret.A.resize(Ncon, Nvar);
  ret.l.resize(Ncon);
  ret.u.resize(Ncon);

  // TODO: allocate nonzero pattern in P an A

  for (auto ival = 0u; ival < N_ivals; ++ival) {

    // let variables in interval be
    //     X = [xi0 xi1 ... xiN]
    //     U = [ui0 ui1 ... uiN-1]

    const auto Ni           = mesh.N_colloc_ival(ival);
    const auto [tau_s, w_s] = mesh.interval_nodes_and_weights(ival);

    /////////////////////////////////
    //// COLLOCATION CONSTRAINTS ////
    /////////////////////////////////

    // in each interval the collocation constraint is
    //
    // [A0 ... Ak-1 0 B0 ... Bk-1] [X; U] + [E0 ... Ek-1] = X D
    //           Ca                             Cb

    // form coefficient matrices with linearized system dynamics
    Mat Ca(ocp.nx, (Ni + 1) * ocp.nx + Ni * ocp.nu);
    Mat Cb(ocp.nx, Ni);

    for (auto i = 0u; i < Ni; ++i) {
      const auto tau_i         = tau_s[i];                       // scaled time
      const auto t_i           = tf * tau_i;                     // unscaled time
      const auto [xl_i, dxl_i] = diff::dr<1>(xl_fun, wrt(t_i));  // linearization trajectory
      const auto ul_i          = ul_fun(t_i);                    // linearization input

      // LINEARIZE RUNNING COST
      const auto [g, dg, d2g] =
        diff::dr<2>([&ocp](auto &&... x) { return ocp.g(x...).x(); }, wrt(t_i, xl_i, ul_i));

      [[maybe_unused]] const Vec g_ti = dg.segment(0, 1);
      const Vec g_xiui                = dg.segment(1, ocp.nx + ocp.nu);
      const Mat G_xiui                = d2g.block(1, 1, ocp.nx + ocp.nu, ocp.nx + ocp.nu);

      std::cout << "g_xiui " << g_xiui.transpose() << std::endl;
      std::cout << "G_xiui " << std::endl << G_xiui << std::endl;

      // TODO: add integral over interval of quadratic function to QP (need Lagrange basis etc...)

      // cost += w_i * [xi; ui] * G_xiui * [xi; ui] / 2 + g_xiui * [xi; ui]

      // LINEARIZE RUNNING CONSTRAINTS
      const auto [cr, dcr]              = diff::dr<1>(ocp.cr, wrt(t_i, xl_i, ul_i));
      [[maybe_unused]] const Mat Acr_ti = dcr.middleCols(0, 1);
      const Mat Acr_xiui                = dcr.middleCols(1, ocp.nx + ocp.nu);

      std::cout << "Acr_xiui " << std::endl << Acr_xiui << std::endl;

      // TODO: add constraint ocp.crl <= Acr_xiui [xi; ui] <= ocp.cru to QP

      // LINEARIZE DYNAMICS
      const auto [f_i, df_i] = diff::dr<1>(ocp.f, wrt(t_i, xl_i, ul_i));

      const Mat A = -0.5 * ad<X>(f_i) - 0.5 * ad<X>(dxl_i) + df_i.middleCols(1, ocp.nx);
      const Mat B = df_i.middleCols(1 + ocp.nx, ocp.nu);
      const Vec E = f_i - dxl_i;

      assert(A.rows() == ocp.nx);
      assert(A.cols() == ocp.nx);
      assert(B.rows() == ocp.nx);
      assert(B.cols() == ocp.nu);
      assert(E.rows() == ocp.nx);
      assert(E.cols() == 1);

      // derivatives are w.r.t. scaled time tau \in [0, 1]
      Ca.block(0, i * ocp.nx, ocp.nx, ocp.nx)                     = tf * A;
      Ca.block(0, (Ni + 1) * ocp.nx + i * ocp.nu, ocp.nx, ocp.nu) = tf * B;
      Cb.col(i)                                                   = tf * E;
    }

    const Mat D = mesh.interval_diffmat(ival);

    // interval collocation constraint: kron(I, Ca) [X; U] - kron(D', I) X = -vec(Cb)

    // TODO: add collocation constraint to QP

    std::cout << "Ca" << std::endl << Ca << std::endl;
    std::cout << "Cb" << std::endl << Cb << std::endl;
    std::cout << "D" << std::endl << D << std::endl;
  }

  // calculate linearized system \dot x = A(t) x + B(t) u + K(t)

  // linearize integral

  // variables are [x0, x1, x2, ..., xN, u0, u1, ..., uN-1]

  return QuadraticProgramSparse<double>{};
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__LINEAR_OCP_HPP_
