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

#ifndef SMOOTH__FEEDBACK__COLLOCATION__DYN_HPP_
#define SMOOTH__FEEDBACK__COLLOCATION__DYN_HPP_

/**
 * @file
 * @brief Collocation dynamics constraints.
 */

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <smooth/diff.hpp>

#include <cstddef>
#include <numeric>
#include <ranges>
#include <vector>

#include "eval.hpp"
#include "mesh.hpp"

namespace smooth::feedback {

using smooth::utils::zip;

/**
 * @brief Evaluate dynamics constraint in all collocation points of a Mesh.
 *
 * @tparam Der return derivatives w.r.t variables
 *
 * @param nx state space degrees of freedom
 * @param f right-hand side of dynamics with signature (t, x, u) -> dx where x and dx are size nx
 * x 1 and u is size nu x 1
 * @param m mesh with a total of N collocation points
 * @param tf final time (variable of size 1)
 * @param x state values (matrix of size nx x N+1)
 * @param u input values (matrix of size nu x N)
 *
 * @return {F, dvecF_dt0, dvecF_dtf, dvecF_dvecX, dvecF_dvecU},
 * where vec(xs) stacks the columns of xs into a single column vector.
 *
 * @note This only works on flat spaces, which is why xs and us are matrices rather than ranges.
 */
template<bool Deriv>
  requires(Deriv == 0 || Deriv == 1)
auto colloc_dyn(
  const std::size_t nx,
  auto && f,
  const MeshType auto & m,
  const double t0,
  const double tf,
  const Eigen::MatrixXd & xs,
  const Eigen::MatrixXd & us)
{
  assert(m.N_colloc() + 1 == static_cast<std::size_t>(xs.cols()));  // extra at the end
  assert(m.N_colloc() == static_cast<std::size_t>(us.cols()));      // one per collocation point
  assert(nx == static_cast<std::size_t>(xs.rows()));                // one per collocation point

  const auto N  = m.N_colloc();
  const auto nu = us.rows();

  CollocEvalResult feval_res(nx, nx, nu, N);

  Eigen::MatrixXd XD(nx, m.N_colloc());
  Eigen::SparseMatrix<double> dvecXD_dvecX;

  colloc_eval<Deriv>(feval_res, f, m, t0, tf, xs.colwise(), us.colwise());

  if constexpr (Deriv == 1) {
    dvecXD_dvecX.resize(XD.size(), xs.size());

    // reserve sparsity pattern
    Eigen::VectorXi pattern = Eigen::VectorXi::Zero(xs.size());
    for (auto M = 0u, i = 0u; i < m.N_ivals(); ++i) {
      const std::size_t K = m.N_colloc_ival(i);
      pattern.segment(M, (K + 1) * nx) += Eigen::VectorXi::Constant((K + 1) * nx, K);
      M += K * nx;
    }
    dvecXD_dvecX.reserve(pattern);
  }

  for (auto i = 0u, M = 0u; i < m.N_ivals(); M += m.N_colloc_ival(i), ++i) {
    const std::size_t K     = m.N_colloc_ival(i);
    const Eigen::MatrixXd D = m.interval_diffmat(i);
    XD.block(0, M, nx, K)   = xs.block(0, M, nx, K + 1) * D;

    if constexpr (Deriv) {
      // vec(xs * D) = kron(D', I) * vec(xs), so derivative w.r.t vec(xs) = kron(D', I)
      for (auto i = 0u; i < K; ++i) {
        for (auto j = 0u; j < K + 1; ++j) {
          for (auto diag = 0u; diag < nx; ++diag) {
            dvecXD_dvecX.coeffRef(M * nx + i * nx + diag, M * nx + j * nx + diag) += D(j, i);
          }
        }
      }
    }
  }

  Eigen::VectorXd Fv = (XD - (tf - t0) * feval_res.F).reshaped();

  // scale equalities by by quadrature weights

  // vec(A * W) = kron(W', I) * vec(A), so we apply kron(W', I) on the left
  Eigen::SparseMatrix<double> W(N, N);
  W.reserve(Eigen::VectorXi::Ones(N));
  for (const auto [i, w] : zip(std::views::iota(0u, N), m.all_weights())) { W.insert(i, i) = w; }
  const Eigen::SparseMatrix<double> W_kron_I = kron_identity(W, nx);

  Fv.applyOnTheLeft(W_kron_I);

  if constexpr (!Deriv) {
    return Fv;
  } else {
    dvecXD_dvecX.makeCompressed();

    Eigen::SparseMatrix<double> dF_dt0 = -(tf - t0) * feval_res.dF_dt0;
    dF_dt0 += feval_res.F.reshaped().sparseView();  // OK since dvecF_dtf is dense
    dF_dt0 = W_kron_I * dF_dt0;

    Eigen::SparseMatrix<double> dF_dtf = -(tf - t0) * feval_res.dF_dtf;
    dF_dtf -= feval_res.F.reshaped().sparseView();  // OK since dvecF_dtf is dense
    dF_dtf = W_kron_I * dF_dtf;

    Eigen::SparseMatrix<double> dF_dvecX = dvecXD_dvecX;
    dF_dvecX -= (tf - t0) * feval_res.dF_dX;
    dF_dvecX = W_kron_I * dF_dvecX;

    Eigen::SparseMatrix<double> dF_dvecU = -(tf - t0) * W_kron_I * feval_res.dF_dU;

    dF_dt0.makeCompressed();
    dF_dtf.makeCompressed();
    dF_dvecX.makeCompressed();
    dF_dvecU.makeCompressed();

    return std::make_tuple(
      std::move(Fv),
      std::move(dF_dt0),
      std::move(dF_dtf),
      std::move(dF_dvecX),
      std::move(dF_dvecU));
  }
}

/**
 * @brief Calculate relative dynamics errors for each interval in mesh.
 *
 * @param nx state space dimension
 * @param f dynamics function
 * @param m Mesh
 * @param t0 initial time variable
 * @param tf final time variable
 * @param x state trajectory
 * @param u input trajectory
 *
 * @return vector with relative errors for every interval in m
 */
Eigen::VectorXd mesh_dyn_error(
  const std::size_t nx,
  auto && f,
  const MeshType auto & m,
  const double t0,
  const double tf,
  const std::function<Eigen::VectorXd(double)> xfun,
  const std::function<Eigen::VectorXd(double)> ufun)
{
  const auto N = m.N_ivals();

  // create a new mesh where each interval is extended
  Mesh mext = m;
  for (auto i = 0u; i < N; ++i) {
    const std::size_t K = m.N_colloc_ival(i);
    mext.set_N_colloc_ival(i, K + 1);
  }

  Eigen::VectorXd ival_errs(N);

  // for each interval
  for (auto i = 0u, M = 0u; i < N; M += m.N_colloc_ival(i), ++i) {
    const std::size_t Kext = mext.N_colloc_ival(i);

    // evaluate xs and F at those points
    Eigen::MatrixXd Fval(nx, Kext + 1);
    Eigen::MatrixXd Xval(nx, Kext + 1);
    for (const auto & [j, tau] : zip(std::views::iota(0u, Kext + 1), mext.interval_nodes(i))) {
      const double tj = t0 + (tf - t0) * tau;

      // evaluate x and u values at tj using current degree polynomials
      const auto Xj = xfun(tj);
      const auto Uj = ufun(tj);

      // evaluate right-hand side of dynamics at tj
      Fval.col(j) = f(tj, Xj, Uj);

      // store x values for later comparison
      Xval.col(j) = Xj;
    }

    // "integrate" system inside interval
    const Eigen::MatrixXd Xval_est =
      Xval.col(0).replicate(1, Kext) + (tf - t0) * Fval.leftCols(Kext) * mext.interval_intmat(i);

    // absolute error in interval
    Eigen::VectorXd e_abs = (Xval_est - Xval.rightCols(Kext)).colwise().norm();
    Eigen::VectorXd e_rel = e_abs / (1. + Xval.rightCols(Kext).colwise().norm().maxCoeff());

    // mex relative error on interval
    ival_errs(i) = e_rel.maxCoeff();
  }

  return ival_errs;
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION__DYN_HPP_
