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

#ifndef SMOOTH__FEEDBACK__COLLOCATION__EVAL_HPP_
#define SMOOTH__FEEDBACK__COLLOCATION__EVAL_HPP_

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <smooth/diff.hpp>

#include <cstddef>
#include <numeric>
#include <ranges>
#include <vector>

#include "mesh.hpp"

namespace smooth::feedback {

/**
 * @brief Output structure for colloc_eval
 */
struct CollocEvalResult
{
  inline CollocEvalResult(
    const std::size_t nf, const std::size_t nx, const std::size_t nu, const std::size_t N)
      : nf(nf), nx(nx), nu(nu), N(N)
  {
    F.resize(nf, N);

    // dense column vector
    dvecF_dt0.resize(nf * N, 1);
    dvecF_dt0.reserve(Eigen::VectorXi::Constant(1, nf * N));

    // dense column vector
    dvecF_dtf.resize(nf * N, 1);
    dvecF_dt0.reserve(Eigen::VectorXi::Constant(1, nf * N));

    // block diagonal matrix (blocks have size nf x nx)
    dvecF_dvecX.resize(nf * N, nx * (N + 1));
    Eigen::VectorXi FX_pattern = Eigen::VectorXi::Constant(nx * (N + 1), nf);
    FX_pattern.tail(nx).setZero();
    dvecF_dvecX.reserve(FX_pattern);

    // block diagonal matrix (blocks have size nf x nu)
    dvecF_dvecU.resize(nf * N, nu * N);
    dvecF_dvecU.reserve(Eigen::VectorXi::Constant(nu * N, nf));
  }

  inline void setZero()
  {
    F.setZero();
    if (dvecF_dt0.isCompressed()) { dvecF_dt0.coeffs().setZero(); }
    if (dvecF_dtf.isCompressed()) { dvecF_dtf.coeffs().setZero(); }
    if (dvecF_dvecX.isCompressed()) { dvecF_dvecX.coeffs().setZero(); }
    if (dvecF_dvecU.isCompressed()) { dvecF_dvecU.coeffs().setZero(); }
  }

  inline void makeCompressed()
  {
    dvecF_dt0.makeCompressed();
    dvecF_dtf.makeCompressed();
    dvecF_dvecX.makeCompressed();
    dvecF_dvecU.makeCompressed();
  }

  std::size_t nf, nx, nu, N;

  /// @brief Function values (size nf x N)
  Eigen::MatrixXd F;
  /// @brief Function derivatives w.r.t. t0 (size N*nf x 1)
  Eigen::SparseMatrix<double> dvecF_dt0;
  /// @brief Function derivatives w.r.t. tf (size N*nf x 1)
  Eigen::SparseMatrix<double> dvecF_dtf;
  /// @brief Function derivatives w.r.t. X (size N*nf x nx*(N+1))
  Eigen::SparseMatrix<double> dvecF_dvecX;
  /// @brief Function derivatives w.r.t. X (size N*nf x nu*N)
  Eigen::SparseMatrix<double> dvecF_dvecU;
};

/**
 * @brief Evaluate a function on all collocation points.
 *
 * Returns a nf x N matrix
 *
 *  F= [ f(t_0, X_0, U_0)  f(t_1, X_1, u_1) ... f(t_{N-1}, X_{N-1}, U_{N-1})]
 *
 * with the function evaluated at all collocation points t_i in the Mesh m.
 *
 * @tparam Deriv differentiation order
 *
 * @param[out] result
 * @param[in] f function (t, x, u) -> R^nf
 * @param[in] m Mesh of time
 * @param[in] t0 initial time variable
 * @param[in] tf final time variable
 * @param[in] xs state variables (size N+1)
 * @param[in] us input variables (size N)
 */
template<uint8_t Deriv = 0>
void colloc_eval(
  CollocEvalResult & res,
  auto && f,
  const MeshType auto & m,
  const double t0,
  const double tf,
  std::ranges::sized_range auto && xs,
  std::ranges::sized_range auto && us)
{
  using X = PlainObject<std::ranges::range_value_t<decltype(xs)>>;
  using U = PlainObject<std::ranges::range_value_t<decltype(us)>>;

  const auto numX = std::ranges::size(xs);
  const auto numU = std::ranges::size(us);

  assert(m.N_colloc() + 1 == numX);  //  extra variable at the end
  assert(m.N_colloc() == numU);      // one input per collocation point

  res.setZero();

  const auto [tau_s, w_s] = m.all_nodes_and_weights();

  for (const auto & [ival, tau, x, u] : utils::zip(std::views::iota(0u), tau_s, xs, us)) {
    const double ti = t0 + (tf - t0) * tau;

    const X x_plain = x;
    const U u_plain = u;

    if constexpr (Deriv == 0u) {
      res.F.col(ival) = f(ti, x_plain, u_plain);
    } else if constexpr (Deriv == 1u) {
      const auto [fval, dfval] = diff::dr<1>(f, wrt(ti, x_plain, u_plain));
      res.F.col(ival)          = fval;
      for (auto row = 0u; row < res.nf; ++row) {
        res.dvecF_dt0.coeffRef(res.nf * ival + row, 0) = dfval(row, 0) * (1. - tau);
        res.dvecF_dtf.coeffRef(res.nf * ival + row, 0) = dfval(row, 0) * tau;
        for (auto col = 0u; col < res.nx; ++col) {
          res.dvecF_dvecX.coeffRef(res.nf * ival + row, ival * res.nx + col) = dfval(row, 1 + col);
        }
        for (auto col = 0u; col < res.nu; ++col) {
          res.dvecF_dvecU.coeffRef(res.nf * ival + row, ival * res.nu + col) =
            dfval(row, 1 + res.nx + col);
        }
      }
    }
  }

  res.makeCompressed();
}

/**
 * @brief Evaluate a function at endpoints.
 *
 * Returns a nf vector
 *
 *  F = f(t_0, t_f, x_0, x_f, q)
 *
 * @tparam Der return derivatives w.r.t variables
 *
 * @param nf dimensionality of f image
 * @param nf state space degrees of freedom
 * @param f function (t0, tf, x0, xf, q) -> R^nf
 * @param t0 initial time
 * @param tf final time
 * @param xs state variables (size N+1)
 * @param q integrals
 *
 * @return If Deriv == false,
 * If Deriv == true, {F, dF_dt0, dF_dtf, dF_dvecX, dF_dQ},
 */
template<bool Deriv>
auto colloc_eval_endpt(
  const std::size_t nf,
  const std::size_t nx,
  auto && f,
  [[maybe_unused]] const double t0,
  const double tf,
  std::ranges::sized_range auto && xs,
  const Eigen::VectorXd & Q)
{
  using X = PlainObject<std::ranges::range_value_t<decltype(xs)>>;

  const auto numX = std::ranges::size(xs);
  assert(numX >= 2);

  // NOTE: for now t0 = 0 and we don't want t0 in signatures
  assert(t0 == 0);

  const X x0 = *std::ranges::begin(xs);
  const X xf = *std::ranges::next(std::ranges::begin(xs), numX - 1);

  if constexpr (!Deriv) {
    return f.template operator()<double>(tf, x0, xf, Q);
  } else {
    const auto [Fval, J] = diff::dr(f, wrt(tf, x0, xf, Q));

    assert(static_cast<std::size_t>(J.rows()) == nf);
    assert(static_cast<std::size_t>(J.cols()) == 1 + 2 * nx + Q.size());

    Eigen::SparseMatrix<double> dF_dt0, dF_dtf, dF_dvecX, dF_dQ;

    dF_dt0.resize(nf, 1);
    // dF_dt0.reserve(nf);
    // for (auto i = 0u; i < nf; ++i) { dF_dt0.insert(i, 0) = J(i, 0); }

    dF_dtf.resize(nf, 1);
    dF_dtf.reserve(nf);
    for (auto i = 0u; i < nf; ++i) { dF_dtf.insert(i, 0) = J(i, 0); }

    dF_dvecX.resize(nf, nx * numX);
    Eigen::VectorXi pattern = Eigen::VectorXi::Zero(nx * numX);
    pattern.head(nx).setConstant(nf);
    pattern.tail(nx).setConstant(nf);
    dF_dvecX.reserve(pattern);

    for (auto row = 0u; row < nf; ++row) {
      for (auto col = 0u; col < nx; ++col) {
        dF_dvecX.insert(row, col)                  = J(row, 1 + col);
        dF_dvecX.insert(row, nx * numX - nx + col) = J(row, 1 + nx + col);
      }
    }

    dF_dQ.resize(nf, Q.size());
    dF_dQ.reserve(Eigen::VectorXi::Constant(Q.size(), nf));

    for (auto row = 0u; row < nf; ++row) {
      for (auto col = 0u; col < Q.size(); ++col) {
        dF_dQ.insert(row, col) = J(row, 1 + 2 * nx + col);
      }
    }

    dF_dt0.makeCompressed();
    dF_dtf.makeCompressed();
    dF_dvecX.makeCompressed();
    dF_dQ.makeCompressed();

    return std::make_tuple(Fval, dF_dt0, dF_dtf, dF_dvecX, dF_dQ);
  }
}

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

  if constexpr (!Deriv) {
    colloc_eval<0>(feval_res, f, m, t0, tf, xs.colwise(), us.colwise());
  } else {
    colloc_eval<1>(feval_res, f, m, t0, tf, xs.colwise(), us.colwise());

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
  const auto [n, w] = m.all_nodes_and_weights();

  // vec(A * W) = kron(W', I) * vec(A), so we apply kron(W', I) on the left

  Eigen::SparseMatrix<double> W(N, N);
  W.reserve(Eigen::VectorXi::Ones(N));
  for (auto i = 0u; i < N; ++i) { W.insert(i, i) = w(i); }

  const Eigen::SparseMatrix<double> W_kron_I = kron_identity(W, nx);

  Fv.applyOnTheLeft(W_kron_I);

  if constexpr (!Deriv) {
    return Fv;
  } else {
    dvecXD_dvecX.makeCompressed();

    Eigen::SparseMatrix<double> dF_dt0 = -(tf - t0) * feval_res.dvecF_dt0;
    dF_dt0 += feval_res.F.reshaped().sparseView();  // OK since dvecF_dtf is dense
    dF_dt0 = W_kron_I * dF_dt0;

    Eigen::SparseMatrix<double> dF_dtf = -(tf - t0) * feval_res.dvecF_dtf;
    dF_dtf -= feval_res.F.reshaped().sparseView();  // OK since dvecF_dtf is dense
    dF_dtf = W_kron_I * dF_dtf;

    Eigen::SparseMatrix<double> dF_dvecX = dvecXD_dvecX;
    dF_dvecX -= (tf - t0) * feval_res.dvecF_dvecX;
    dF_dvecX = W_kron_I * dF_dvecX;

    Eigen::SparseMatrix<double> dF_dvecU = -(tf - t0) * W_kron_I * feval_res.dvecF_dvecU;

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

    const auto [tau_s, weights] = mext.interval_nodes_and_weights(i);

    assert(std::size_t(tau_s.size()) == Kext + 1);

    // evaluate xs and F at those points
    Eigen::MatrixXd Fval(nx, Kext + 1);
    Eigen::MatrixXd Xval(nx, Kext + 1);
    for (auto j = 0u; j < Kext + 1; ++j) {
      const double tj = t0 + (tf - t0) * tau_s(j);

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

/**
 * @brief Refine intervals in mesh to satisfy a target error criterion.
 * @param[in, out] m mesh to refine
 * @param[in] errs relative errors for all intervals (@see mesh_dyn_error())
 * @param[in] target_err target relative error
 */
void mesh_refine(MeshType auto & m, const Eigen::VectorXd & errs, const double target_err)
{
  const auto N = m.N_ivals();

  assert(N == std::size_t(errs.size()));

  for (auto i = 0u; i < N; ++i) {
    const auto Nmi = N - 1 - i;
    const auto Ki  = m.N_colloc_ival(Nmi);

    if (errs(Nmi) > target_err) {
      const auto Ktarget = Ki + std::lround(std::log(errs(Nmi) / target_err) / std::log(Ki) + 1);
      m.refine_ph(Nmi, Ktarget);
    }
  }
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION__EVAL_HPP_
