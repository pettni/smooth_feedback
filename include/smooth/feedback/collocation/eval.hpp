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

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION__EVAL_HPP_
