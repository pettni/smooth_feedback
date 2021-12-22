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

/**
 * @file
 * @brief Evaluate transform-like functions and derivatives on collocation points.
 */

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <smooth/diff.hpp>

#include <ranges>

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
    dF_dt0.resize(nf * N, 1);
    dF_dt0.reserve(Eigen::VectorXi::Constant(1, nf * N));

    // dense column vector
    dF_dtf.resize(nf * N, 1);
    dF_dt0.reserve(Eigen::VectorXi::Constant(1, nf * N));

    // block diagonal matrix (blocks have size nf x nx)
    dF_dX.resize(nf * N, nx * (N + 1));
    Eigen::VectorXi FX_pattern = Eigen::VectorXi::Constant(nx * (N + 1), nf);
    FX_pattern.tail(nx).setZero();
    dF_dX.reserve(FX_pattern);

    // block diagonal matrix (blocks have size nf x nu)
    dF_dU.resize(nf * N, nu * N);
    dF_dU.reserve(Eigen::VectorXi::Constant(nu * N, nf));
  }

  inline void setZero()
  {
    F.setZero();
    if (dF_dt0.isCompressed()) { dF_dt0.coeffs().setZero(); }
    if (dF_dtf.isCompressed()) { dF_dtf.coeffs().setZero(); }
    if (dF_dX.isCompressed()) { dF_dX.coeffs().setZero(); }
    if (dF_dU.isCompressed()) { dF_dU.coeffs().setZero(); }
  }

  inline void makeCompressed()
  {
    dF_dt0.makeCompressed();
    dF_dtf.makeCompressed();
    dF_dX.makeCompressed();
    dF_dU.makeCompressed();
  }

  std::size_t nf, nx, nu, N;

  /// @brief Function values (size nf x N)
  Eigen::MatrixXd F;
  /// @brief Function derivatives w.r.t. t0 (size N*nf x 1)
  Eigen::SparseMatrix<double> dF_dt0;
  /// @brief Function derivatives w.r.t. tf (size N*nf x 1)
  Eigen::SparseMatrix<double> dF_dtf;
  /// @brief Function derivatives w.r.t. X (size N*nf x nx*(N+1))
  Eigen::SparseMatrix<double> dF_dX;
  /// @brief Function derivatives w.r.t. X (size N*nf x nu*N)
  Eigen::SparseMatrix<double> dF_dU;
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
 * @tparam DT differentiation method
 *
 * @param[out] result
 * @param[in] f function (t, x, u) -> R^nf
 * @param[in] m Mesh of time
 * @param[in] t0 initial time variable
 * @param[in] tf final time variable
 * @param[in] xs state variables (size N+1)
 * @param[in] us input variables (size N)
 */
template<uint8_t Deriv = 0, diff::Type DT = diff::Type::Default>
  requires(Deriv == 0 || Deriv == 1)
void colloc_eval(
  CollocEvalResult & res,
  auto && f,
  const MeshType auto & m,
  const double t0,
  const double tf,
  std::ranges::range auto && xs,
  std::ranges::range auto && us)
{
  using X = PlainObject<std::decay_t<std::ranges::range_value_t<decltype(xs)>>>;
  using U = PlainObject<std::decay_t<std::ranges::range_value_t<decltype(us)>>>;

  const auto N = m.N_colloc();

  res.setZero();

  for (const auto & [ival, tau, x, u] :
       utils::zip(std::views::iota(0u, N), m.all_nodes(), xs, us)) {
    const double ti = t0 + (tf - t0) * tau;

    const X x_plain = x;
    const U u_plain = u;

    if constexpr (Deriv == 0u) {
      res.F.col(ival) = f(ti, x_plain, u_plain);
    } else if constexpr (Deriv == 1u) {
      const auto [fval, dfval] = diff::dr<1, DT>(f, wrt(ti, x_plain, u_plain));

      res.F.col(ival) = fval;
      for (auto row = 0u; row < res.nf; ++row) {
        res.dF_dt0.coeffRef(res.nf * ival + row, 0) = dfval(row, 0) * (1. - tau);
        res.dF_dtf.coeffRef(res.nf * ival + row, 0) = dfval(row, 0) * tau;
        for (auto col = 0u; col < res.nx; ++col) {
          res.dF_dX.coeffRef(res.nf * ival + row, ival * res.nx + col) = dfval(row, 1 + col);
        }
        for (auto col = 0u; col < res.nu; ++col) {
          res.dF_dU.coeffRef(res.nf * ival + row, ival * res.nu + col) =
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
 * @tparam DT differentiation method
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
 * If Deriv == true, {F, dF_dt0, dF_dtf, dF_dX, dF_dQ},
 */
template<uint8_t Deriv, diff::Type DT = diff::Type::Default>
  requires(Deriv == 0 || Deriv == 1)
auto colloc_eval_endpt(
  const std::size_t nf,
  const std::size_t nx,
  auto && f,
  [[maybe_unused]] const double t0,
  const double tf,
  std::ranges::range auto && xs,
  const smooth::traits::RnType auto & q)
{
  using X = PlainObject<std::ranges::range_value_t<decltype(xs)>>;
  using Q = PlainObject<std::decay_t<decltype(q)>>;

  // NOTE: for now t0 = 0 and we don't want t0 in signatures
  assert(t0 == 0);

  const X x0 = *std::ranges::begin(xs);

  // hack to find size and last element in case if input range..
  // TODO write more efficient code for sized ranges
  auto numX  = 0u;
  const X xf = [&]() {
    X ret;
    for (const auto & x : xs) {
      ++numX;
      ret = x;
    }
    return ret;
  }();

  const Q q_plain = q;

  if constexpr (Deriv == 0u) {
    return f.template operator()<double>(tf, x0, xf, q_plain);
  } else if constexpr (Deriv == 1u) {
    const auto [Fval, J] = diff::dr<1, DT>(f, wrt(tf, x0, xf, q_plain));

    assert(static_cast<std::size_t>(J.rows()) == nf);
    assert(static_cast<std::size_t>(J.cols()) == 1 + 2 * nx + q_plain.size());

    Eigen::SparseMatrix<double> dF_dt0, dF_dtf, dF_dX, dF_dQ;

    dF_dt0.resize(nf, 1);
    // dF_dt0.reserve(nf);
    // for (auto i = 0u; i < nf; ++i) { dF_dt0.insert(i, 0) = J(i, 0); }

    dF_dtf.resize(nf, 1);
    dF_dtf.reserve(nf);
    for (auto i = 0u; i < nf; ++i) { dF_dtf.insert(i, 0) = J(i, 0); }

    dF_dX.resize(nf, nx * numX);
    Eigen::VectorXi pattern = Eigen::VectorXi::Zero(nx * numX);
    pattern.head(nx).setConstant(nf);
    pattern.tail(nx).setConstant(nf);
    dF_dX.reserve(pattern);

    for (auto row = 0u; row < nf; ++row) {
      for (auto col = 0u; col < nx; ++col) {
        dF_dX.insert(row, col)                  = J(row, 1 + col);
        dF_dX.insert(row, nx * numX - nx + col) = J(row, 1 + nx + col);
      }
    }

    dF_dQ.resize(nf, q_plain.size());
    dF_dQ.reserve(Eigen::VectorXi::Constant(q_plain.size(), nf));

    for (auto row = 0u; row < nf; ++row) {
      for (auto col = 0u; col < q_plain.size(); ++col) {
        dF_dQ.insert(row, col) = J(row, 1 + 2 * nx + col);
      }
    }

    dF_dt0.makeCompressed();
    dF_dtf.makeCompressed();
    dF_dX.makeCompressed();
    dF_dQ.makeCompressed();

    return std::make_tuple(Fval, dF_dt0, dF_dtf, dF_dX, dF_dQ);
  }
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION__EVAL_HPP_
