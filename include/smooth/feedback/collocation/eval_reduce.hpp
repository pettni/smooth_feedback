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

#ifndef SMOOTH__FEEDBACK__COLLOCATION__EVAL_REDUCE_HPP_
#define SMOOTH__FEEDBACK__COLLOCATION__EVAL_REDUCE_HPP_

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
struct CollocEvalReduceResult
{
  inline CollocEvalReduceResult(
    const std::size_t nf, const std::size_t nx, const std::size_t nu, const std::size_t N)
      : nf(nf), nx(nx), nu(nu), N(N)
  {
    F.resize(nf, 1);

    dF_dt0.resize(nf, 1);
    dF_dt0.reserve(Eigen::VectorXi::Constant(1, nf));
    dF_dtf.resize(nf, 1);
    dF_dtf.reserve(Eigen::VectorXi::Constant(1, nf));
    dF_dvecX.resize(nf, nx * (N + 1));
    dF_dvecX.reserve(Eigen::VectorXi::Constant(nx * (N + 1), nf));
    dF_dvecU.resize(nf, nu * N);
    dF_dvecU.reserve(Eigen::VectorXi::Constant(nu * N, nf));
  }

  inline void setZero()
  {
    F.setZero();
    if (dF_dt0.isCompressed()) { dF_dt0.setZero(); }
    if (dF_dtf.isCompressed()) { dF_dtf.setZero(); }
    if (dF_dvecX.isCompressed()) { dF_dvecX.setZero(); }
    if (dF_dvecU.isCompressed()) { dF_dvecU.setZero(); }

    // Eigen::Map<Eigen::VectorXd>(d2F.valuePtr(), d2F.nonZeros()).setZero();
  }

  inline void makeCompressed()
  {
    dF_dt0.makeCompressed();
    dF_dtf.makeCompressed();
    dF_dvecX.makeCompressed();
    dF_dvecU.makeCompressed();
  }

  std::size_t nf, nx, nu, N;

  /// @brief Function value (size N)
  Eigen::VectorXd F;
  /// @brief Function derivatives w.r.t. t0 (size nf x 1)
  Eigen::SparseMatrix<double> dF_dt0;
  /// @brief Function derivatives w.r.t. tf (size nf x 1)
  Eigen::SparseMatrix<double> dF_dtf;
  /// @brief Function derivatives w.r.t. X (size nf x nx*(N+1))
  Eigen::SparseMatrix<double> dF_dvecX;
  /// @brief Function derivatives w.r.t. X (size nf x nu*N)
  Eigen::SparseMatrix<double> dF_dvecU;
  /// @brief Second order derivative (side 2 + nx*(N+1) + nu*N)
  // Eigen::SparseMatrix<double> d2F;
};

/**
 * @brief Evaluate-reduce a function on all collocation points.
 *
 * Calculate the nf vector
 *
 *  F = \sum_i \lambda_i f(t_i, x_i, u_i)
 *
 * and its derivatives.
 *
 * @tparam Deriv differentiation order
 *
 * @param[out] result
 * @param[in] f function (t, x, u) -> R^nf
 * @param[in] ls multiplication \lambda_i weights
 * @param[in] m Mesh of time
 * @param[in] t0 initial time variable
 * @param[in] tf final time variable
 * @param[in] xs state variables x_i (size N+1)
 * @param[in] us input variables u_i (size N)
 */
template<uint8_t Deriv = 0>
void colloc_eval_reduce(
  CollocEvalReduceResult & res,
  std::ranges::range auto && ls,
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

  for (const auto & [ival, tau, l, x, u] : utils::zip(std::views::iota(0u), tau_s, ls, xs, us)) {
    const double ti = t0 + (tf - t0) * tau;

    const X x_plain = x;
    const U u_plain = u;

    if constexpr (Deriv == 0u) {
      res.F += l * f(ti, x_plain, u_plain);
    } else if constexpr (Deriv == 1u) {
      const auto [fval, dfval] = diff::dr<1>(f, wrt(ti, x_plain, u_plain));
      res.F += l * fval;
      for (auto row = 0u; row < res.nf; ++row) {
        res.dF_dt0.coeffRef(row, 0) += l * dfval(row, 0) * (1. - tau);
        res.dF_dtf.coeffRef(row, 0) += l * dfval(row, 0) * tau;
        for (auto col = 0u; col < res.nx; ++col) {
          res.dF_dvecX.coeffRef(row, ival * res.nx + col) += l * dfval(row, 1 + col);
        }
        for (auto col = 0u; col < res.nu; ++col) {
          res.dF_dvecU.coeffRef(row, ival * res.nu + col) += l * dfval(row, 1 + res.nx + col);
        }
      }
    }
  }

  res.makeCompressed();
}

/**
 * @brief Evaluate integral on Mesh.
 *
 * @tparam Deriv differentiation order.
 *
 * @param[out] result structure
 * @param[in] g integrand with signature (t, x, u) -> R^{nq}
 * @param[in] m mesh
 * @param[in] t0 initial time (variable of size 1)
 * @param[in] tf final time (variable of size 1)
 * @param[in] xs state values (variable of size N+1)
 * @param[in] us input values (variable of size N)
 */
template<uint8_t Deriv>
void colloc_integrate(
  CollocEvalReduceResult & res,
  auto && g,
  const MeshType auto & m,
  const double t0,
  const double tf,
  std::ranges::sized_range auto && xs,
  std::ranges::sized_range auto && us)
{
  const auto [n, w] = m.all_nodes_and_weights();

  colloc_eval_reduce<Deriv>(res, w, g, m, t0, tf, xs, us);

  // result is equal to (tf - t0) * F
  //
  //  d/dt0  : -F + (tf - t0) * dF/dt0
  //  d/dtf  : F + (tf - t0) * dF/dt0
  //  d/dx   : (tf - t0) dF/dx
  //  d/du   : (tf - t0) dF/du

  res.dF_dt0 *= (tf - t0);
  res.dF_dt0 -= res.F.sparseView();
  res.dF_dtf *= (tf - t0);
  res.dF_dtf += res.F.sparseView();
  res.dF_dvecX *= (tf - t0);
  res.dF_dvecU *= (tf - t0);

  res.F *= (tf - t0);  // must be last
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION__EVAL_REDUCE_HPP_
