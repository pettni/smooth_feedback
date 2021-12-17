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
    dF_dX.resize(nf, nx * (N + 1));
    dF_dX.reserve(Eigen::VectorXi::Constant(nx * (N + 1), nf));
    dF_dU.resize(nf, nu * N);
    dF_dU.reserve(Eigen::VectorXi::Constant(nu * N, nf));

    d2F_dt0t0.resize(1, 1);
    d2F_dt0t0.reserve(Eigen::VectorXi::Constant(1, 1));

    d2F_dt0tf.resize(1, 1);
    d2F_dt0tf.reserve(Eigen::VectorXi::Constant(1, 1));

    d2F_dt0X.resize(1, nx * (N + 1));
    d2F_dt0X.reserve(Eigen::VectorXi::Constant(nx * (N + 1), 1));

    d2F_dt0U.resize(1, nu * N);
    d2F_dt0U.reserve(Eigen::VectorXi::Constant(nu * N, 1));

    d2F_dtftf.resize(1, 1);
    d2F_dtftf.reserve(Eigen::VectorXi::Constant(1, 1));

    d2F_dtfX.resize(1, nx * (N + 1));
    d2F_dtfX.reserve(Eigen::VectorXi::Constant(nx * (N + 1), 1));

    d2F_dtfU.resize(1, nu * N);
    d2F_dtfU.reserve(Eigen::VectorXi::Constant(nu * N, 1));

    d2F_dXX.resize(nx * (N + 1), nx * (N + 1));
    d2F_dXX.reserve(Eigen::VectorXi::Constant(nx * (N + 1), nx));

    d2F_dXU.resize(nx * (N + 1), nu * N);
    d2F_dXU.reserve(Eigen::VectorXi::Constant(nu * N, nx));

    d2F_dUU.resize(nu * N, nu * N);
    d2F_dUU.reserve(Eigen::VectorXi::Constant(nu * N, nu));
  }

  inline void setZero()
  {
    F.setZero();
    if (dF_dt0.isCompressed()) { dF_dt0.setZero(); }
    if (dF_dtf.isCompressed()) { dF_dtf.setZero(); }
    if (dF_dX.isCompressed()) { dF_dX.setZero(); }
    if (dF_dU.isCompressed()) { dF_dU.setZero(); }

    if (d2F_dt0t0.isCompressed()) { d2F_dt0t0.setZero(); }
    if (d2F_dt0tf.isCompressed()) { d2F_dt0tf.setZero(); }
    if (d2F_dt0X.isCompressed()) { d2F_dt0X.setZero(); }
    if (d2F_dt0U.isCompressed()) { d2F_dt0U.setZero(); }
    if (d2F_dtftf.isCompressed()) { d2F_dtftf.setZero(); }
    if (d2F_dtfX.isCompressed()) { d2F_dtfX.setZero(); }
    if (d2F_dtfU.isCompressed()) { d2F_dtfU.setZero(); }
    if (d2F_dXX.isCompressed()) { d2F_dXX.setZero(); }
    if (d2F_dXU.isCompressed()) { d2F_dXU.setZero(); }
    if (d2F_dUU.isCompressed()) { d2F_dUU.setZero(); }
  }

  inline void makeCompressed()
  {
    dF_dt0.makeCompressed();
    dF_dtf.makeCompressed();
    dF_dX.makeCompressed();
    dF_dU.makeCompressed();

    d2F_dt0t0.makeCompressed();
    d2F_dt0tf.makeCompressed();
    d2F_dt0X.makeCompressed();
    d2F_dt0U.makeCompressed();
    d2F_dtftf.makeCompressed();
    d2F_dtfX.makeCompressed();
    d2F_dtfU.makeCompressed();
    d2F_dXX.makeCompressed();
    d2F_dXU.makeCompressed();
    d2F_dUU.makeCompressed();
  }

  std::size_t nf, nx, nu, N;

  /// @brief Function value (size N)
  Eigen::VectorXd F;
  /// @brief Function derivatives w.r.t. t0 (size nf x 1)
  Eigen::SparseMatrix<double> dF_dt0;
  /// @brief Function derivatives w.r.t. tf (size nf x 1)
  Eigen::SparseMatrix<double> dF_dtf;
  /// @brief Function derivatives w.r.t. X (size nf x nx*(N+1))
  Eigen::SparseMatrix<double> dF_dX;
  /// @brief Function derivatives w.r.t. X (size nf x nu*N)
  Eigen::SparseMatrix<double> dF_dU;

  /// @brief Second order derivatives when F is scalar
  //
  //  dF =
  //    [ 2F_dt0t0  d2F_dt0tf  d2F_dt0X  d2F_dt0U      <- 1
  //         *      d2F_dtftf  d2F_dtfX  d2F_dtfU      <- 1
  //         *          *       d2F_dXX   d2F_dXU      <- nX
  //         *          *          *      d2F_dUU]     <- nU
  //
  //         ^          ^          ^         ^
  //         1          1          nX        nU

  Eigen::SparseMatrix<double> d2F_dt0t0;

  Eigen::SparseMatrix<double> d2F_dt0tf;
  Eigen::SparseMatrix<double> d2F_dtftf;

  Eigen::SparseMatrix<double> d2F_dt0X;
  Eigen::SparseMatrix<double> d2F_dtfX;
  Eigen::SparseMatrix<double> d2F_dXX;

  Eigen::SparseMatrix<double> d2F_dtfU;
  Eigen::SparseMatrix<double> d2F_dt0U;
  Eigen::SparseMatrix<double> d2F_dXU;
  Eigen::SparseMatrix<double> d2F_dUU;
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
  std::ranges::range auto && xs,
  std::ranges::range auto && us)
{
  using X = PlainObject<std::ranges::range_value_t<decltype(xs)>>;
  using U = PlainObject<std::ranges::range_value_t<decltype(us)>>;

  if constexpr (std::ranges::sized_range<decltype(xs)>) {
    assert(m.N_colloc() + 1 == std::ranges::size(xs));
  }
  if constexpr (std::ranges::sized_range<decltype(us)>) {
    assert(m.N_colloc() == std::ranges::size(us));
  }

  res.setZero();

  for (const auto & [i, tau, l, x, u] :
       utils::zip(std::views::iota(0u), m.all_nodes_range(), ls, xs, us)) {
    const double ti = t0 + (tf - t0) * tau;

    const X x_plain = x;
    const U u_plain = u;

    if constexpr (Deriv == 0u) {
      res.F += l * f(ti, x_plain, u_plain);
    } else if constexpr (Deriv == 1u) {
      const auto [F, dF] = diff::dr<1>(f, wrt(ti, x_plain, u_plain));
      res.F += l * F;
      for (auto row = 0u; row < res.nf; ++row) {
        res.dF_dt0.coeffRef(row, 0) += l * dF(row, 0) * (1. - tau);
        res.dF_dtf.coeffRef(row, 0) += l * dF(row, 0) * tau;
        for (auto col = 0u; col < res.nx; ++col) {
          res.dF_dX.coeffRef(row, i * res.nx + col) += l * dF(row, 1 + col);
        }
        for (auto col = 0u; col < res.nu; ++col) {
          res.dF_dU.coeffRef(row, i * res.nu + col) += l * dF(row, 1 + res.nx + col);
        }
      }
    } else if constexpr (Deriv == 2u) {

      assert(res.nf == 1);

      const auto [F, dF, d2F] = diff::dr<2>(
        [&](auto &&... args) { return f(std::forward<decltype(args)>(args)...).x(); },
        wrt(ti, x_plain, u_plain));
      // value
      res.F.x() += l * F;

      // 1st deriv
      for (auto row = 0u; row < res.nf; ++row) {
        res.dF_dt0.coeffRef(row, 0) += l * dF(row, 0) * (1. - tau);
        res.dF_dtf.coeffRef(row, 0) += l * dF(row, 0) * tau;
        for (auto col = 0u; col < res.nx; ++col) {
          res.dF_dX.coeffRef(row, i * res.nx + col) += l * dF(row, 1 + col);
        }
        for (auto col = 0u; col < res.nu; ++col) {
          res.dF_dU.coeffRef(row, i * res.nu + col) += l * dF(row, 1 + res.nx + col);
        }
      }

      // 2nd deriv

      // first column
      res.d2F_dt0t0.coeffRef(0, 0) += l * d2F(0, 0) * (1. - tau) * (1. - tau);

      // second column
      res.d2F_dt0tf.coeffRef(0, 0) += l * d2F(0, 0) * (1. - tau) * tau;
      res.d2F_dtftf.coeffRef(0, 0) += l * d2F(0, 0) * tau * tau;

      // third column
      for (auto col = 0u; col < res.nx; ++col) {
        res.d2F_dt0X.coeffRef(0, i * res.nx + col) += l * d2F(0, 1 + col) * (1. - tau);
        res.d2F_dtfX.coeffRef(0, i * res.nx + col) += l * d2F(0, 1 + col) * tau;

        for (auto row = 0u; row < res.nx; ++row) {
          res.d2F_dXX.coeffRef(i * res.nx + row, i * res.nx + col) += l * d2F(1 + row, 1 + col);
        }
      }

      // fourth column
      for (auto col = 0u; col < res.nu; ++col) {
        res.d2F_dt0U.coeffRef(0, i * res.nu + col) += l * d2F(0, 1 + res.nx + col) * (1. - tau);
        res.d2F_dtfU.coeffRef(0, i * res.nu + col) += l * d2F(0, 1 + res.nx + col) * tau;

        for (auto row = 0u; row < res.nx; ++row) {
          res.d2F_dXU.coeffRef(i * res.nx + row, i * res.nu + col) +=
            l * d2F(1 + row, 1 + res.nx + col);
        }

        for (auto row = 0u; row < res.nu; ++row) {
          res.d2F_dUU.coeffRef(i * res.nu + row, i * res.nu + col) +=
            l * d2F(1 + res.nx + row, 1 + res.nx + col);
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
  std::ranges::range auto && xs,
  std::ranges::range auto && us)
{
  colloc_eval_reduce<Deriv>(res, m.all_weights_range(), g, m, t0, tf, xs, us);

  // result is equal to (tf - t0) * F, must update derivatives

  // SECOND DERIVATIVES

  // first row

  res.d2F_dt0t0 *= (tf - t0);
  res.d2F_dt0t0 -= 2 * res.dF_dt0;

  res.d2F_dt0tf *= (tf - t0);
  res.d2F_dt0tf += res.dF_dt0;
  res.d2F_dt0tf -= res.dF_dtf;

  res.d2F_dt0X *= (tf - t0);
  res.d2F_dt0X -= res.dF_dX;

  res.d2F_dt0U *= (tf - t0);
  res.d2F_dt0U -= res.dF_dU;

  // second row

  res.d2F_dtftf *= (tf - t0);
  res.d2F_dtftf += 2 * res.dF_dtf;

  res.d2F_dtfX *= (tf - t0);
  res.d2F_dtfX += res.dF_dX;

  res.d2F_dtfU *= (tf - t0);
  res.d2F_dtfU += res.dF_dU;

  // third row

  res.d2F_dXX *= (tf - t0);
  res.d2F_dXU *= (tf - t0);

  // fourth row

  res.d2F_dUU *= (tf - t0);

  // FIRST DERIVATIVES

  res.dF_dt0 *= (tf - t0);
  res.dF_dt0 -= res.F.sparseView();
  res.dF_dtf *= (tf - t0);
  res.dF_dtf += res.F.sparseView();
  res.dF_dX *= (tf - t0);
  res.dF_dU *= (tf - t0);

  // VALUE

  res.F *= (tf - t0);
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION__EVAL_REDUCE_HPP_
