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

// DESIGN DECISIONS:
//  - Output structure: align with variables or align with function args
//  - How to do reduce-style functions and their second derivatives

/**
 * @file
 * @brief Evaluate transform-like functions and derivatives on collocation points.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <smooth/diff.hpp>

#include "mesh.hpp"

namespace smooth::feedback {

template<uint8_t Deriv>
struct MeshValue;

/**
 * @brief Result of Mesh function.
 *
 * A generic function is a mapping (t0, tf, q, X, U) -> R^M
 *
 * The function variables are
 * NAME     SIZE
 *   t0        1
 *   tf        1
 *    X nx*(N+1)
 *    U     nu*N
 *
 * Let the total number of variables be K = 2 + nq + nx*(N+1) + nu*N.
 */
template<>
struct MeshValue<0>
{
  /// @brief Function value (size M)
  Eigen::VectorXd F;

  bool allocated{false};
};

/**
 * @brief Result and first derivative of Mesh function.
 *
 * @see MeshValue<0>
 */
template<>
struct MeshValue<1> : public MeshValue<0>
{
  /// @brief Size M x numVar
  Eigen::SparseMatrix<double> dF;
};

/**
 * @brief Result, first, and second derivatives w.r.t. collocation variables
 *
 * @see MeshValue<0>, MeshValue<1>
 */
template<>
struct MeshValue<2> : public MeshValue<1>
{
  /// @brief Size numVar x numVar
  Eigen::SparseMatrix<double> d2F;
};

/**
 * @brief Reset MeshValue to zeros.
 *
 * If sparse matrices are compressed the coefficients are set to zero, otherwise coefficients are
 * removed but memory is kept.
 */
template<uint8_t Deriv>
void setZero(MeshValue<Deriv> & mv)
{
  mv.F.setZero();

  if constexpr (Deriv >= 1) {
    if (mv.dF.isCompressed()) {
      mv.dF.coeffs().setZero();
    } else {
      mv.dF.setZero();
    }
  }
  if constexpr (Deriv >= 2) {
    if (mv.d2F.isCompressed()) {
      mv.d2F.coeffs().setZero();
    } else {
      mv.d2F.setZero();
    }
  }
}

/**
 * @brief Evaluate function on all points in mesh
 *
 * The variables layout is:
 * NAME       LEN
 *   t0         1
 *   tf         1
 *    x  nx*(N+1)
 *    u    nu * N
 */
template<uint8_t Deriv, diff::Type DT = diff::Type::Default>
  requires(Deriv <= 1)
void mesh_eval(
  MeshValue<Deriv> & out,
  const MeshType auto & mesh,
  auto && f,
  const double t0,
  const double tf,
  std::ranges::range auto && xs,
  std::ranges::range auto && us)
{
  using utils::zip, std::views::iota;
  using X = PlainObject<std::decay_t<std::ranges::range_value_t<decltype(xs)>>>;
  using U = PlainObject<std::decay_t<std::ranges::range_value_t<decltype(us)>>>;

  static constexpr auto nx = Dof<X>;
  static constexpr auto nu = Dof<U>;
  static constexpr auto nf = Dof<std::invoke_result_t<decltype(f), double, X, U>>;

  static_assert(nx > -1, "State dimension must be static");
  static_assert(nu > -1, "Input dimension must be static");
  static_assert(nf > -1, "Output size must be static");

  const auto N = mesh.N_colloc();

  const Eigen::Index numOuts = nf * N;
  const Eigen::Index numVars = 2 + nx * (N + 1) + nu * N;

  if (!out.allocated) {
    out.F.resize(numOuts);

    if constexpr (Deriv == 1) {
      out.dF.resize(numOuts, numVars);
      Eigen::VectorXi pattern = Eigen::VectorXi::Zero(numVars);
      pattern.segment(0, 1).setConstant(numOuts);                 // t0 is dense
      pattern.segment(1, 1).setConstant(numOuts);                 // tf is dense
      pattern.segment(2, nx * N).setConstant(nf);                 // block diagonal, last x not used
      pattern.segment(2 + nx * (N + 1), nu * N).setConstant(nf);  // block diagonal
      out.dF.reserve(pattern);
    }

    out.allocated = true;
  }

  setZero(out);

  for (const auto & [ival, tau, x, u] : zip(iota(0u, N), mesh.all_nodes(), xs, us)) {
    const double ti = t0 + (tf - t0) * tau;

    const X x_plain = x;
    const U u_plain = u;

    const auto fval = diff::dr<1, DT>(f, wrt(ti, x_plain, u_plain));

    out.F.segment(ival * nf, nf) = std::get<0>(fval);

    if constexpr (Deriv >= 1u) {
      const auto & df = std::get<1>(fval);
      block_add(out.dF, nf * ival, 0, df.middleCols(0, 1), 1. - tau);
      block_add(out.dF, nf * ival, 1, df.middleCols(0, 1), tau);
      block_add(out.dF, nf * ival, 2 + ival * nx, df.middleCols(1, nx));
      block_add(out.dF, nf * ival, 2 + nx * (N + 1) + nu * ival, df.middleCols(1 + nx, nu));
    }
  }

  if constexpr (Deriv >= 1) { out.dF.makeCompressed(); }
}

template<uint8_t Deriv, diff::Type DT = diff::Type::Default>
  requires(Deriv <= 2)
void mesh_eval_reduce(
  MeshValue<Deriv> & out,
  const MeshType auto & mesh,
  auto && f,
  const double t0,
  const double tf,
  std::ranges::range auto && xs,
  std::ranges::range auto && us,
  std::ranges::range auto && ls)
{
  using utils::zip, std::views::iota;
  using X = PlainObject<std::decay_t<std::ranges::range_value_t<decltype(xs)>>>;
  using U = PlainObject<std::decay_t<std::ranges::range_value_t<decltype(us)>>>;

  static constexpr auto nx = Dof<X>;
  static constexpr auto nu = Dof<U>;
  static constexpr auto nf = Dof<std::invoke_result_t<decltype(f), double, X, U>>;

  static_assert(nx > -1, "State dimension must be static");
  static_assert(nu > -1, "Input dimension must be static");
  static_assert(nf > -1, "Output size must be static");

  const auto N = mesh.N_colloc();

  const Eigen::Index numOuts = nf;
  const Eigen::Index numVars = 2 + nx * (N + 1) + nu * N;

  if (!out.allocated) {
    out.F.resize(numOuts);

    if constexpr (Deriv >= 1) {
      Eigen::VectorXi pattern = Eigen::VectorXi::Constant(numVars, numOuts);  // dense
      out.dF.resize(numOuts, numVars);
      out.dF.reserve(pattern);
    }

    if constexpr (Deriv >= 2) {
      Eigen::VectorXi pattern = Eigen::VectorXi::Zero(numVars);  // dense
      pattern(0)              = numVars;                         // t0 dense
      pattern(1)              = numVars;                         // tf dense
      pattern.segment(2, nx * N).setConstant(2 + nx);
      pattern.segment(2 + nx * (N + 1), nu * N).setConstant(2 + nu);

      out.d2F.resize(numVars, numOuts * numVars);
      out.d2F.reserve(pattern.replicate(numOuts, 1).reshaped());
    }

    out.allocated = true;
  }

  setZero(out);

  for (const auto & [i, tau, l, x, u] : zip(iota(0u, N), mesh.all_nodes(), ls, xs, us)) {
    const double ti = t0 + (tf - t0) * tau;

    const X x_plain = x;
    const U u_plain = u;

    const auto fval = diff::dr<Deriv, DT>(f, wrt(ti, x_plain, u_plain));

    out.F.noalias() += l * std::get<0>(fval);

    if constexpr (Deriv >= 1u) {
      const auto & df = std::get<1>(fval);
      block_add(out.dF, 0, 0, df.middleCols(0, 1), l * (1. - tau));
      block_add(out.dF, 0, 1, df.middleCols(0, 1), l * tau);
      block_add(out.dF, 0, 2 + i * nx, df.middleCols(1, nx), l);
      block_add(out.dF, 0, 2 + nx * (N + 1) + nu * i, df.middleCols(1 + nx, nu), l);
    }

    if constexpr (Deriv >= 2u) {
      const auto & d2f = std::get<2>(fval);
      for (auto j = 0u; j < nf; ++j) {
        // source block locations
        const auto b_s = (1 + nx + nu) * j;  // horizontal block
        const auto t_s = 0;                  // t
        const auto x_s = 1;                  // x
        const auto u_s = 1 + nx;             // u

        // destination block locations
        const auto b_d  = (2 + nx * (N + 1) + nu * N) * j;  // horizontal block
        const auto t0_d = 0;                                // t0
        const auto tf_d = 1;                                // tf
        const auto x_d  = 2 + nx * i;                       // x[i]
        const auto u_d  = 2 + nx * (N + 1) + nu * i;        // u[i]

        // clang-format off
        block_add(out.d2F, t0_d, b_d + t0_d, d2f.block(t_s, b_s + t_s, 1,  1), l * (1. - tau) * (1. - tau));  // t0t0
        block_add(out.d2F, t0_d, b_d + tf_d, d2f.block(t_s, b_s + t_s, 1,  1), l * (1. - tau) * tau);         // t0tf
        block_add(out.d2F, t0_d, b_d + x_d,  d2f.block(t_s, b_s + x_s, 1, nx), l * (1. - tau));               // t0x
        block_add(out.d2F, t0_d, b_d + u_d,  d2f.block(t_s, b_s + u_s, 1, nu), l * (1. - tau));               // t0u

        block_add(out.d2F, tf_d, b_d + t0_d, d2f.block(t_s, b_s + t_s, 1,  1), l * tau * (1. - tau));         // tft0
        block_add(out.d2F, tf_d, b_d + tf_d, d2f.block(t_s, b_s + t_s, 1,  1), l * tau * tau);                // tftf
        block_add(out.d2F, tf_d, b_d + x_d,  d2f.block(t_s, b_s + x_s, 1, nx), l * tau);                      // tfx
        block_add(out.d2F, tf_d, b_d + u_d,  d2f.block(t_s, b_s + u_s, 1, nu), l * tau);                      // tfu

        block_add(out.d2F, x_d, b_d + t0_d, d2f.block(x_s, b_s + t_s, nx,  1), l * (1. - tau));               // xt0
        block_add(out.d2F, x_d, b_d + tf_d, d2f.block(x_s, b_s + t_s, nx,  1), l * tau);                      // xtf
        block_add(out.d2F, x_d, b_d + x_d,  d2f.block(x_s, b_s + x_s, nx, nx), l);                            // xx
        block_add(out.d2F, x_d, b_d + u_d,  d2f.block(x_s, b_s + u_s, nx, nu), l);                            // xu

        block_add(out.d2F, u_d, b_d + t0_d, d2f.block(u_s, b_s + t_s, nu,  1), l * (1. - tau));               // ut0
        block_add(out.d2F, u_d, b_d + tf_d, d2f.block(u_s, b_s + t_s, nu,  1), l * tau);                      // utf
        block_add(out.d2F, u_d, b_d + x_d,  d2f.block(u_s, b_s + x_s, nu, nx), l);                            // ux
        block_add(out.d2F, u_d, b_d + u_d,  d2f.block(u_s, b_s + u_s, nu, nu), l);                            // uu
        // clang-format on
      }
    }
  }
}

template<uint8_t Deriv, diff::Type DT = diff::Type::Default>
  requires(Deriv <= 2)
void mesh_integrate(
  MeshValue<Deriv> & out,
  const MeshType auto & mesh,
  auto && f,
  const double t0,
  const double tf,
  std::ranges::range auto && xs,
  std::ranges::range auto && us)
{
  mesh_eval_reduce(out, mesh, std::forward<decltype(f)>(f), t0, tf, xs, us, mesh.all_weights());

  if constexpr (Deriv >= 2) {
    const auto nf   = out.F.size();
    const auto nvar = out.dF.cols();

    // scale second derivative
    out.d2F *= (tf - t0);
    for (auto j = 0u; j < nf; ++j) {
      const auto b0 = nvar * j;
      block_add(out.d2F, 0, b0 + 0, out.dF.row(j), -1);
      block_add(out.d2F, 0, b0 + 0, out.dF.row(j).transpose(), -1);
      block_add(out.d2F, 1, b0 + 0, out.dF.row(j), 1);
      block_add(out.d2F, 0, b0 + 1, out.dF.row(j).transpose(), 1);
    }
  }

  if constexpr (Deriv >= 1) {
    // scale first derivative
    out.dF *= (tf - t0);
    out.dF.col(0) -= out.F.sparseView();
    out.dF.col(1) += out.F.sparseView();
  }

  // scale value
  out.F *= (tf - t0);
}

template<uint8_t Deriv, diff::Type DT = diff::Type::Default>
  requires(Deriv <= 1)
void mesh_eval_endpt(
  MeshValue<Deriv> & out,
  auto && f,
  const double t0,
  const double tf,
  const smooth::traits::RnType auto & q,
  const auto && x0,
  const auto && xf)
{
  using Q = PlainObject<std::decay_t<decltype(q)>>;
  using X = PlainObject<std::decay_t<decltype(x0)>>;

  static_assert(std::is_same_v<X, PlainObject<std::decay_t<decltype(xf)>>>);

  static constexpr auto nx = Dof<X>;
  static constexpr auto nq = Dof<Q>;
  static constexpr auto nf = Dof<std::invoke_result_t<decltype(f), double, X, X, Q>>;

  static_assert(nx > 0, "State dimension must be static");
  static_assert(nq > 0, "Integral dimension must be static");
  static_assert(nf > 0, "Output size must be static");
  static_assert(Deriv < 2 || nf == 1, "Second derivative requires output size equal to 1");

  const Eigen::Index numOuts = nf;
  const Eigen::Index numVars = 2 + nq + 2 * nx;

  if (!out.allocated) {
    out.F.resize(numOuts, 1);
    if constexpr (Deriv >= 1) {
      out.dF.resize(numOuts, numVars);
      out.dF.reserve(Eigen::VectorXi::Constant(numVars, numOuts));
    }

    out.allocated = true;
  }

  const Q q_plain  = q;
  const X x0_plain = x0;
  const X xf_plain = xf;

  if constexpr (Deriv == 0u) {
    out.F = f(t0, tf, q_plain, x0_plain, xf_plain);
  } else if constexpr (Deriv == 1u) {
    const auto [fval, dfval] = diff::dr<1, DT>(f, wrt(t0, tf, q_plain, x0_plain, xf_plain));

    // value
    out.F = fval;

    // first derivative
    block_copy(out.dF, 0, 0, dfval);
  }

  if constexpr (Deriv >= 1) { out.dF.makeCompressed(); }
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION__EVAL_HPP_
