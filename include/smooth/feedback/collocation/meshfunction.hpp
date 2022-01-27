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

// TODOS
// - [ ] Create lambda vector in MeshValue<2>
// - [ ] Add second derivative too all functions that sums with lambda
// - [ ] Write tests that check for reallocation: call, compress, call again (expect compressed)

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

  /// @brief If set to true correct allocation is assumed, and no allocation is performed.
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
 * @brief Evaluate function over a mesh.
 *
 * @tparam Deriv differentiation order (0, 1, or 2)
 * @tparam DT inner differentiation method (used to differentiate f)
 *
 * TODO implement Deriv = 2
 * TODO double check allocation pattern
 *
 * Computes the expression
 * \f[
 *  \begin{bmatrix}
 *    f(t_0 + (t_f - t_0) \tau_0, x_0, u_0) \\
 *    f(t_0 + (t_f - t_0) \tau_1, x_1, u_1) \\
 *    \vdots \\
 *    f(t_{N-1} + (t_f - t_0) \tau_{N-1}, x_{N-1}, u_{N-1})
 *  \end{bmatrix}.
 * \f]
 *
 * If DT > 0 derivatives w.r.t. t0, tf, {xi} and {ui} are returned.
 *
 * @param out result structure
 * @param m mesh
 * @param f integrand
 * @param t0 initial time parameter
 * @param tf final time parameter
 * @param xs state parameters {xi}
 * @param us input parameters {ui}
 */
template<uint8_t Deriv, diff::Type DT = diff::Type::Default>
  requires(Deriv <= 1)
void mesh_eval(
  MeshValue<Deriv> & out,
  const MeshType auto & m,
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

  const auto N = m.N_colloc();

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

  for (const auto & [i, tau, x, u] : zip(iota(0u, N), m.all_nodes(), xs, us)) {
    const double ti = t0 + (tf - t0) * tau;

    const X x_plain = x;
    const U u_plain = u;

    const auto fval = diff::dr<Deriv, DT>(f, wrt(ti, x_plain, u_plain));

    out.F.segment(i * nf, nf) = std::get<0>(fval);

    if constexpr (Deriv >= 1u) {
      const auto & df = std::get<1>(fval);
      block_add(out.dF, nf * i, 0, df.middleCols(0, 1), 1. - tau);
      block_add(out.dF, nf * i, 1, df.middleCols(0, 1), tau);
      block_add(out.dF, nf * i, 2 + i * nx, df.middleCols(1, nx));
      block_add(out.dF, nf * i, 2 + nx * (N + 1) + nu * i, df.middleCols(1 + nx, nu));
    }
  }
}

/**
 * @brief Evaluate integral over a mesh.
 *
 * TODO double check allocation pattern
 *
 * @tparam Deriv differentiation order (0, 1, or 2)
 * @tparam DT inner differentiation method (used to differentiate f)

 * This function approximates the integral
 * \f[
 *   \int_{t_0}^{t_f} f(s, x(s), u(s)) \mathrm{d} s.
 * \f]
 * by computing the quadrature
 * \f[
 *    (t_f - t_0) * \sum_{i = 0}^N w_i f(t_0 + (t_f - t_0) \tau_i, x_i, u_i).
 * \f]
 *
 * If DT > 0 derivatives w.r.t. t0, tf, {xi} and {ui} are returned.
 *
 * @param out result structure
 * @param m mesh
 * @param f integrand
 * @param t0 initial time parameter
 * @param tf final time parameter
 * @param xs state parameters {xi}
 * @param us input parameters {ui}
 */
template<uint8_t Deriv, diff::Type DT = diff::Type::Default>
  requires(Deriv <= 2)
void mesh_integrate(
  MeshValue<Deriv> & out,
  const MeshType auto & m,
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

  const auto N = m.N_colloc();

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

  for (const auto & [i, tau, w, x, u] : zip(iota(0u, N), m.all_nodes(), m.all_weights(), xs, us)) {
    const double ti = t0 + (tf - t0) * tau;

    const X x_plain = x;
    const U u_plain = u;

    const double c    = w * (tf - t0);
    const double mtau = 1. - tau;

    const auto fval = diff::dr<Deriv, DT>(f, wrt(ti, x_plain, u_plain));

    const auto & f = std::get<0>(fval);
    out.F.noalias() += c * f;

    if constexpr (Deriv >= 1u) {
      const auto & df = std::get<1>(fval);
      // t0
      block_add(out.dF, 0, 0, df.middleCols(0, 1), c * mtau);
      block_add(out.dF, 0, 0, f, -w);
      // tf
      block_add(out.dF, 0, 1, df.middleCols(0, 1), c * tau);
      block_add(out.dF, 0, 1, f, w);
      // x
      block_add(out.dF, 0, 2 + i * nx, df.middleCols(1, nx), c);
      // u
      block_add(out.dF, 0, 2 + nx * (N + 1) + nu * i, df.middleCols(1 + nx, nu), c);

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
          // t0t0
          block_add(out.d2F, t0_d, b_d + t0_d, d2f.block(t_s, b_s + t_s, 1,  1), c * mtau * mtau);
          block_add(out.d2F, t0_d, b_d + t0_d,  df.block(j,         t_s, 1,  1), -w * 2 * mtau);
          // t0tf
          block_add(out.d2F, t0_d, b_d + tf_d, d2f.block(t_s, b_s + t_s, 1,  1), c * mtau * tau);
          block_add(out.d2F, t0_d, b_d + tf_d,  df.block(j,         t_s, 1,  1), w * (1 - 2 * tau));
          // t0x
          block_add(out.d2F, t0_d, b_d + x_d,  d2f.block(t_s, b_s + x_s, 1, nx), c * mtau);
          block_add(out.d2F, t0_d, b_d + x_d,   df.block(j,         x_s, 1, nx), -w);
          // t0u
          block_add(out.d2F, t0_d, b_d + u_d,  d2f.block(t_s, b_s + u_s, 1, nu), c * mtau);
          block_add(out.d2F, t0_d, b_d + u_d,   df.block(j,         u_s, 1, nu), -w);

          // tft0
          block_add(out.d2F, tf_d, b_d + t0_d, d2f.block(t_s, b_s + t_s, 1,  1), c * tau * mtau);
          block_add(out.d2F, tf_d, b_d + t0_d,  df.block(j,         t_s, 1,  1), w * (1. - 2 * tau));
          // tftf
          block_add(out.d2F, tf_d, b_d + tf_d, d2f.block(t_s, b_s + t_s, 1,  1), c * tau * tau);
          block_add(out.d2F, tf_d, b_d + tf_d,  df.block(j,         t_s, 1,  1), w * 2 * tau);
          // tfx
          block_add(out.d2F, tf_d, b_d + x_d,  d2f.block(t_s, b_s + x_s, 1, nx), c * tau);
          block_add(out.d2F, tf_d, b_d + x_d,   df.block(j,         x_s, 1, nx), w);
          // tfu
          block_add(out.d2F, tf_d, b_d + u_d,  d2f.block(t_s, b_s + u_s, 1, nu), c * tau);
          block_add(out.d2F, tf_d, b_d + u_d,   df.block(j,         u_s, 1, nu), w);

          // xt0
          block_add(out.d2F, x_d, b_d + t0_d, d2f.block(x_s, b_s + t_s, nx,  1), c * mtau);
          block_add(out.d2F, x_d, b_d + t0_d,  df.block(j,         x_s, 1,  nx).transpose(), -w);
          // xtf
          block_add(out.d2F, x_d, b_d + tf_d, d2f.block(x_s, b_s + t_s, nx,  1), c * tau);
          block_add(out.d2F, x_d, b_d + tf_d,  df.block(j,         x_s, 1,  nx).transpose(), w);
          // xx
          block_add(out.d2F, x_d, b_d + x_d,  d2f.block(x_s, b_s + x_s, nx, nx), c);
          // xu
          block_add(out.d2F, x_d, b_d + u_d,  d2f.block(x_s, b_s + u_s, nx, nu), c);

          // ut0
          block_add(out.d2F, u_d, b_d + t0_d, d2f.block(u_s, b_s + t_s, nu,  1), c * mtau);
          block_add(out.d2F, u_d, b_d + t0_d,  df.block(j,         u_s, 1,  nu).transpose(), -w);
          // utf
          block_add(out.d2F, u_d, b_d + tf_d, d2f.block(u_s, b_s + t_s, nu,  1), c * tau);
          block_add(out.d2F, u_d, b_d + tf_d,  df.block(j,         u_s, 1,  nu).transpose(), w);
          // ux
          block_add(out.d2F, u_d, b_d + x_d,  d2f.block(u_s, b_s + x_s, nu, nx), c);
          // uu
          block_add(out.d2F, u_d, b_d + u_d,  d2f.block(u_s, b_s + u_s, nu, nu), c);
          // clang-format on
        }
      }
    }
  }
}

/**
 * @brief Evaluate dynamic constraints over a mesh (differentiation version).
 *
 * @tparam Deriv differentiation order (0, 1, or 2)
 * @tparam DT inner differentiation method (used to differentiate f)
 *
 * TODO implement Deriv = 2
 * TODO double check allocation pattern
 *
 * For interval \f$j\f$ with nodes \f$\tau_i\f$ and weights \f$w_i\f$ the corresponding dynamic
 * constraints are a block of the form
 * \f[
 *   \begin{bmatrix}
 *     w_0 \left( f\left(t_0 + (t_f - t_0) \tau_0, x_0, u_0\right) - \sum_{k=0}^{N_j} D_{k, 0} x_k
 * \right) \\
 *     \vdots  \\
 *     w_{N_j-1} \left( f\left(t_0 + (t_f - t_0) \tau_{N_j-1}, x_{N_j - 1}, u_{N_j - 1}\right) -
 * \sum_{k=0}^{N_j} D_{k, N_j-1} x_k \right) \end{bmatrix}, \f] where \f$D\f$ is the interval
 * differentiation matrix (see interval_diffmat() in Mesh).
 *
 * This function returnes all such blocks stacked into a 1D vector.
 *
 * If DT > 0 derivatives w.r.t. t0, tf, {xi} and {ui} are returned.
 *
 * @param out result structure
 * @param m mesh
 * @param f integrand
 * @param t0 initial time parameter
 * @param tf final time parameter
 * @param xs state parameters {xi}
 * @param us input parameters {ui}
 */
template<uint8_t Deriv, diff::Type DT = diff::Type::Default>
  requires(Deriv <= 1)
void mesh_dyn(
  MeshValue<Deriv> & out,
  const MeshType auto & m,
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
  static_assert(nx == nf, "Output dimension must be same as state dimension");

  const auto N               = m.N_colloc();
  const Eigen::Index numOuts = nx * N;
  const Eigen::Index numVars = 2 + nx * (N + 1) + nu * N;

  if (!out.allocated) {
    out.F.resize(numOuts);

    if constexpr (Deriv == 1) {
      Eigen::VectorXi pattern = Eigen::VectorXi::Zero(numVars);
      pattern(0)              = numOuts;  // t0 dense
      pattern(1)              = numOuts;  // tf dense

      // x has blocks depending on mesh size
      for (auto ival = 0u, idx0 = 2u; ival < m.N_ivals(); idx0 += m.N_colloc_ival(ival), ++ival) {
        const std::size_t K = m.N_colloc_ival(ival);
        pattern.segment(idx0, (K + 1) * nx) += Eigen::VectorXi::Constant((K + 1) * nx, K);
      }

      // u is block diagonal with small blocks
      pattern.segment(2 + nx * (N + 1), nu * N).setConstant(nu);

      out.dF.resize(numOuts, numVars);
      out.dF.reserve(pattern);
    }

    out.allocated = true;
  }

  setZero(out);

  // We build the constraint through two loops over xi:
  //   - the first loop adds wk * f(tk, xk, uk)
  //   - the second loop adds -wk * [x0 ... Xni] * dk

  // ADD FIRST PART

  for (const auto & [i, tau, w, x, u] : zip(iota(0u, N), m.all_nodes(), m.all_weights(), xs, us)) {
    const double ti = t0 + (tf - t0) * tau;
    const X xi      = x;
    const U ui      = u;

    const auto fval = diff::dr<Deriv, DT>(f, wrt(ti, xi, ui));
    const auto & f  = std::get<0>(fval);

    out.F.segment(i * nx, nx) += w * (tf - t0) * f;

    if constexpr (Deriv >= 1) {
      const auto & df = std::get<1>(fval);

      // dF/dt0 = -f + (tf - t0) * df/dti * (1-tau)
      block_add(out.dF, nx * i, 0, f, -w);
      block_add(out.dF, nx * i, 0, df.col(0), w * (tf - t0) * (1. - tau));
      // dF/dtf = f + (tf - t0) * df/dti * tau
      block_add(out.dF, nx * i, 1, f, w);
      block_add(out.dF, nx * i, 1, df.col(0), w * (tf - t0) * tau);
      // dF/dx
      block_add(out.dF, nx * i, 2 + nx * i, df.middleCols(1, nx), w * (tf - t0));
      // dF/du
      block_add(
        out.dF, nx * i, 2 + nx * (N + 1) + nu * i, df.middleCols(1 + nx, nu), w * (tf - t0));
    }
  }

  // ADD SECOND PART (LINEAR IN X)

  auto ival      = 0u;  // current interval index
  auto ival_idx0 = 0u;  // node start index of current interval
  auto Nival     = m.N_colloc_ival(ival);

  for (const auto & [i, x] : zip(iota(0u, N + 1), xs)) {
    if (i == ival_idx0 + Nival) {
      // jumping to new interval --- add overlap to current interval before switching
      const auto [alpha, Dus] = m.interval_diffmat_unscaled(ival);
      for (const auto & [j, w] : zip(iota(0u, Nival), m.interval_weights(ival))) {
        const auto row0   = (ival_idx0 + j) * nx;
        const double coef = -w * alpha * Dus(i - ival_idx0, j);

        out.F.segment(row0, nx) += coef * x;

        if constexpr (Deriv >= 1) {
          // add diagonal matrix
          for (auto k = 0u; k < nx; ++k) { out.dF.coeffRef(row0 + k, 2 + nx * i + k) += coef; }
        }
      }

      // update interval
      ++ival;
      ival_idx0 += Nival;
      Nival = m.N_colloc_ival(ival);
    }

    if (i < N) {
      const auto [alpha, Dus] = m.interval_diffmat_unscaled(ival);
      for (const auto & [j, w] : zip(iota(0u, Nival), m.interval_weights(ival))) {
        const auto row0   = (ival_idx0 + j) * nx;
        const double coef = -w * alpha * Dus(i - ival_idx0, j);

        out.F.segment(row0, nx) += coef * x;

        if constexpr (Deriv >= 1) {
          // add diagonal matrix
          for (auto k = 0u; k < nx; ++k) { out.dF.coeffRef(row0 + k, 2 + nx * i + k) += coef; }
        }
      }
    }
  }
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION__EVAL_HPP_
