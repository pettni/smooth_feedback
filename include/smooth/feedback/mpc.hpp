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

#ifndef SMOOTH__FEEDBACK__MPC_HPP_
#define SMOOTH__FEEDBACK__MPC_HPP_

/**
 * @file
 * @brief Model-Predictive Control (MPC) on Lie groups.
 */

#include <Eigen/Core>

#include <smooth/concepts.hpp>
#include <smooth/diff.hpp>

#include "qp.hpp"

namespace smooth::feedback {

/**
 * @brief Optimal control problem defintiion.
 *
 * @tparam G State space Lie group.
 * @tparam U Input space Lie group.
 *
 * The optimal control problem is
 * \f[
 *   \begin{cases}
 *    \min_{u(\cdot)} & \int_{0}^T \left( (g(t) \ominus g_{des}(t))^T Q (g(t) \ominus g_{des}(t))^T
 *     + (u(t) \ominus u_{des}(t))^T R (u(t) \ominus u_{des}(t))^T \right)
 *     + (g(T) \ominus g_{des}(T))^T Q_T (g(T) \ominus g_{des}(T)) \\
 *     \text{s.t.}    & \mathrm{d}^r g_t = f(g, u),  \\
 *                    & g(0) = x_0,                  \\
 *                    & g(t) \ominus g_{min} \geq 0, \\
 *                    & g_{max} \ominus g(t) \geq 0, \\
 *                    & u(t) \ominus u_{min} \geq 0, \\
 *                    & u_{max} \ominus u(t) \geq 0,
 *   \end{cases}
 * \f]
 * where the cost matrices must be positive semi-definite.
 */
template<LieGroup G, Manifold U>
struct OptimalControlProblem
{
  /// State tangent dimension
  static constexpr Eigen::Index nx = G::SizeAtCompileTime;
  /// Input tangent dimension
  static constexpr Eigen::Index nu = U::SizeAtCompileTime;

  /// Initial state
  G x0 = G::Identity();

  /// Time horizon
  double T{1};

  /// Desired state trajectory
  std::function<G(double)> gdes = [](double) { return G::Identity(); };
  /// Desired input trajectory
  std::function<U(double)> udes = [](double) { return U::Identity(); };

  ///@{
  /// @brief Input bounds s.t. \f$ u_{min} \ominus e_U \leq u \ominus e_U \leq u_{max} \ominus e_U
  /// \f$
  std::optional<U> umin{}, umax{};
  ///@}

  ///@{
  /// @brief State bounds s.t. \f$ x_{min} \ominus e_G \leq g \ominus e_G \leq x_{max} \ominus e_G
  /// \f$
  std::optional<G> gmin{}, gmax{};
  ///@}

  /// Running state cost
  Eigen::Matrix<double, nx, nx> Q = Eigen::Matrix<double, nx, nx>::Identity();
  /// Final state cost
  Eigen::Matrix<double, nx, nx> QT = Eigen::Matrix<double, nx, nx>::Identity();
  /// Running input cost
  Eigen::Matrix<double, nu, nu> R = Eigen::Matrix<double, nu, nu>::Identity();
};

/**
 * @brief Convert OptimalControlProblem on \f$ (\mathbb{G}, \mathbb{U}) \f$ into a tangent space
 * QuadraticProgram on \f$ (\mathbb{R}^{\dim \mathfrak{g}}, \mathbb{R}^{\dim \mathfrak{u}}) \f$.
 *
 * The OptimalControlProblem is encoded into a QuadraticProgram via linearization around
 * \f$(g_{lin}(t), u_{lin}(t))\f$ followed by time discretization. The variables of the QP are \f[
 * \begin{bmatrix} \mu_0 & \mu_1 & \ldots & \mu_{K - 1} & x_1 & x_2 & \ldots & x_K
 * \end{bmatrix}, \f] where the discrete time index \f$k\f$ corresponds to time \f$t_k = k
 * \frac{T}{K} \f$ for \f$ k = 0, 1, \ldots, K \f$.
 *
 * The resulting QP has \f$ K \dim \mathfrak{g} + K \dim \mathfrak{u} \f$ variables and \f$ 2
 * K \dim \mathfrak{g} + K \dim \mathfrak{u} \f$ constraints.
 *
 * @note Given a solution \f$(x^*, \mu^*)\f$ to the QuadraticProgram, the corresponding
 * solution to the OptimalControlProblem is \f$ u^*(t) = u_{lin}(t) \oplus \mu^*(t) \f$ and the
 * optimal trajectory is \f$ g^*(t) = g_{lin}(t) \oplus x^*(t) \f$.
 *
 * @note Constraints are added as \f$ g_{min} \ominus g_{lin} \leq x \leq g_{max} \ominus g_{lin}
 * \f$ and similarly for the input. Beware of using constraints on non-Euclidean spaces.
 *
 * @note \p f and \p glin must be differentiable with the default smooth::diff method. If using an
 * automatic differentiation method this means that the functions must be templated on the scalar
 * type.
 *
 * @tparam K number of discretization steps
 * @tparam G problem state group type \f$ \mathbb{G} \f$
 * @tparam U problem input group type \f$ \mathbb{U} \f$
 *
 * @param pbm optimal control problem
 * @param f dynamics \f$ f : \mathbb{G} \times \mathbb{U} \rightarrow \mathbb{R}^{\dim \mathfrak
 * g}\f$ s.t. \f$ \mathrm{d}^r g_t = f(g, u) \f$
 * @param glin state trajectory \f$g_{lin}(t)\f$ to linearize around
 * @param ulin input trajectory \f$u_{lin}(t)\f$ to linearize around
 *
 * @return QuadraticProgram modeling the input optimal control problem.
 */
template<std::size_t K, LieGroup G, Manifold U, typename Dyn, typename GLin, typename ULin>
auto ocp_to_qp(const OptimalControlProblem<G, U> & pbm, Dyn && f, GLin && glin, ULin && ulin)
{
  using std::placeholders::_1;

  // problem info
  static constexpr int nx = G::SizeAtCompileTime;
  static constexpr int nu = U::SizeAtCompileTime;

  static constexpr int nX   = K * nx;
  static constexpr int nU   = K * nu;
  static constexpr int nvar = nX + nU;

  static constexpr int n_eq   = nX;  // equality constraints from dynamics
  static constexpr int n_u_iq = nU;  // input bounds
  static constexpr int n_x_iq = nX;  // state bounds
  static constexpr int ncon   = n_eq + n_u_iq + n_x_iq;

  using AT = Eigen::Matrix<double, nx, nx>;
  using BT = Eigen::Matrix<double, nx, nu>;
  using ET = Eigen::Matrix<double, nx, 1>;

  const double dt = pbm.T / static_cast<double>(K);

  QuadraticProgram<ncon, nvar> ret;

  // DYNAMICS CONSTRAINTS

  for (auto k = 0u; k < K; ++k) {
    const double t = k * dt;

    // LINEARIZATION

    Eigen::Matrix<double, 1, 1> t_vec(t);
    auto [xl, dxl] = diff::dr([&](const auto & v) { return glin(v(0)); }, wrt(t_vec));
    auto ul        = ulin(t);

    const auto [flin, df_xu] = diff::dr(f, wrt(xl, ul));

    // cltv system \dot x = At x(t) + Bt u(t) + Et
    const AT At = (-0.5 * G::ad(flin) - 0.5 * G::ad(dxl) + df_xu.template leftCols<nx>());
    const BT Bt = df_xu.template rightCols<nu>();
    const ET Et = flin - dxl;

    // TIME DISCRETIZATION

    const AT At2     = At * At;
    const AT At3     = At2 * At;
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;

    // dltv system x^+ = Ak x + Bk u + Ek by truncated taylor expansion of the matrix exponential
    const AT Ak = AT::Identity() + At * dt + At2 * dt2 / 2. + At3 * dt3 / 6.;
    const BT Bk = Bt * dt + At * Bt * dt2 / 2. + At2 * Bt * dt3 / 6.;
    const ET Ek = Et * dt + At * Et * dt2 / 2. + At2 * Et * dt3 / 6.;

    // DYNAMICS CONSTRANTS

    if (k == 0) {
      // x(1) - B u(0) = A x0 + E
      ret.A.template block<nx, nx>(nx * k, nU + nx * k).setIdentity();
      ret.A.template block<nx, nu>(nx * k, nu * k) = -Bk;
      ret.u.template segment<nx>(nx * k)           = Ak * (pbm.x0 - glin(t)) + Ek;
      ret.l.template segment<nx>(nx * k)           = ret.u.template segment<nx>(nx * k);

    } else {
      // x(k) - A x(k-1) - B u(k-1) = E
      ret.A.template block<nx, nx>(nx * k, nU + nx * k).setIdentity();
      ret.A.template block<nx, nx>(nx * k, nU + nx * (k - 1)) = -Ak;
      ret.A.template block<nx, nu>(nx * k, nu * k)            = -Bk;
      ret.u.template segment<nx>(nx * k)                      = Ek;
      ret.l.template segment<nx>(nx * k)                      = Ek;
    }
  }

  // INPUT CONSTRAINTS

  ret.A.template block<nU, nU>(n_eq, 0).setIdentity();
  if (pbm.umin) {
    for (auto k = 0u; k < K; ++k) {
      ret.l.template segment<nu>(n_eq + k * nu) = pbm.umin.value() - ulin(k * dt);
    }
  } else {
    ret.l.template segment<nU>(n_eq).setConstant(-std::numeric_limits<double>::infinity());
  }
  if (pbm.umax) {
    for (auto k = 0u; k < K; ++k) {
      ret.u.template segment<nu>(n_eq + k * nu) = pbm.umax.value() - ulin(k * dt);
    }
  } else {
    ret.u.template segment<nU>(n_eq).setConstant(std::numeric_limits<double>::infinity());
  }

  // STATE CONSTRAINTS

  ret.A.template block<nX, nX>(n_eq + n_u_iq, nU).setIdentity();
  if (pbm.gmin) {
    for (auto k = 1u; k < K; ++k) {
      ret.l.template segment<nx>(n_eq + n_u_iq + (k - 1) * nx) = pbm.gmin.value() - glin(k * dt);
    }
  } else {
    ret.l.template segment<nX>(n_eq + n_u_iq).setConstant(-std::numeric_limits<double>::infinity());
  }
  if (pbm.gmax) {
    for (auto k = 1u; k < K; ++k) {
      ret.u.template segment<nx>(n_eq + n_u_iq + (k - 1) * nx) = pbm.gmax.value() - glin(k * dt);
    }
  } else {
    ret.u.template segment<nX>(n_eq + n_u_iq).setConstant(std::numeric_limits<double>::infinity());
  }

  // INPUT COSTS

  for (auto k = 0u; k < K; ++k) {
    ret.P.template block<nu, nu>(k * nu, k * nu) = pbm.R * dt;
    ret.q.template segment<nu>(k * nu)           = pbm.R * (ulin(k * dt) - pbm.udes(k * dt));
  }

  // STATE COSTS

  for (auto k = 1u; k < K; ++k) {
    ret.P.template block<nx, nx>(nU + (k - 1) * nx, nU + (k - 1) * nx) = pbm.Q * dt;
    ret.q.template segment<nx>(nU + (k - 1) * nx) = pbm.Q * (glin(k * dt) - pbm.gdes(k * dt));
  }
  ret.P.template block<nx, nx>(nU + (K - 1) * nx, nU + (K - 1) * nx) = pbm.QT;
  ret.q.template segment<nx>(nU + (K - 1) * nx) = pbm.QT * (glin(pbm.T) - pbm.gdes(pbm.T));

  return ret;
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__MPC_HPP_
