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

#ifndef SMOOTH__FEEDBACK__OCP_HPP_
#define SMOOTH__FEEDBACK__OCP_HPP_

/**
 * @file
 * @brief Optimal control problem definition.
 */

#include <Eigen/Core>
#include <smooth/lie_group.hpp>

#include "traits.hpp"

namespace smooth::feedback {

/**
 * @brief Optimal control problem definition
 * @tparam _X state space
 * @tparam _U input space
 *
 * Problem is defined on the interval \f$ t \in [0, t_f] \f$.
 * \f[
 * \begin{cases}
 *  \min              & \theta(t_f, x_0, x_f, q)                                         \\
 *  \text{s.t.}       & x(0) = x_0                                                       \\
 *                    & x(t_f) = x_f                                                     \\
 *                    & \dot x(t) = f(t, x(t), u(t))                                     \\
 *                    & q = \int_{0}^{t_f} g(t, x(t), u(t)) \mathrm{d}t                  \\
 *                    & c_{rl} \leq c_r(t, x(t), u(t)) \leq c_{ru} \quad t \in [0, t_f]  \\
 *                    & c_{el} \leq c_e(t_f, x_0, x_f, q) \leq c_{eu}
 * \end{cases}
 * \f]
 *
 * The optimal control problem depends on arbitrary functions \f$ \theta, f, g, c_r, c_e \f$.
 * The type of those functions are template pararamters in this structure.
 *
 * @note To enable automatic differentiation \f$ \theta, f, g, c_r, c_e \f$ must be templated over
 * the scalar type.
 */
template<LieGroup _X, Manifold _U, typename Theta, typename F, typename G, typename CR, typename CE>
struct OCP
{
  using X = _X;
  using U = _U;

  /// @brief State space dimension
  static constexpr int Nx = Dof<X>;
  /// @brief Input space dimension
  static constexpr int Nu = Dof<U>;
  /// @brief Number of integrals
  static constexpr int Nq = std::invoke_result_t<G, double, X, U>::SizeAtCompileTime;
  /// @brief Number of running constraints
  static constexpr int Ncr = std::invoke_result_t<CR, double, X, U>::SizeAtCompileTime;
  /// @brief Number of end constraints
  static constexpr int Nce =
    std::invoke_result_t<CE, double, X, X, Eigen::Matrix<double, Nq, 1>>::SizeAtCompileTime;

  static_assert(Nx > 0, "Static size required");
  static_assert(Nu > 0, "Static size required");
  static_assert(Nq > 0, "Static size required");
  static_assert(Ncr > 0, "Static size required");
  static_assert(Nce > 0, "Static size required");

  /// @brief Objective function \f$ \theta : R \times X \times X \times R^{n_q} \rightarrow R \f$
  Theta theta;

  /// @brief System dynamics \f$ f : R \times X \times U \rightarrow Tangent<X> \f$
  F f;
  /// @brief Integrals \f$ g : R \times X \times U \rightarrow R^{n_q} \f$
  G g;

  /// @brief Running constraint \f$ c_r : R \times X \times U \rightarrow R^{n_{cr}} \f$
  CR cr;
  /// @brief Running constraint lower bound \f$ c_{rl} \in R^{n_{cr}} \f$
  Eigen::Vector<double, Ncr> crl;
  /// @brief Running constraint upper bound \f$ c_{ru} \in R^{n_{cr}} \f$
  Eigen::Vector<double, Ncr> cru;

  /// @brief End constraint \f$ c_e : R \times X \times X \times R^{n_q} \rightarrow R^{n_{ce}} \f$
  CE ce;
  /// @brief End constraint lower bound \f$ c_{el} \in R^{n_{ce}} \f$
  Eigen::Vector<double, Nce> cel;
  /// @brief End constraint upper bound \f$ c_{eu} \in R^{n_{ce}} \f$
  Eigen::Vector<double, Nce> ceu;
};

/// @brief Concept that is true for OCP specializations
template<typename T>
concept OCPType = traits::is_specialization_of_v<T, OCP>;

/// @brief Concept that is true for FlatOCP specializations
template<typename T>
concept FlatOCPType =
  OCPType<T> &&(smooth::traits::RnType<typename T::X> && smooth::traits::RnType<typename T::U>);

/**
 * @brief Solution to OCP problem.
 */
template<LieGroup X, Manifold U, int Nq, int Ncr, int Nce>
struct OCPSolution
{
  double t0;
  double tf;

  /// @brief Integral values
  Eigen::Vector<double, Nq> Q{};

  /// @brief Callable functions for state and input
  std::function<U(double)> u;
  std::function<X(double)> x;

  /// @brief Multipliers for integral constraints
  Eigen::Vector<double, Nq> lambda_q{};

  /// @brief Multipliers for endpoint constraints
  Eigen::Vector<double, Nce> lambda_ce{};

  /// @brief Multipliers for dynamics equality constraint
  std::function<Eigen::Vector<double, Dof<X>>(double)> lambda_dyn{};

  /// @brief Multipliers for active running constraints
  std::function<Eigen::Vector<double, Ncr>(double)> lambda_cr{};
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__OCP_HPP_
