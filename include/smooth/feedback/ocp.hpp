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

  /// @brief State dimension \f$ n_{x} \f$
  std::size_t nx;
  /// @brief Input dimension \f$ n_{u} \f$
  std::size_t nu;
  /// @brief Number of integrals \f$ n_{q} \f$
  std::size_t nq;
  /// @brief Number of running constraints \f$ n_{cr} \f$
  std::size_t ncr;
  /// @brief Number of end constraints \f$ n_{ce} \f$
  std::size_t nce;

  /// @brief Objective function \f$ \theta : R \times X \times X \times R^{n_q} \rightarrow R \f$
  Theta theta;

  /// @brief System dynamics \f$ f : R \times X \times U \rightarrow Tangent<X> \f$
  F f;
  /// @brief Integrals \f$ g : R \times X \times U \rightarrow R^{n_q} \f$
  G g;

  /// @brief Running constraint \f$ c_r : R \times X \times U \rightarrow R^{n_{cr}} \f$
  CR cr;
  /// @brief Running constraint lower bound \f$ c_{rl} \in R^{n_{cr}} \f$
  Eigen::VectorXd crl;
  /// @brief Running constraint upper bound \f$ c_{ru} \in R^{n_{cr}} \f$
  Eigen::VectorXd cru;

  /// @brief End constraint \f$ c_e : R \times X \times X \times R^{n_q} \rightarrow R^{n_{ce}} \f$
  CE ce;
  /// @brief End constraint lower bound \f$ c_{el} \in R^{n_{ce}} \f$
  Eigen::VectorXd cel;
  /// @brief End constraint upper bound \f$ c_{eu} \in R^{n_{ce}} \f$
  Eigen::VectorXd ceu;
};

/// @brief Concept that is true for OCP specializations
template<typename T>
concept OCPType = traits::is_specialization_of_v<T, OCP>;

/// @brief OCP defined on flat spaces
template<typename Theta, typename F, typename G, typename CR, typename CE>
using FlatOCP = OCP<Eigen::VectorXd, Eigen::VectorXd, Theta, F, G, CR, CE>;

/// @brief Concept that is true for FlatOCP specializations
template<typename T>
concept FlatOCPType =
  OCPType<T> &&(smooth::traits::RnType<typename T::X> && smooth::traits::RnType<typename T::U>);

/**
 * @brief Check if an OCP is properly defined.
 */
inline bool check_ocp(const OCPType auto & ocp)
{
  using X = typename std::decay_t<decltype(ocp)>::X;
  using U = typename std::decay_t<decltype(ocp)>::U;

  const double t = 0;
  const X x      = Default<X>(ocp.nx);
  const U u      = Default<U>(ocp.nu);

  if (!(static_cast<std::size_t>(dof(x)) == ocp.nx)) { return false; }
  if (!(static_cast<std::size_t>(dof(u)) == ocp.nu)) { return false; }

  const auto dx = ocp.f.template operator()<double>(t, x, u);
  if (!(static_cast<std::size_t>(dx.size()) == ocp.nx)) { return false; }

  const auto g = ocp.g.template operator()<double>(t, x, u);
  if (!(static_cast<std::size_t>(g.size()) == ocp.nq)) { return false; }

  [[maybe_unused]] const double obj = ocp.theta.template operator()<double>(t, x, x, g);

  const auto cr = ocp.cr.template operator()<double>(t, x, u);
  if (!(static_cast<std::size_t>(cr.size()) == ocp.ncr)) { return false; }
  if (!(static_cast<std::size_t>(ocp.crl.size()) == ocp.ncr)) { return false; }
  if (!(static_cast<std::size_t>(ocp.cru.size()) == ocp.ncr)) { return false; }

  const auto ce = ocp.ce.template operator()<double>(t, x, x, g);
  if (!(static_cast<std::size_t>(ce.size()) == ocp.nce)) { return false; }
  if (!(static_cast<std::size_t>(ocp.cel.size()) == ocp.nce)) { return false; }
  if (!(static_cast<std::size_t>(ocp.ceu.size()) == ocp.nce)) { return false; }

  return true;
}

/**
 * @brief Solution to OCP problem.
 */
template<LieGroup X, Manifold U>
struct OCPSolution
{
  double t0;
  double tf;

  /// @brief Integral values
  Eigen::VectorXd Q;

  /// @brief Callable functions for state and input
  std::function<U(double)> u;
  std::function<X(double)> x;

  /// @brief Multipliers for integral constraints
  Eigen::VectorXd lambda_q;

  /// @brief Multipliers for endpoint constraints
  Eigen::VectorXd lambda_ce;

  /// @brief Multipliers for dynamics equality constraint
  std::function<Eigen::VectorXd(double)> lambda_dyn;

  /// @brief Multipliers for active running constraints
  std::function<Eigen::VectorXd(double)> lambda_cr;
};

/// @brief Solution to OCP problem defined on flat spaces
using FlatOCPSolution = OCPSolution<Eigen::VectorXd, Eigen::VectorXd>;

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__OCP_HPP_
