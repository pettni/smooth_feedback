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

#ifndef SMOOTH__FEEDBACK__ASIF_HPP_
#define SMOOTH__FEEDBACK__ASIF_HPP_

/**
 * @file
 * @brief Active Set Invariance Filtering (ASIF) on Lie groups.
 */

#include <Eigen/Core>
#include <boost/numeric/odeint.hpp>
#include <smooth/compat/odeint.hpp>

#include <smooth/concepts.hpp>
#include <smooth/diff.hpp>

#include "qp.hpp"

namespace smooth::feedback {

/**
 * @brief Parameters for asif_to_qp
 */
struct AsifParams
{
  /// Barrier function time constant
  float alpha = 1.;

  /// time between barrier constraints
  float dt = 1;

  /// integration time step
  float dt_int = 0.1;

  /// relaxation cost
  float relax_cost = 100;
};

/**
 * @brief Pose active set invariance problem as a QuadraticProgram.
 *
 * @tparam K number of constraints.
 * @tparam G state Lie group type \f$\mathbb{G}\f$
 * @tparam U input Lie group type \f$\mathbb{G}\f$
 *
 * @param x0 current state of the system
 * @param u_des desired system input
 * @param u_lin system input to linearize around
 * @param f system model \f$f : \mathbb{G} \times \mathbb{U} \rightarrow \mathbb{R}^{\dim \mathfrak
 * g}\f$ s.t. \f$ \mathrm{d}^r x_t = f(x, u) \f$
 * @param h safe set \f$h : \mathbb{G} \rightarrow \mathbb{R}^{n_h}\f$ s.t. \f$ h(x) \geq 0 \f$
 * denote safe set
 * @param bu backup controller \f$ub : \mathbb{G} \rightarrow \mathbb{U} \f$
 * @param prm asif parameters
 *
 * @return QuadraticProgram modeling the ASIF filtering problem
 *
 * An solution \f$ \mu \f$ to the QuadraticProgram corresponds to an input \f$ u_lin \oplus \mu \f$
 * applied to the system.
 */
template<std::size_t K, LieGroup G, Manifold U, typename Dyn, typename SafeSet, typename BackupU>
auto asif_to_qp(const G & x0,
  const U & u_des,
  U & u_lin,
  Dyn && f,
  SafeSet && h,
  BackupU && bu,
  const AsifParams & prm)
{
  using boost::numeric::odeint::euler, boost::numeric::odeint::vector_space_algebra;
  using std::placeholders::_1;
  using Tangent    = typename G::Tangent;
  using TangentMap = typename G::TangentMap;

  static constexpr int nx = G::SizeAtCompileTime;
  static constexpr int nu = U::SizeAtCompileTime;
  static constexpr int nh = std::invoke_result_t<SafeSet, G>::SizeAtCompileTime;

  euler<G, double, Tangent, double, vector_space_algebra> state_stepper{};
  euler<TangentMap, double, TangentMap, double, vector_space_algebra> sensi_stepper{};

  static constexpr int M = K * nh + nu;
  static constexpr int N = nu + 1;

  // iteration variables
  double t                      = 0;
  G x                           = x0;
  typename G::TangentMap dx_dx0 = G::TangentMap::Identity();

  // define ODEs for closed-loop dynamics and its sensitivity
  const auto x_ode = [&f, &bu](
                       const auto & x_v, auto & dx_dt_v, double) { dx_dt_v = f(x_v, bu(x_v)); };
  const auto dx_dx0_ode = [&f, &bu, &x](const auto & S_v, auto & dS_dt_v, double) {
    const auto [fcl, dr_fcl_dx] =
      diff::dr([&f, &bu, &x](const auto & x_v) { return f(x_v, bu(x_v)); }, wrt(x));
    dS_dt_v = (-G::ad(fcl) + dr_fcl_dx) * S_v;
  };

  // value of dynamics at call time
  const auto [f0, d_f0_dulin] = diff::dr(std::bind(f, x, _1), wrt(u_lin));

  QuadraticProgram<M, N> ret;
  ret.A.template block<K * nh, 1>(0, nu).setConstant(1);   // coeffs for delta (upper part)
  ret.A.template block<nu, 1>(K * nh, nu).setConstant(0);  // coeffs for delta (lower part)

  // loop over constraints on input
  for (auto k = 0u; k != K; ++k) {
    // differentiate barrier function w.r.t. x(t)
    const auto [hval, dh_dx] = diff::dr([&](const auto & x_v) { return h(x_v); }, wrt(x));

    // insert barrier constraint on u
    Eigen::Matrix<double, nh, nx> dh_dx0    = dh_dx * dx_dx0;
    ret.A.template block<nh, nu>(k * nh, 0) = dh_dx0 * d_f0_dulin;
    ret.l.template segment<nh>(k * nh)      = -prm.alpha * hval - dh_dx0 * f0;
    ret.u.template segment<nh>(k * nh).setZero();

    // integrate system and sensitivity forward
    while (t < prm.dt * (k + 1)) {
      state_stepper.do_step(x_ode, x, t, prm.dt_int);
      sensi_stepper.do_step(dx_dx0_ode, dx_dx0, t, prm.dt_int);
      t += prm.dt_int;
    }
  }

  // set input constraints
  ret.A.template bottomRows<nu>().setIdentity();
  ret.l.template tail<nu>().setConstant(-1);
  ret.u.template tail<nu>().setConstant(1);

  ret.P.setIdentity();
  ret.P(nu, nu)             = prm.relax_cost;
  ret.q.template head<nu>() = (u_des - u_lin);
  ret.q(nu)                 = 0;

  return ret;
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__ASIF_HPP_
