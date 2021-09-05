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

#include <smooth/diff.hpp>
#include <smooth/lie_group.hpp>

#include "mpc.hpp"
#include "qp.hpp"

namespace smooth::feedback {

/**
 * @brief Active set invariance problem
 */
template<LieGroup G, Manifold U>
struct ASIFProblem
{
  /// time horizon
  double T{1};

  /// initial state
  G x0 = Identity<G>();

  /// desired input
  U u_des = Identity<U>();

  /// input bounds
  OptimalControlBounds<U> ulim{};
};

/**
 * @brief Parameters for asif_to_qp
 */
struct ASIFtoQPParams
{
  /// Barrier function time constant
  double alpha{1};

  /// maximal integration time step
  double dt{0.1};

  /// relaxation cost
  double relax_cost{100};
};

/**
 * @brief Convert an active set invariance problem to a QuadraticProgram.
 *
 * The objective is to impose constraints on the current input \f$ u \f$ of a system \f$
 * \mathrm{d}^r x_t = f(x, u) \f$ s.t.
 * \f[
 *    \frac{\mathrm{d}}{\mathrm{d}t} h(\phi(t; x_0, bu(\cdot)))
 *    \geq \alpha h(\phi(t; x_0, bu(\cdot)))
 * \f]
 * which enforces forward invariance of the set \f$ \{ x
 * : h(t, x) \geq 0 \} \f$ along the **backup trajectory** \f$ bu \f$. The constraint is enforced at
 * \f$K\f$ look-ahead time steps \f$ t_k = k \tau\f$ for \f$k = 0, \ldots, K \f$.
 *
 * This function encodes the problem as a QuadraticProgram that solves
 * \f[
 *  \begin{cases}
 *   \min_{\mu}  & \left\| u - u_{des} \right\|^2 \\
 *   \text{s.t.} & \text{constraint above holds for } u = u_{lin} + \mu
 *   \end{cases}
 * \f]
 * A solution \f$ \mu^* \f$ to the QuadraticProgram corresponds to an input \f$ u_{lin} \oplus \mu^*
 * \f$ applied to the system.
 *
 * @tparam K number of constraints (\p ASIFtoQPParams::tau controls time between constraints)
 * @tparam G state Lie group type \f$\mathbb{G}\f$
 * @tparam U input Lie group type \f$\mathbb{G}\f$
 *
 * @param x0 current state of the system
 * @param u_des desired system input
 * @param f system model \f$f : \mathbb{R} \times \mathbb{G} \times \mathbb{U} \rightarrow
 * \mathbb{R}^{\dim \mathfrak g}\f$ s.t. \f$ \mathrm{d}^r x_t = f(t, x, u) \f$
 * @param h safe set \f$h : \mathbb{R} \times \mathbb{G} \rightarrow \mathbb{R}^{n_h}\f$ s.t. \f$
 * S(t) = \{ h(t, x) \geq 0 \} \f$ denotes the safe set at time \f$ t \f$
 * @param bu backup controller \f$ub : \mathbb{R} \times \mathbb{G} \rightarrow \mathbb{U} \f$
 * @param prm asif parameters
 * @param u_lin input to linearize around (defaults to identity)
 *
 * @return QuadraticProgram modeling the ASIF filtering problem
 *
 * \note Differentiability requirements:
 *   * f differentiable w.r.t. x and u
 *   * h differentiable w.r.t. t and x
 *   * bu differentiable w.r.t. x
 *
 * \note For the common case of \f$ U = \mathbb{R}^n \f$ and control-affine dynamics on the form \f$
 * f(t, x, u) = f_x(t, x) + f_u(t, x) u \f$, the linearization point \f$u_{lin}\f$ can safely be
 * left to the default value.
 */
template<std::size_t K, LieGroup G, Manifold U, typename Dyn, typename SafeSet, typename BackupU>
auto asif_to_qp(const ASIFProblem<G, U> & pbm,
  Dyn && f,
  SafeSet && h,
  BackupU && bu,
  const ASIFtoQPParams & prm,
  const U & u_lin = Identity<U>())
{
  using boost::numeric::odeint::euler, boost::numeric::odeint::vector_space_algebra;
  using std::placeholders::_1;

  static constexpr int nx = Dof<G>;
  static constexpr int nu = Dof<U>;
  static constexpr int nh = std::invoke_result_t<SafeSet, double, G>::SizeAtCompileTime;

  static_assert(nx > 0, "State space dimension must be static");
  static_assert(nu > 0, "Input space dimension must be static");
  static_assert(nh > 0, "Safe set dimension must be static");

  euler<G, double, Tangent<G>, double, vector_space_algebra> state_stepper{};
  euler<TangentMap<G>, double, TangentMap<G>, double, vector_space_algebra> sensi_stepper{};

  const int nu_ineq = pbm.ulim.A.rows();

  const int M            = K * nh + nu_ineq + 1;
  static constexpr int N = nu + 1;

  // iteration variables
  const double tau     = pbm.T / static_cast<double>(K);
  const double dt      = std::min<double>(prm.dt, tau);
  double t             = 0;
  G x                  = pbm.x0;
  TangentMap<G> dx_dx0 = TangentMap<G>::Identity();

  // define ODEs for closed-loop dynamics and its sensitivity
  const auto x_ode = [&f, &bu](
                       const G & xx, Tangent<G> & dd, double tt) { dd = f(tt, xx, bu(tt, xx)); };

  const auto dx_dx0_ode = [&f, &bu, &x](const auto & S_v, auto & dS_dt_v, double tt) {
    auto f_cl = [&]<typename T>(const CastT<T, G> & vx) { return f(T(tt), vx, bu(T(tt), vx)); };
    const auto [fcl, dr_fcl_dx] = diff::dr(std::move(f_cl), wrt(x));
    dS_dt_v                     = (-ad<G>(fcl) + dr_fcl_dx) * S_v;
  };

  // value of dynamics at call time
  const auto [f0, d_f0_du] = diff::dr(
    [&]<typename T>(const CastT<T, U> & vu) { return f(T(t), cast<T>(x), vu); }, wrt(u_lin));

  QuadraticProgram<-1, -1> ret;

  ret.A.resize(M, N);
  ret.l.resize(M);
  ret.u.resize(M);

  ret.P.resize(N, N);
  ret.q.resize(N);

  // loop over constraint number
  for (auto k = 0u; k != K; ++k) {
    // differentiate barrier function w.r.t. x
    const auto [hval, dh_dtx] = diff::dr(
      [&h]<typename T>(const T & vt, const CastT<T, G> & vx) { return h(vt, vx); }, wrt(t, x));

    const Eigen::Matrix<double, nh, 1> dh_dt  = dh_dtx.template leftCols<1>();
    const Eigen::Matrix<double, nh, nx> dh_dx = dh_dtx.template rightCols<nx>();

    // insert barrier constraint
    const Eigen::Matrix<double, nh, nx> dh_dx0 = dh_dx * dx_dx0;
    ret.A.template block<nh, nu>(k * nh, 0)    = dh_dx0 * d_f0_du;
    ret.l.template segment<nh>(k * nh)         = -dh_dt - prm.alpha * hval - dh_dx0 * f0;
    ret.u.template segment<nh>(k * nh).setConstant(std::numeric_limits<double>::infinity());

    // integrate system and sensitivity forward until next constraint
    double dt_act = std::min(dt, tau * (k + 1) - t);
    while (t < tau * (k + 1)) {
      state_stepper.do_step(x_ode, x, t, dt_act);
      sensi_stepper.do_step(dx_dx0_ode, dx_dx0, t, dt_act);
      t += dt_act;
    }
  }

  // relaxation of barrier constraints
  ret.A.template block<K * nh, 1>(0, nu).setConstant(1);

  // input bounds
  ret.A.block(K * nh, 0, nu_ineq, nu) = pbm.ulim.A;
  ret.A.block(K * nh, nu, nu_ineq, 1).setZero();
  ret.l.segment(K * nh, nu_ineq) = pbm.ulim.l - pbm.ulim.A * (u_lin - Identity<U>());
  ret.u.segment(K * nh, nu_ineq) = pbm.ulim.u - pbm.ulim.A * (u_lin - Identity<U>());

  // upper and lower bounds on delta
  ret.A.row(K * nh + nu_ineq).setZero();
  ret.A(K * nh + nu_ineq, nu) = 1;
  ret.l(K * nh + nu_ineq)     = 0;
  ret.u(K * nh + nu_ineq)     = std::numeric_limits<double>::infinity();

  ret.P.setIdentity();
  ret.P(nu, nu)             = prm.relax_cost;
  ret.q.template head<nu>() = (u_lin - pbm.u_des);
  ret.q(nu)                 = 0;

  return ret;
}

struct ASIFParams
{
  /// Horizon
  double T{1};
  /// ASIF algorithm parameters
  ASIFtoQPParams asif;
  /// QP solver parameters
  QPSolverParams qp;
};

template<std::size_t K,
  LieGroup G,
  Manifold U,
  typename Dyn,
  typename SS,
  typename BU,
  smooth::diff::Type DiffType = smooth::diff::Type::DEFAULT>
class ASIF
{
public:
  ASIF(
    Dyn && f, SS && h, BU && bu, OptimalControlBounds<U> && ulim, ASIFParams && prm = ASIFParams{})
      : f_(std::move(f)), h_(std::move(h)), bu_(std::move(bu)), ulim_(std::move(ulim)), prm_(prm)
  {}

  ASIF(const Dyn & f,
    const SS & h,
    const BU & bu,
    const OptimalControlBounds<U> & ulim,
    const ASIFParams & prm = ASIFParams{})
      : ASIF(Dyn(f), SS(h), BU(bu), OptimalControlBounds<U>(ulim), ASIFParams(prm))
  {}

  QPSolutionStatus operator()(U & u, double t, const G & g)
  {
    using std::chrono::duration, std::chrono::nanoseconds;

    auto f = [this, &t]<typename T>(T t_loc, const CastT<T, G> & vx, const CastT<T, U> & vu) {
      return f_(T(t) + t_loc, vx, vu);
    };
    auto h = [this, &t]<typename T>(
               T t_loc, const CastT<T, G> & vx) { return h_(T(t) + t_loc, vx); };
    auto bu = [this, &t]<typename T>(
                T t_loc, const CastT<T, G> & vx) { return bu_(T(t) + t_loc, vx); };

    ASIFProblem<G, U> pbm{
      .T     = prm_.T,
      .x0    = g,
      .u_des = u,
      .ulim  = ulim_,
    };

    const U u_lin = u;

    auto qp  = smooth::feedback::asif_to_qp<K>(pbm, f, h, bu, prm_.asif, u_lin);
    auto sol = smooth::feedback::solve_qp(qp, prm_.qp, warmstart_);

    u = rplus(u_lin, sol.primal.template head<Dof<U>>());

    if (sol.code == QPSolutionStatus::Optimal) { warmstart_ = sol; }
    return sol.code;
  }

private:
  Dyn f_;
  SS h_;
  BU bu_;

  OptimalControlBounds<U> ulim_;

  ASIFParams prm_;

  std::optional<QPSolution<-1, -1, double>> warmstart_;
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__ASIF_HPP_
