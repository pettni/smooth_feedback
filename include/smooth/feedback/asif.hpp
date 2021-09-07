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
 * @brief Active Set Invariance (ASI) filtering on Lie groups.
 */

#include <Eigen/Core>
#include <boost/numeric/odeint.hpp>
#include <smooth/compat/odeint.hpp>

#include <smooth/diff.hpp>
#include <smooth/lie_group.hpp>

#include "common.hpp"
#include "qp.hpp"

namespace smooth::feedback {

/**
 * @brief Active set invariance problem definition.
 *
 * The active set invariance problem is
 * \f[
 * \begin{cases}
 *  \min_u        & (u \ominus u_{des})' W_u (u \ominus u_{des})  \\
 *  \text{s.t.}   & x(0) = x_0 \\
 *                & h(x(t)) \geq 0, \quad t \in [0, T]    \\
 * \end{cases}
 * \f]
 * for a system \f$ \mathrm{d}^r x_t = f(x(t), u(t)) \f$.
 */
template<LieGroup G, Manifold U>
struct ASIFProblem
{
  /// time horizon
  double T{1};
  /// initial state
  G x0;
  /// desired input
  U u_des;
  /// weights on desired input
  Eigen::Matrix<double, Dof<U>, 1> W_u = Eigen::Matrix<double, Dof<U>, 1>::Ones();
  /// input bounds
  ManifoldBounds<U> ulim{};
};

/**
 * @brief Parameters for asif_to_qp
 */
struct ASIFtoQPParams
{
  /// barrier function time constant s.t. \f$ \dot h - \alpha h \geq 0 \f$.
  double alpha{1};
  /// maximal integration time step
  double dt{0.1};
  /// relaxation cost
  double relax_cost{100};
};

/**
 * @brief Convert a ASIFProblem to a QuadraticProgram.
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
 *   \text{s.t.} & \text{constraint above holds for } u = u_{des} + \mu
 *   \end{cases}
 * \f]
 * A solution \f$ \mu^* \f$ to the QuadraticProgram corresponds to an input \f$ u_{des} \oplus \mu^*
 * \f$ applied to the system.
 *
 * @tparam K number of constraints (\p ASIFtoQPParams::tau controls time between constraints)
 * @tparam G state LieGroup type \f$\mathbb{G}\f$
 * @tparam U input Manifold type \f$\mathbb{G}\f$
 *
 * @param pbm problem definition
 * @param prm algorithm parameters
 * @param f system model \f$f : \mathbb{R} \times \mathbb{G} \times \mathbb{U} \rightarrow
 * \mathbb{R}^{\dim \mathfrak g}\f$ s.t. \f$ \mathrm{d}^r x_t = f(t, x, u) \f$
 * @param h safe set \f$h : \mathbb{R} \times \mathbb{G} \rightarrow \mathbb{R}^{n_h}\f$ s.t. \f$
 * S(t) = \{ h(t, x) \geq 0 \} \f$ denotes the safe set at time \f$ t \f$
 * @param bu backup controller \f$ub : \mathbb{R} \times \mathbb{G} \rightarrow \mathbb{U} \f$
 *
 * \note The algorithm relies on automatic differentiation. The following supplied functions must be
 * differentiable (i.e. be templated on the scalar type if an automatic differentiation method is
 * selected):
 *   * \p f differentiable w.r.t. x and u
 *   * h differentiable w.r.t. t and x
 *   * bu differentiable w.r.t. x
 *
 * @return QuadraticProgram modeling the ASIF filtering problem
 */
template<std::size_t K,
  LieGroup G,
  Manifold U,
  typename Dyn,
  typename SafeSet,
  typename BackupU,
  diff::Type DT = diff::Type::DEFAULT>
auto asif_to_qp(
  const ASIFProblem<G, U> & pbm, const ASIFtoQPParams & prm, Dyn && f, SafeSet && h, BackupU && bu)
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
  const int M       = K * nh + nu_ineq + 1;
  const int N       = nu + 1;

  QuadraticProgram<-1, -1> ret;

  ret.A.resize(M, N);
  ret.l.resize(M);
  ret.u.resize(M);

  ret.P.resize(N, N);
  ret.q.resize(N);

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
    const auto [fcl, dr_fcl_dx] = diff::dr<DT>(std::move(f_cl), wrt(x));
    dS_dt_v                     = (-ad<G>(fcl) + dr_fcl_dx) * S_v;
  };

  // value of dynamics at call time
  const auto [f0, d_f0_du] = diff::dr<DT>(
    [&]<typename T>(const CastT<T, U> & vu) { return f(T(t), cast<T>(x), vu); }, wrt(pbm.u_des));

  // loop over constraint number
  for (auto k = 0u; k != K; ++k) {
    // differentiate barrier function w.r.t. x
    const auto [hval, dh_dtx] = diff::dr<DT>(
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
  ret.l.segment(K * nh, nu_ineq) = pbm.ulim.l - pbm.ulim.A * rminus(pbm.u_des, pbm.ulim.c);
  ret.u.segment(K * nh, nu_ineq) = pbm.ulim.u - pbm.ulim.A * rminus(pbm.u_des, pbm.ulim.c);

  // upper and lower bounds on delta
  ret.A.row(K * nh + nu_ineq).setZero();
  ret.A(K * nh + nu_ineq, nu) = 1;
  ret.l(K * nh + nu_ineq)     = 0;
  ret.u(K * nh + nu_ineq)     = std::numeric_limits<double>::infinity();

  ret.P.template block<nu, nu>(0, 0) = pbm.W_u.asDiagonal();
  ret.P.template block<nu, 1>(0, nu).setZero();
  ret.q.template head<nu>().setZero();

  ret.P.row(nu).setZero();
  ret.P(nu, nu) = prm.relax_cost;
  ret.q(nu)     = 0;

  return ret;
}

/**
 * @brief ASIFilter filter parameters
 */
template<Manifold U>
struct ASIFilterParams
{
  /// Horizon
  double T{1};
  /// Weights on desired input
  Eigen::Matrix<double, Dof<U>, 1> u_weight = Eigen::Matrix<double, Dof<U>, 1>::Ones();
  /// Input bounds
  ManifoldBounds<U> u_lim{};
  /// ASIFilter algorithm parameters
  ASIFtoQPParams asif;
  /// QP solver parameters
  QPSolverParams qp;
};

/**
 * @brief ASI Filter
 *
 * Thin wrapper around asif_to_qp() and solve_qp() that keeps track of the most
 * recent solution for warmstarting, and facilitates working with time-varying
 * problems.
 */
template<std::size_t K,
  LieGroup G,
  Manifold U,
  typename Dyn,
  typename SS,
  typename BU,
  diff::Type DT = diff::Type::DEFAULT>
class ASIFilter
{
public:
  /**
   * @brief Construct an ASI filter
   *
   * @param f dynamics as function \f$ \mathbb{R} \times \mathbb{G} \times \mathbb{U} \rightarrow
   * \mathbb{R}^{\dim \mathbb{G}} \f$
   * @param h safety set definition as function \f$ \mathbb{G} \rightarrow \mathbb{R}^{n_h} \f$
   * @param bu backup controller as function \f$ \mathbb{R} \times \mathbb{G} \rightarrow \mathbb{U}
   * \f$
   * @param prm filter parameters
   *
   * @note These functions are defined in global time. As opposed to MPC the time
   * variable must be a \p double since differentiation of bu w.r.t. t is required.
   */
  ASIFilter(const Dyn & f,
    const SS & h,
    const BU & bu,
    const ASIFilterParams<U> & prm = ASIFilterParams<U>{})
      : ASIFilter(Dyn(f), SS(h), BU(bu), ASIFilterParams<U>(prm))
  {}

  /**
   * @brief Construct an ASI filter (rvalue version).
   */
  ASIFilter(Dyn && f, SS && h, BU && bu, ASIFilterParams<U> && prm = ASIFilterParams<U>{})
      : f_(std::move(f)), h_(std::move(h)), bu_(std::move(bu)), prm_(prm)
  {}

  /**
   * @brief Filter an input
   *
   * @param[in, out] u desired input in, filter output out
   * @param[in] t current global time
   * @param[in] g current state
   */
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
      .W_u   = prm_.u_weight,
      .ulim  = ulim_,
    };

    auto qp = feedback::asif_to_qp<K, G, U, decltype(f), decltype(h), decltype(bu), DT>(
      pbm, prm_.asif, std::move(f), std::move(h), std::move(bu));
    auto sol = feedback::solve_qp(qp, prm_.qp, warmstart_);

    u = rplus(u, sol.primal.template head<Dof<U>>());

    if (sol.code == QPSolutionStatus::Optimal) { warmstart_ = sol; }
    return sol.code;
  }

private:
  Dyn f_;
  SS h_;
  BU bu_;

  ManifoldBounds<U> ulim_;

  ASIFilterParams<U> prm_;

  std::optional<QPSolution<-1, -1, double>> warmstart_;
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__ASIF_HPP_
