// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Functions for active Set Invariance (ASI) filtering on Lie groups.
 */

#include <Eigen/Core>
#include <boost/numeric/odeint.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/concepts/lie_group.hpp>
#include <smooth/diff.hpp>

#include <algorithm>
#include <limits>
#include <utility>

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
 *                & u \in ulim \\
 *                & h(x(t)) \geq 0, \quad t \in [0, T]    \\
 * \end{cases}
 * \f]
 * for a system \f$ \mathrm{d}^r x_t = f(x(t), u(t)) \f$.
 */
template<LieGroup X, Manifold U>
  requires(Dof<X> > 0 && Dof<U> > 0)
struct ASIFProblem
{
  /// time horizon
  double T{1};
  /// initial state
  X x0{Default<X>()};
  /// desired input
  U u_des{Default<U>()};
  /// weights on desired input
  Eigen::Matrix<double, Dof<U>, 1> W_u{Eigen::Matrix<double, Dof<U>, 1>::Ones()};
  /// input bounds
  ManifoldBounds<U> ulim{};
};

/**
 * @brief Parameters for asif_to_qp
 */
struct ASIFtoQPParams
{
  /// number of constraint instances (equally spaced over the time horizon)
  std::size_t K{10};
  /// barrier function time constant \f$ \alpha \f$ s.t. \f$ \dot h - \alpha h \geq 0 \f$.
  double alpha{1};
  /// maximal integration time step
  double dt{0.1};
  /// relaxation cost
  double relax_cost{100};
};

/**
 * @brief Allocate QP matrices (part 1 of asif_to_qp())
 *
 * @param[out] qp allocated QP with zero matrices
 * @param[in] K number of constraint instances
 * @param[in] nu_ineq number in inequalities in input constraint
 * @param[in] nh number of barrier constraints
 */
template<LieGroup X, Manifold U>
  requires(Dof<X> > 0 && Dof<U> > 0)
void asif_to_qp_allocate(QuadraticProgram<-1, -1, double> & qp, std::size_t K, std::size_t nu_ineq, std::size_t nh)
{
  static constexpr int nx = Dof<X>;
  static constexpr int nu = Dof<U>;

  static_assert(nx > 0, "State space dimension must be static");
  static_assert(nu > 0, "Input space dimension must be static");

  const int M = K * nh + nu_ineq + 1;
  const int N = nu + 1;

  qp.A.setZero(M, N);
  qp.l.setZero(M);
  qp.u.setZero(M);

  qp.P.setZero(N, N);
  qp.q.setZero(N);
}

/**
 * @brief Fill QP matrices (part 2 of asif_to_qp())
 *
 * Note that the (dense) QP matrices must be pre-allocated and filled with zeros.
 */
template<LieGroup X, Manifold U, diff::Type DT = diff::Type::Default>
  requires(Dof<X> > 0 && Dof<U> > 0)
void asif_to_qp_update(
  QuadraticProgram<-1, -1, double> & qp,
  const ASIFProblem<X, U> & pbm,
  const ASIFtoQPParams & prm,
  auto && f,
  auto && h,
  auto && bu)
{
  using boost::numeric::odeint::euler, boost::numeric::odeint::vector_space_algebra;
  using std::placeholders::_1;

  static constexpr int nx = Dof<X>;
  static constexpr int nu = Dof<U>;
  static constexpr int nh = std::invoke_result_t<decltype(h), double, X>::SizeAtCompileTime;

  euler<X, double, Tangent<X>, double, vector_space_algebra> state_stepper{};
  euler<TangentMap<X>, double, TangentMap<X>, double, vector_space_algebra> sensi_stepper{};

  const int nu_ineq = pbm.ulim.A.rows();

  [[maybe_unused]] const int M = prm.K * nh + nu_ineq + 1;
  [[maybe_unused]] const int N = nu + 1;

  assert(qp.A.rows() == M);
  assert(qp.A.cols() == N);
  assert(qp.l.rows() == M);
  assert(qp.u.rows() == M);

  assert(qp.P.rows() == N);
  assert(qp.P.cols() == N);
  assert(qp.q.rows() == N);

  // iteration variables
  const double tau     = pbm.T / static_cast<double>(prm.K);
  const double dt      = std::min<double>(prm.dt, tau);
  double t             = 0;
  X x                  = pbm.x0;
  TangentMap<X> dx_dx0 = TangentMap<X>::Identity();

  // define ODEs for closed-loop dynamics and its sensitivity
  const auto x_ode = [&f, &bu](const X & xx, Tangent<X> & dd, double tt) { dd = f(xx, bu(tt, xx)); };

  const auto dx_dx0_ode = [&f, &bu, &x](const auto & S_v, auto & dS_dt_v, double tt) {
    auto f_cl                   = [&]<typename T>(const CastT<T, X> & vx) { return f(vx, bu(T(tt), vx)); };
    const auto [fcl, dr_fcl_dx] = diff::dr<1, DT>(std::move(f_cl), wrt(x));
    dS_dt_v                     = (-ad<X>(fcl) + dr_fcl_dx) * S_v;
  };

  // value of dynamics at call time
  const auto [f0, d_f0_du] =
    diff::dr<1, DT>([&]<typename T>(const CastT<T, U> & vu) { return f(cast<T>(x), vu); }, wrt(pbm.u_des));

  // loop over constraint number
  for (auto k = 0u; k != prm.K; ++k) {
    // differentiate barrier function w.r.t. x
    const auto [hval, dh_dtx] =
      diff::dr<1, DT>([&h]<typename T>(const T & vt, const CastT<T, X> & vx) { return h(vt, vx); }, wrt(t, x));

    const Eigen::Matrix<double, nh, 1> dh_dt  = dh_dtx.template leftCols<1>();
    const Eigen::Matrix<double, nh, nx> dh_dx = dh_dtx.template rightCols<nx>();

    // insert barrier constraint
    const Eigen::Matrix<double, nh, nx> dh_dx0 = dh_dx * dx_dx0;
    qp.A.template block<nh, nu>(k * nh, 0)     = dh_dx0 * d_f0_du;
    qp.l.template segment<nh>(k * nh)          = -dh_dt - prm.alpha * hval - dh_dx0 * f0;
    qp.u.template segment<nh>(k * nh).setConstant(std::numeric_limits<double>::infinity());

    // integrate system and sensitivity forward until next constraint
    double dt_act = std::min(dt, tau * (k + 1) - t);
    while (t < tau * (k + 1)) {
      state_stepper.do_step(x_ode, x, t, dt_act);
      sensi_stepper.do_step(dx_dx0_ode, dx_dx0, t, dt_act);
      t += dt_act;
    }
  }

  // relaxation of barrier constraints
  qp.A.block(0, nu, prm.K * nh, 1).setConstant(1);

  // input bounds
  qp.A.block(prm.K * nh, 0, nu_ineq, nu) = pbm.ulim.A;
  qp.l.segment(prm.K * nh, nu_ineq)      = pbm.ulim.l - pbm.ulim.A * rminus(pbm.u_des, pbm.ulim.c);
  qp.u.segment(prm.K * nh, nu_ineq)      = pbm.ulim.u - pbm.ulim.A * rminus(pbm.u_des, pbm.ulim.c);

  // upper and lower bounds on delta
  qp.A(prm.K * nh + nu_ineq, nu) = 1;
  qp.l(prm.K * nh + nu_ineq)     = 0;
  qp.u(prm.K * nh + nu_ineq)     = std::numeric_limits<double>::infinity();

  qp.P.template block<nu, nu>(0, 0) = pbm.W_u.asDiagonal();

  qp.P(nu, nu) = prm.relax_cost;
  qp.q(nu)     = 0;
}

/**
 * @brief Convert an ASIFProblem to a QuadraticProgram.
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
 *   \min_{\mu}  & \left\| \mu \right\|^2 \\
 *   \text{s.t.} & \text{constraint above holds for } u = u_{des} + \mu
 *   \end{cases}
 * \f]
 * A solution \f$ \mu^* \f$ to the QuadraticProgram corresponds to an input \f$ u_{des} \oplus \mu^*
 * \f$ applied to the system.
 *
 * @tparam X state LieGroup type \f$\mathbb{X}\f$
 * @tparam U input Manifold type \f$\mathbb{X}\f$
 *
 * @param pbm problem definition
 * @param prm algorithm parameters
 * @param f system model \f$f : \mathbb{R} \times \mathbb{X} \times \mathbb{U} \rightarrow
 * \mathbb{R}^{\dim \mathfrak g}\f$ s.t. \f$ \mathrm{d}^r x_t = f(t, x, u) \f$
 * @param h safe set \f$h : \mathbb{R} \times \mathbb{X} \rightarrow \mathbb{R}^{n_h}\f$ s.t. \f$
 * S(t) = \{ h(t, x) \geq 0 \} \f$ denotes the safe set at time \f$ t \f$
 * @param bu backup controller \f$ub : \mathbb{R} \times \mathbb{X} \rightarrow \mathbb{U} \f$
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
template<LieGroup X, Manifold U, diff::Type DT = diff::Type::Default>
QuadraticProgram<-1, -1, double>
asif_to_qp(const ASIFProblem<X, U> & pbm, const ASIFtoQPParams & prm, auto && f, auto && h, auto && bu)
{
  static constexpr int nh = std::invoke_result_t<decltype(h), double, X>::SizeAtCompileTime;

  static_assert(Dof<X> > 0, "State space dimension must be static");
  static_assert(Dof<U> > 0, "Input space dimension must be static");
  static_assert(nh > 0, "Safe set dimension must be static");

  const int nu_ineq = pbm.ulim.A.rows();
  QuadraticProgram<-1, -1, double> qp;
  asif_to_qp_allocate<X, U>(qp, prm.K, nu_ineq, nh);
  asif_to_qp_update(
    qp, pbm, prm, std::forward<decltype(f)>(f), std::forward<decltype(h)>(h), std::forward<decltype(bu)>(bu));
  return qp;
}

}  // namespace smooth::feedback
