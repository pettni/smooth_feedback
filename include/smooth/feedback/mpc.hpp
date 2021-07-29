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

#include <chrono>
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
 * @brief Struct to define a linearization point.
 */
template<LieGroup G, Manifold U>
struct LinearizationInfo
{
  /// state trajectory \f$g_{lin}(t)\f$ to linearize around as a function \f$ \mathbb{R} \rightarrow
  /// G \f$
  std::function<G(double)> g = [](double) -> G { return G::Identity(); };
  /// state trajectory derivative \f$\mathrm{d}^r (g_{lin})_t\f$ to linearize around as a function
  /// \f$ \mathbb{R} \rightarrow \mathbb{R}^{\dim \mathfrak g} \f$
  std::function<typename G::Tangent(double)> dg = [](double) ->
    typename G::Tangent { return G::Tangent::Zero(); };
  /// input trajectory \f$u_{lin}(t)\f$ to linearize around as a function \f$ \mathbb{R} \rightarrow
  /// G \f$
  std::function<U(double)> u = [](double) -> U {
    if constexpr (LieGroup<U>) {
      return U::Identity();
    } else {
      return U::Zero();
    }
  };
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
 * @note \p f must be differentiable with the default smooth::diff method. If using an automatic
 * differentiation method this means that it must be templated on the scalar type.
 *
 * @tparam K number of discretization points. More points create a larger QP, but the distance \f$ T
 * / K \f$ between points should be smaller than the smallest system time constant for adequate
 * performance.
 * @tparam G problem state group type \f$ \mathbb{G} \f$
 * @tparam U problem input group type \f$ \mathbb{U} \f$
 *
 * @param pbm optimal control problem
 * @param f dynamics \f$ f : \mathbb{G} \times \mathbb{U} \rightarrow \mathbb{R}^{\dim \mathfrak
 * g}\f$ s.t. \f$ \mathrm{d}^r g_t = f(g, u) \f$
 * @param lin linearization point, default is to linearize around Identity (LieGroup) / Zero
 * (non-LieGroup manifold)
 *
 * @return QuadraticProgramSparse modeling the input optimal control problem.
 */
template<std::size_t K, LieGroup G, Manifold U, typename Dyn>
QuadraticProgramSparse<double> ocp_to_qp(
  const OptimalControlProblem<G, U> & pbm, Dyn && f, const LinearizationInfo<G, U> & lin)
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

  QuadraticProgramSparse ret;
  ret.P.resize(nvar, nvar);
  ret.q.setZero(nvar);

  ret.A.resize(ncon, nvar);
  ret.l.setZero(ncon);
  ret.u.setZero(ncon);

  // SET SPARSITY PATTERNS

  Eigen::Matrix<int, ncon, 1> A_sparsity;     // row-wise
  Eigen::Matrix<int, nU + nX, 1> P_sparsity;  // column-wise (but symmetric so doesn't matter)

  // dynamics
  A_sparsity.template head<nx>().setConstant(nu + 1);                     // B, I
  A_sparsity.template segment<nx *(K - 1)>(nx).setConstant(nx + nu + 1);  // B, A, I

  // input and state bounds
  A_sparsity.template segment<n_u_iq>(n_eq).setConstant(1);
  A_sparsity.template segment<n_x_iq>(n_eq + n_u_iq).setConstant(1);

  // cost matrix is block-diagonal with blocks of size nu x nu and nx x nx
  P_sparsity.template head<nU>().setConstant(nu);
  P_sparsity.template segment<nX>(nU).setConstant(nx);

  ret.A.reserve(A_sparsity);
  ret.P.reserve(P_sparsity);

  // DYNAMICS CONSTRAINTS

  for (auto k = 0u; k != K; ++k) {
    const double t = k * dt;

    // LINEARIZATION

    const auto xl  = lin.g(t);
    const auto dxl = lin.dg(t);
    const auto ul  = lin.u(t);

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

      // identity matrix on x(1)
      for (auto i = 0u; i != nx; ++i) { ret.A.insert(nx * k + i, nU + nx * k + i) = 1; }

      // B matrix on u(0)
      for (auto i = 0u; i != nx; ++i) {
        for (auto j = 0u; j != nu; ++j) { ret.A.insert(nx * k + i, nu * k + j) = -Bk(i, j); }
      }

      ret.u.template segment<nx>(nx * k) = Ak * (pbm.x0 - lin.g(t)) + Ek;
      ret.l.template segment<nx>(nx * k) = ret.u.template segment<nx>(nx * k);

    } else {
      // x(k) - A x(k-1) - B u(k-1) = E

      // identity matrix on x(k)
      for (auto i = 0u; i != nx; ++i) { ret.A.insert(nx * k + i, nU + nx * k + i) = 1; }

      // A matrix on x(k-1)
      for (auto i = 0u; i != nx; ++i) {
        for (auto j = 0u; j != nx; ++j) {
          ret.A.insert(nx * k + i, nU + nx * (k - 1) + j) = -Ak(i, j);
        }
      }

      // B matrix on u(k-1)
      for (auto i = 0u; i != nx; ++i) {
        for (auto j = 0u; j != nu; ++j) { ret.A.insert(nx * k + i, nu * k + j) = -Bk(i, j); }
      }

      ret.u.template segment<nx>(nx * k) = Ek;
      ret.l.template segment<nx>(nx * k) = Ek;
    }
  }

  // INPUT CONSTRAINTS

  for (auto i = 0u; i != n_u_iq; ++i) { ret.A.insert(n_eq + i, i) = 1; }

  if (pbm.umin) {
    for (auto k = 0u; k < K; ++k) {
      ret.l.template segment<nu>(n_eq + k * nu) = pbm.umin.value() - lin.u(k * dt);
    }
  } else {
    ret.l.template segment<nU>(n_eq).setConstant(-std::numeric_limits<double>::infinity());
  }
  if (pbm.umax) {
    for (auto k = 0u; k < K; ++k) {
      ret.u.template segment<nu>(n_eq + k * nu) = pbm.umax.value() - lin.u(k * dt);
    }
  } else {
    ret.u.template segment<nU>(n_eq).setConstant(std::numeric_limits<double>::infinity());
  }

  // STATE CONSTRAINTS

  for (auto i = 0u; i != n_x_iq; ++i) { ret.A.insert(n_eq + n_u_iq + i, nU + i) = 1; }

  if (pbm.gmin) {
    for (auto k = 1u; k < K; ++k) {
      ret.l.template segment<nx>(n_eq + n_u_iq + (k - 1) * nx) = pbm.gmin.value() - lin.g(k * dt);
    }
  } else {
    ret.l.template segment<nX>(n_eq + n_u_iq).setConstant(-std::numeric_limits<double>::infinity());
  }
  if (pbm.gmax) {
    for (auto k = 1u; k < K; ++k) {
      ret.u.template segment<nx>(n_eq + n_u_iq + (k - 1) * nx) = pbm.gmax.value() - lin.g(k * dt);
    }
  } else {
    ret.u.template segment<nX>(n_eq + n_u_iq).setConstant(std::numeric_limits<double>::infinity());
  }

  // INPUT COSTS

  for (auto k = 0u; k < K; ++k) {
    for (auto i = 0u; i != nu; ++i) {
      for (auto j = 0u; j != nu; ++j) { ret.P.insert(k * nu + i, k * nu + j) = pbm.R(i, j) * dt; }
    }
    ret.q.template segment<nu>(k * nu) = pbm.R * (lin.u(k * dt) - pbm.udes(k * dt));
  }

  // STATE COSTS

  // intermediate states x(1) ... x(K-1)
  for (auto k = 1u; k < K; ++k) {
    for (auto i = 0u; i != nx; ++i) {
      for (auto j = 0u; j != nx; ++j) {
        ret.P.insert(nU + (k - 1) * nx + i, nU + (k - 1) * nx + j) = pbm.Q(i, j) * dt;
      }
    }
    ret.q.template segment<nx>(nU + (k - 1) * nx) = pbm.Q * (lin.g(k * dt) - pbm.gdes(k * dt)) * dt;
  }

  // last state x(K) ~ x(T)
  for (auto i = 0u; i != nx; ++i) {
    for (auto j = 0u; j != nx; ++j) {
      ret.P.insert(nU + (K - 1) * nx + i, nU + (K - 1) * nx + j) = pbm.QT(i, j);
    }
  }
  ret.q.template segment<nx>(nU + (K - 1) * nx) = pbm.QT * (lin.g(pbm.T) - pbm.gdes(pbm.T));

  ret.A.makeCompressed();
  ret.P.makeCompressed();

  return ret;
}

/**
 * @brief Model-Predictive Control (MPC) on Lie groups.
 *
 * @tparam K number of MPC discretization steps (seee ocp_to_qp).
 * @tparam T time type, must be a std::chrono::duration-like
 * @tparam G state space LieGroup type
 * @tparam U input space Manifold type
 * @tparam Dyn callable type that represents dynamics
 */
template<std::size_t K, typename T, LieGroup G, Manifold U, typename Dyn>
class MPC
{
public:
  /**
   * @brief Create an MPC instance
   *
   * @param dyn callable object that represents dynamics \f$ \mathrm{d}^r f_t = f(x, u) \f$.
   * @param t time horizon
   * @param qp_prm optional parameters for the QP solver (see \p solve_qp)
   */
  MPC(Dyn && dyn, T t, const SolverParams & qp_prm = SolverParams{})
      : dyn_(std::forward<Dyn>(dyn)), qp_prm_(qp_prm)
  {
    // TODO set this stuff
    ocp_.Q.setIdentity();
    ocp_.QT.setIdentity();
    ocp_.R.setIdentity();
    ocp_.T = std::chrono::duration_cast<std::chrono::duration<double>>(t).count();
  }
  /// See constructor above
  MPC(const Dyn & dyn, T t, const SolverParams & qp_prm = SolverParams{}) : MPC(Dyn(dyn), t, qp_prm) {}
  /// Default constructor
  MPC() = default;
  /// Default copy constructor
  MPC(const MPC &) = default;
  /// Default move constructor
  MPC(MPC &&) = default;
  /// Default copy assignment
  MPC & operator=(const MPC &) = default;
  /// Default move assignment
  MPC & operator=(MPC &&) = default;
  /// Default destructor
  ~MPC() = default;

  /**
   * @brief Get input
   *
   * @param t current time
   * @param g current state
   *
   * @return input calculated by MPC
   */
  U operator()(const T & t, const G & g)
  {
    ocp_.x0   = g;
    ocp_.gdes = [this, &t](double t_loc) {
      return x_des_(t + duration_cast<T>(std::chrono::duration<double>(t_loc)));
    };
    ocp_.udes = [this, &t](double t_loc) {
      return u_des_(t + duration_cast<T>(std::chrono::duration<double>(t_loc)));
    };

    const auto qp  = smooth::feedback::ocp_to_qp<K>(ocp_, dyn_, lin_);
    const auto sol = smooth::feedback::solve_qp(qp, qp_prm_);

    return lin_.u(0) + sol.primal.template head<U::SizeAtCompileTime>();

    // TODO: update linearization point here
  }

  /**
   * @brief Set input bounds
   */
  void set_ulim(const U & umin, const U & umax)
  {
    ocp_.umin = umin;
    ocp_.umax = umax;
  }

  /**
   * @brief Set the desired state and input trajectories (lvalue version)
   *
   * @param x_des desired state trajectory \f$ g_{des} (t) \f$ as function \f$ T \rightarrow G \f$
   * @param u_des desired input trajectory \f$ u_{des} (t) \f$ as function \f$ T \rightarrow U \f$
   */
  void set_xudes(const std::function<G(T)> & x_des, const std::function<U(T)> & u_des)
  {
    x_des_ = x_des;
    u_des_ = u_des;
  };

  /**
   * @brief Set the desired state and input trajectories (rvalue version)
   *
   * @param x_des desired state trajectory \f$ g_{des} (t) \f$ as function \f$ T \rightarrow G \f$
   * @param u_des desired input trajectory \f$ u_{des} (t) \f$ as function \f$ T \rightarrow U \f$
   */
  void set_xudes(std::function<G(T)> && x_des, std::function<U(T)> && u_des)
  {
    x_des_ = std::move(x_des);
    u_des_ = std::move(u_des);
  };

private:
  Dyn dyn_{};

  SolverParams qp_prm_{};
  OptimalControlProblem<G, U> ocp_{};

  LinearizationInfo<G, U> lin_{};

  std::function<G(T)> x_des_{};
  std::function<U(T)> u_des_{};
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__MPC_HPP_
