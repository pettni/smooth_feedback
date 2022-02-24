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

#include <smooth/lie_group.hpp>

#include <memory>

#include "ocp_to_qp.hpp"
#include "time.hpp"

namespace smooth::feedback {

template<LieGroup X, Manifold U>
struct TrackingWeights
{
  static constexpr auto Nx = Dof<X>;
  static constexpr auto Nu = Dof<U>;

  /// Running state cost
  Eigen::Matrix<double, Nx, Nx> Q = Eigen::Matrix<double, Nx, Nx>::Identity();
  /// Final state cost
  Eigen::Matrix<double, Nx, Nx> QT = Eigen::Matrix<double, Nx, Nx>::Identity();
  /// Running input cost
  Eigen::Matrix<double, Nu, Nu> R = Eigen::Matrix<double, Nu, Nu>::Identity();
};

namespace detail {

template<Time T, LieGroup X>
struct XDes
{
  T t0;
  std::function<X(T)> xdes           = [](T) -> X { return Default<X>(); };
  std::function<Tangent<X>(T)> dxdes = [](T) -> Tangent<X> { return Tangent<X>::Zero(); };

  X operator()(const double t_rel) const
  {
    const T t_abs = time_trait<T>::plus(t0, t_rel);
    return xdes(t_abs);
  };

  Tangent<X> jacobian(const double t_rel) const
  {
    const T t_abs = time_trait<T>::plus(t0, t_rel);
    return dxdes(t_abs);
  }
};

template<Time T, Manifold U>
struct UDes
{
  T t0;
  std::function<U(T)> udes = [](T) -> U { return Default<U>(); };

  U operator()(const double t_rel) const
  {
    const T t_abs = time_trait<T>::plus(t0, t_rel);
    return udes(t_abs);
  };
};

template<LieGroup X>
struct TrackingCost
{
  static constexpr auto Nx = Dof<X>;

  X xf_des;
  Eigen::Matrix<double, Nx, Nx> QT = Eigen::Matrix<double, Nx, Nx>::Identity();

  template<typename S>
  S operator()(
    const S &, const CastT<S, X> &, const CastT<S, X> & xf, const Eigen::Vector<S, 1> & q) const
  {
    const Tangent<CastT<S, X>> xf_err = rminus(xf, xf_des.template cast<S>());
    return q.sum() + S(0.5) * (QT * xf_err).dot(xf_err);
  }
};

template<Time T, LieGroup X, Manifold U>
struct TrackingIntegral
{
  static constexpr auto Nx = Dof<X>;
  static constexpr auto Nu = Dof<U>;

  T t0{0};
  std::shared_ptr<XDes<T, X>> xdes;
  std::shared_ptr<UDes<T, U>> udes;

  Eigen::Matrix<double, Nx, Nx> Q = Eigen::Matrix<double, Nx, Nx>::Identity();
  Eigen::Matrix<double, Nu, Nu> R = Eigen::Matrix<double, Nu, Nu>::Identity();

  template<typename S>
  Eigen::Vector<S, 1>
  operator()(const S & t_loc, const CastT<S, X> & x, const CastT<S, U> & u) const
  {
    const T t_abs = time_trait<T>::plus(t0, static_cast<double>(t_loc));

    const Tangent<CastT<S, X>> x_err = rminus(x, (*xdes)(t_abs));
    const Tangent<CastT<S, U>> u_err = rminus(u, (*udes)(t_abs));

    return Eigen::Vector<S, 1>{
      S(0.5) * (Q * x_err).dot(x_err) + S(0.5) * (R * u_err).dot(u_err),
    };
  }
};

template<LieGroup X>
struct TrackingCE
{
  X x0val;

  template<typename S>
  Tangent<CastT<S, X>> operator()(
    const S &, const CastT<S, X> & x0, const CastT<S, X> &, const Eigen::Vector<S, 1> &) const
  {
    return rminus(x0, x0val.template cast<S>());
  }
};

}  // namespace detail

/**
 * @brief Parameters for MPC
 */
struct MPCParams
{
  /**
   * @brief MPC time horizon (seconds)
   */
  double tf{1};

  /**
   * @brief Enable warmstarting
   */
  bool warmstart{true};

  /**
   * @brief QP solvers parameters.
   */
  QPSolverParams qp{};
};

/**
 * @brief Model-Predictive Control (MPC) on Lie groups.
 *
 * @tparam T time type, must be a std::chrono::duration-like
 * @tparam X state space LieGroup type
 * @tparam U input space Manifold type
 * @tparam F callable type that represents dynamics
 * @tparam CR callable type that represents running constraints
 * @tparam CE callable type that represents end constraints
 * @tparam DT differentiation method
 *
 * This MPC class keeps and repeatedly solves an internal OptimalControlProblem that is updated to
 * track a time-dependent trajectory defined through set_xudes().
 *
 * @note If the MPC problem is nonlinear a good (re-)linearization policy is required for good
 * performance. See MPCParams for details. MPC likely performs better on models that
 * are control-affine.
 */
template<
  std::size_t K,
  Time T,
  LieGroup X,
  Manifold U,
  typename F,
  typename CR,
  diff::Type DT = diff::Type::Default>
class MPC
{
  static constexpr auto Ncr = std::invoke_result_t<CR, double, X, U>::SizeAtCompileTime;

public:
  /**
   * @brief Create an MPC instance.
   *
   * @param f callable object that represents dynamics \f$ \mathrm{d}^r x_t = f(f, x, u) \f$ as a
   * function \f$ f : T \times X \times U \rightarrow \mathbb{R}^{\dim \mathfrak{g}} \f$.
   * @param prm MPC parameters
   *
   * @note \f$ f \f$ is copied/moved into the class. In order to modify the dynamics from the
   * outside the type F can be created to contain references to outside objects that are
   * updated by the user.
   */
  inline MPC(
    F && f,
    CR && cr,
    Eigen::Vector<double, Ncr> && crl,
    Eigen::Vector<double, Ncr> && cru,
    MPCParams && prm = MPCParams{})
      : xdes_(std::make_shared<detail::XDes<T, X>>()),
        udes_(std::make_shared<detail::UDes<T, U>>()),
        ocp_{
          .theta = detail::TrackingCost<X>{},
          .f     = std::forward<F>(f),
          .g =
            detail::TrackingIntegral<T, X, U>{
              .t0   = 0,
              .xdes = xdes_,
              .udes = udes_,
            },
          .cr  = std::forward<CR>(cr),
          .crl = std::move(crl),
          .cru = std::move(cru),
          .ce  = detail::TrackingCE<X>{},
          .cel = Eigen::Vector<double, Dof<X>>::Zero(),
          .ceu = Eigen::Vector<double, Dof<X>>::Zero(),
        },
        prm_(std::move(prm))
  {
    detail::ocp_to_qp_allocate<DT>(qp_, work_, ocp_, mesh_);
  }
  /// Same as above but for lvalues
  inline MPC(
    const F & f,
    const CR & cr,
    const Eigen::Vector<double, Ncr> & crl,
    const Eigen::Vector<double, Ncr> & cru,
    const MPCParams & prm = MPCParams{})
      : MPC(
        F(f),
        CR(cr),
        Eigen::Vector<double, Ncr>(crl),
        Eigen::Vector<double, Ncr>(cru),
        MPCParams(prm))
  {}
  /// Default constructor
  inline MPC() = default;
  /// Default copy constructor
  inline MPC(const MPC &) = default;
  /// Default move constructor
  inline MPC(MPC &&) = default;
  /// Default copy assignment
  inline MPC & operator=(const MPC &) = default;
  /// Default move assignment
  inline MPC & operator=(MPC &&) = default;
  /// Default destructor
  inline ~MPC() = default;

  /**
   * @brief Solve MPC problem and return input.
   *
   * @param[in] t current time
   * @param[in] g current state
   * @param[out] u_traj (optional) return MPC input solution \f$ [\mu_0, \mu_1, \ldots, \mu_{K - 1}]
   * \f$
   * @param[out] x_traj (optional) return MPC state solution \f$ [x_0, x_1, \ldots, x_K] \f$
   *
   * @return {u, code}
   */
  inline std::pair<U, QPSolutionStatus> operator()(
    const T & t,
    const X & g,
    std::optional<std::reference_wrapper<std::vector<U>>> u_traj = std::nullopt,
    std::optional<std::reference_wrapper<std::vector<X>>> x_traj = std::nullopt)
  {
    static constexpr auto Nx = Dof<X>;
    static constexpr auto Nu = Dof<U>;
    const auto N             = mesh_.N_colloc();

    const auto xvar_L = Nx * (N + 1);

    const auto xvar_B = 0u;
    const auto uvar_B = xvar_L;

    // update problem
    xdes_->t0     = t;
    udes_->t0     = t;
    ocp_.ce.x0val = g;
    ocp_.g.t0     = t;

    // transcribe to QP
    ocp_to_qp_update<DT, smooth::diff::Type::Analytic>(
      qp_, work_, ocp_, mesh_, prm_.tf, *xdes_, *udes_);

    // solve QP
    auto sol = solve_qp(qp_, prm_.qp, warmstart_);

    // output solution trajectories
    if (u_traj.has_value()) {
      u_traj.value().get().resize(N);
      for (const auto & [i, trel] : zip(std::views::iota(0u, N), mesh_.all_nodes())) {
        u_traj.value().get()[i] = (*udes_)(trel) + sol.primal.template segment<Nu>(uvar_B + i * Nu);
      }
    }
    if (x_traj.has_value()) {
      x_traj.value().get().resize(N + 1);
      for (const auto & [i, trel] : zip(std::views::iota(0u, N + 1), mesh_.all_nodes())) {
        x_traj.value().get()[i] = (*xdes_)(trel) + sol.primal.template segment<Nx>(xvar_B + i * Nx);
      }
    }

    // save solution for warmstart
    if (prm_.warmstart) {
      if (
        // clang-format off
        sol.code == QPSolutionStatus::Optimal
        || sol.code == QPSolutionStatus::MaxTime
        || sol.code == QPSolutionStatus::MaxIterations
        // clang-format n
      ) {
        warmstart_ = sol;
      }
    }

    return {rplus((*udes_)(0), sol.primal.template segment<Nu>(Nx * (N + 1))), sol.code};
  }

  /**
   * @brief Set the desired input trajectory (absolute time)
   */
  inline void set_udes(std::function<U(T)> && u_des)
  {
    udes_->udes = std::move(u_des);
  }

  /**
   * @brief Set the desired input trajectory (absolute time, rvalue version)
   */
  inline void set_udes(const std::function<U(T)> & u_des) { set_udes(std::function<U(T)>(u_des)); }

  /**
   * @brief Set the desired input trajectory (relative time).
   *
   * @param f function double -> U<double> s.t. u_des(t) = f(t - t0)
   * @param t0 absolute zero time for the desired trajectory
   */
  template<typename Fun>
    requires(std::is_same_v<std::invoke_result_t<Fun, Scalar<U>>, U>)
  inline void set_udes_rel(Fun && f, T t0 = T(0))
  {
    set_udes([t0 = t0, f = std::forward<Fun>(f)](T t) -> U {
      const double t_rel = time_trait<T>::minus(t, t0);
      return std::invoke(f, t_rel);
    });
  }

  /**
   * @brief Set the desired state trajectory and velocity (absolute time)
   */
  inline void set_xdes(std::function<X(T)> && x_des, std::function<Tangent<X>(T)> && dx_des)
  {
    ocp_.theta.xf_des = std::invoke(x_des, prm_.tf);

    xdes_->xdes = std::move(x_des);
    xdes_->dxdes = std::move(dx_des);
  }

  /**
   * @brief Set the desired state trajectory (absolute time, rvalue version)
   */
  inline void set_xdes(const std::function<X(T)> & x_des, const std::function<Tangent<X>(T)> & dx_des)
  {
    set_xdes(std::function<X(T)>(x_des), std::function<Tangent<X>(T)>(dx_des));
  }

  /**
   * @brief Set the desired state trajectry (relative time, automatic differentiation).
   *
   * Instead of providing {x(t), dx(t)} of the desired trajectory, this function differentiates a
   * function x(t) so that the derivative does not need to be specified.
   *
   * @param f function s.t. desired trajectory is x(t) = f(t - t0)
   * @param t0 absolute zero time for the desired trajectory
   */
  template<typename Fun>
    requires(std::is_same_v<std::invoke_result_t<Fun, Scalar<X>>, X>)
  inline void set_xdes_rel(Fun && f, T t0 = T(0))
  {
    std::function<X(T)> x_des = [t0 = t0, f = f](T t) -> X {
      const double t_rel = time_trait<T>::minus(t, t0);
      return std::invoke(f, t_rel);
    };

    std::function<Tangent<X>(T)> dx_des = [t0 = t0, f = f](T t) -> Tangent<X> {
      const double t_rel = time_trait<T>::minus(t, t0);
      return std::get<1>(diff::dr<1, DT>(f, wrt(t_rel)));
    };

    set_xdes(std::move(x_des), std::move(dx_des));
  }

  /**
   * @brief Update MPC weights
   */
  inline void set_weights(const TrackingWeights<X, U> & weights)
  {
    ocp_.g.R      = weights.R;
    ocp_.g.Q      = weights.Q;
    ocp_.theta.QT = weights.QT;
  }

  /**
   * @brief Reset initial guess for next iteration to zero.
   */
  inline void reset_warmstart() { warmstart_ = {}; }

private:
  // linearization
  std::shared_ptr<detail::XDes<T, X>> xdes_;
  std::shared_ptr<detail::UDes<T, U>> udes_;

  // problem description
  using ocp_t = OCP< X, U, detail::TrackingCost<X>, F, detail::TrackingIntegral<T, X, U>, CR, detail::TrackingCE<X>>;
  ocp_t ocp_;

  // parameters
  MPCParams prm_{};

  // collocation mesh
  Mesh<K, K> mesh_{};

  // pre-allocated QP matrices
  detail::OcpToQpWorkmemory work_;
  QuadraticProgramSparse<double> qp_;

  // store last solution for warmstarting
  std::optional<QPSolution<-1, -1, double>> warmstart_{};
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__MPC_HPP_
