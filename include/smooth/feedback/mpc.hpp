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

#include "mpc_func.hpp"

namespace smooth::feedback {

/**
 * @brief Parameters for MPC
 */
template<LieGroup G, Manifold U>
struct MPCParams
{
  /**
   * @brief MPC time horizon (seconds)
   */
  double T{1};

  /**
   * @brief Number of discretization steps
   */
  std::size_t K{10};

  /**
   * @brief MPC state and input weights
   */
  typename OptimalControlProblem<G, U>::Weights weights{};

  /**
   * @brief State bounds
   */
  ManifoldBounds<G> glim{};

  /**
   * @brief Input bounds
   */
  ManifoldBounds<U> ulim{};

  /**
   * @brief Enable warmstarting
   */
  bool warmstart{true};

  /**
   * @brief Relinearize the problem around solution after each solve.
   *
   * If this parameter is false
   *  - Problem is relinearized around the desired trajectory at each iteration
   *
   * If this is set to true:
   *  - Problem is relinearized once around a new desired trajectory
   *  - Problem is relinearized after each solve
   */
  bool relinearize_around_solution{false};

  /**
   * @brief Iterative relinearization.
   *
   * If this is set to a positive number an iterative procedure is used:
   *
   *  1. Solve problem at current linearization
   *  2. If solution does not touch linearization bounds g_bound, stop
   *  3. Else relinearize around solution and go to 1
   *
   * This process is repeated at most MPCParams::iterative_relinearization times.
   *
   * @note Requires LinearizationInfo::g_domain to be set to an appropriate value.
   */
  uint32_t iterative_relinearization{0};

  /**
   * @brief QP solvers parameters.
   */
  QPSolverParams qp{};
};

/**
 * @brief Model-Predictive Control (MPC) on Lie groups.
 *
 * @tparam Time time type, must be a std::chrono::duration-like
 * @tparam G state space LieGroup type
 * @tparam U input space Manifold type
 * @tparam Dyn callable type that represents dynamics
 * @tparam DT differentiation method
 *
 * This MPC class keeps and repeatedly solves an internal OptimalControlProblem that is updated to
 * track a time-dependent trajectory defined through set_xudes().
 *
 * @note If the MPC problem is nonlinear a good (re-)linearization policy is required for good
 * performance. See MPCParams for details. MPC likely performs better on models that
 * are control-affine.
 */
template<typename Time, LieGroup G, Manifold U, typename Dyn, diff::Type DT = diff::Type::DEFAULT>
class MPC
{
public:
  /**
   * @brief Create an MPC instance.
   *
   * @param f callable object that represents dynamics \f$ \mathrm{d}^r x_t = f(f, x, u) \f$ as a
   * function \f$ f : Time \times G \times U \rightarrow \mathbb{R}^{\dim \mathfrak{g}} \f$.
   * @param prm MPC parameters
   *
   * @note \f$ f \f$ is copied/moved into the class. In order to modify the dynamics from the
   * outside the type Dyn can be created to contain references to outside objects that are
   * updated by the user.
   */
  inline MPC(Dyn && f, MPCParams<G, U> && prm = MPCParams<G, U>{})
      : prm_(std::move(prm)), dyn_(std::move(f))
  {
    ocp_.T       = prm_.T;
    ocp_.glim    = prm_.glim;
    ocp_.ulim    = prm_.ulim;
    ocp_.weights = prm_.weights;
    ocp_to_qp_allocate<G, U>(ocp_, prm_.K, qp_);
  }
  /// Same as above but for lvalues
  inline MPC(const Dyn & f, const MPCParams<G, U> & prm = MPCParams<G, U>{})
      : MPC(Dyn(f), MPCParams<G, U>(prm))
  {}
  /// Default constructor
  inline MPC() : MPC(Dyn(), MPCParams<G, U>()) {}
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
   * @param[out] u_traj (optional) return MPC input solution \f$ [\mu_0, \mu_1, \ldots, \mu_{K - 1}] \f$
   * @param[out] x_traj (optional) return MPC state solution \f$ [x_0, x_1, \ldots, x_K] \f$
   *
   * @return {u, code}
   */
  inline std::pair<U, QPSolutionStatus> operator()(const Time & t,
    const G & g,
    std::optional<std::reference_wrapper<std::vector<U>>> u_traj = std::nullopt,
    std::optional<std::reference_wrapper<std::vector<G>>> x_traj = std::nullopt)
  {
    using std::chrono::duration, std::chrono::duration_cast, std::chrono::nanoseconds;

    static constexpr int Nx = Dof<G>;
    static constexpr int Nu = Dof<U>;
    const int NU            = prm_.K * Nu;

    // update problem with functions defined in "MPC time"
    ocp_.x0   = g;
    ocp_.gdes = [this, &t](double t_loc) -> G {
      return x_des_(t + duration_cast<nanoseconds>(duration<double>(t_loc))).first;
    };
    ocp_.udes = [this, &t](double t_loc) -> U {
      return u_des_(t + duration_cast<nanoseconds>(duration<double>(t_loc)));
    };

    // define linearization if not already updated around previous solution
    if (!prm_.relinearize_around_solution || new_desired_) {
      lin_.u = [this, &t](double t_loc) -> U {
        return u_des_(t + duration_cast<nanoseconds>(duration<double>(t_loc)));
      };
      lin_.g = [this, &t](double t_loc) -> std::pair<G, Tangent<G>> {
        return x_des_(t + duration_cast<nanoseconds>(duration<double>(t_loc)));
      };
      new_desired_ = false;
    }

    const double dt = ocp_.T / static_cast<double>(prm_.K);

    // define dynamics in "MPC time"
    const auto dyn = [this, &t]<typename S>(
                       double t_loc, const CastT<S, G> & vx, const CastT<S, U> & vu) {
      return dyn_(t + duration_cast<nanoseconds>(duration<double>(t_loc)), vx, vu);
    };

    ocp_to_qp_fill<G, U, decltype(dyn), DT>(ocp_, prm_.K, dyn, lin_, qp_);
    auto sol = solve_qp(qp_, prm_.qp, warmstart_);

    for (auto i = 0u; i < prm_.iterative_relinearization; ++i) {
      // check if solution touches linearization domain
      bool touches = false;
      for (auto k = 0u; !touches && k < prm_.K; ++k) {
        // clang-format off
        if (((1. - 1e-6) * lin_.g_domain - sol.primal.template segment<Nx>(NU + k * Nx).cwiseAbs()).minCoeff() < 0) { touches = true; }
        // clang-format on
      }
      if (touches) {
        // relinearize around solution and solve again
        relinearize_around_sol(sol);
        ocp_to_qp_fill<G, U, decltype(dyn), DT>(ocp_, prm_.K, dyn, lin_, qp_);
        sol = solve_qp(qp_, prm_.qp, warmstart_);
      } else {
        // solution seems fine
        break;
      }
    }

    // output solution trajectories
    if (u_traj.has_value()) {
      u_traj.value().get().resize(prm_.K);
      for (auto i = 0u; i < prm_.K; ++i) {
        u_traj.value().get()[i] = lin_.u(i * dt) + sol.primal.template segment<Nu>(i * Nu);
      }
    }
    if (x_traj.has_value()) {
      x_traj.value().get().resize(prm_.K + 1);
      x_traj.value().get()[0] = ocp_.x0;
      for (auto i = 1u; i < prm_.K + 1; ++i) {
        x_traj.value().get()[i] =
          lin_.g(i * dt).first + sol.primal.template segment<Nx>(NU + (i - 1) * Nx);
      }
    }

    // update linearization for next iteration
    if (sol.code == QPSolutionStatus::Optimal) {
      if (prm_.relinearize_around_solution) relinearize_around_sol(sol);
    }

    // save solution for warmstart
    if (prm_.warmstart) {
      if (sol.code == QPSolutionStatus::Optimal || sol.code == QPSolutionStatus::MaxTime
          || sol.code == QPSolutionStatus::MaxIterations) {
        warmstart_ = sol;
      }
    }

    return {rplus(lin_.u(0), sol.primal.template head<Nu>()), sol.code};
  }

  /**
   * @brief Set the desired input trajectory (absolute time)
   */
  inline void set_udes(std::function<U(Time)> && u_des)
  {
    u_des_       = std::move(u_des);
    new_desired_ = true;
  }

  /**
   * @brief Set the desired input trajectory (absolute time, rvalue version)
   */
  inline void set_udes(const std::function<U(Time)> & u_des)
  {
    set_udes(std::function<U(Time)>(u_des));
  }

  /**
   * @brief Set the desired input trajectory (relative time).
   *
   * @note This function triggers a relinearization around the desired input and trajectory at the
   * next call to operator()().
   *
   * @param f function double -> U<double> s.t. u_des(t) = f(t - t0)
   * @param t0 absolute zero time for the desired trajectory
   */
  template<typename Fun>
    // \cond
    requires(std::is_same_v<std::invoke_result_t<Fun, Scalar<U>>, U>)
  // \endcond
  inline void set_udes(Fun && f, Time t0 = Time(0))
  {
    set_udes([t0 = t0, f = std::forward<Fun>(f)](Time t) -> U {
      return std::invoke(
        f, std::chrono::duration_cast<std::chrono::duration<double>>(t - t0).count());
    });
  }

  /**
   * @brief Set the desired state trajectory and velocity (absolute time)
   */
  inline void set_xdes(std::function<std::pair<G, Tangent<G>>(Time)> && x_des)
  {
    x_des_       = std::move(x_des);
    new_desired_ = true;
  }

  /**
   * @brief Set the desired state trajectory (absolute time, rvalue version)
   */
  inline void set_xdes(const std::function<std::pair<G, Tangent<G>>(Time)> & x_des)
  {
    set_xdes(std::function<std::pair<G, Tangent<G>>(Time)>(x_des));
  }

  /**
   * @brief Set the desired state trajectry (relative time, automatic differentiation).
   *
   * Instead of providing {x(t), dx(t)} of the desired trajectory, this function differentiates a
   * function x(t) so that the derivative does not need to be specified.
   *
   * @param f function s.t. desired trajectory is x(t) = f(t - t0)
   * @param t0 absolute zero time for the desired trajectory
   *
   * @note This function triggers a relinearization around the desired input and trajectory at the
   * next call to operator()().
   */
  template<typename Fun>
    // \cond
    requires(std::is_same_v<std::invoke_result_t<Fun, Scalar<G>>, G>)
  // \endcond
  inline void set_xdes(Fun && f, Time t0 = Time(0))
  {
    set_xdes([t0 = t0, f = std::forward<Fun>(f)](Time t) -> std::pair<G, Tangent<G>> {
      const auto t_rel = std::chrono::duration_cast<std::chrono::duration<double>>(t - t0).count();
      return diff::dr<DT>(f, wrt(t_rel));
    });
  }

  /**
   * @brief Update MPC weights
   */
  inline void update_weights(const typename OptimalControlProblem<G, U>::Weights & weights)
  {
    ocp_.weights = weights;
  }

  /**
   * @brief Reset initial guess for next iteration to zero.
   */
  inline void reset_warmstart() { warmstart_ = {}; }

  /**
   * @brief Relinearize state around a solution.
   */
  inline void relinearize_around_sol(const QPSolution<-1, -1, double> & sol)
  {
    const double dt = ocp_.T / static_cast<double>(prm_.K);

    // clang-format off
    auto g_spline = smooth::fit_cubic_bezier(
        std::views::iota(0u, prm_.K + 1) | std::views::transform([&](auto k) -> double { return dt * k; }),
        std::views::iota(0u, prm_.K + 1) | std::views::transform([&](auto k) -> G {
          if (k == 0) {
            return ocp_.x0;
          } else {
            return rplus(lin_.g(dt * k).first, sol.primal.template segment<Dof<G>>(prm_.K * Dof<U> + (k - 1) * Dof<G>));
          }
        })
      );
    // clang-format on

    lin_.g = [g_spline = std::move(g_spline)](double t) -> std::pair<G, Tangent<G>> {
      Tangent<G> dg;
      auto g = g_spline(t, dg);
      return std::make_pair(g, dg);
    };
  }

private:
  // parameters
  MPCParams<G, U> prm_{};
  // dynamics description
  Dyn dyn_{};
  // problem description
  OptimalControlProblem<G, U> ocp_;
  // flag to keep track of linearization point, and current linearization
  bool new_desired_{false};
  LinearizationInfo<G, U> lin_{};
  // pre-allocated QP matrices
  QuadraticProgramSparse<double> qp_;
  // store last solution for warmstarting
  std::optional<QPSolution<-1, -1, double>> warmstart_{};
  // desired state (pos + vel) and input trajectories
  std::function<std::pair<G, Tangent<G>>(Time)> x_des_{[](Time) -> std::pair<G, Tangent<G>> {
    return {Default<G>(), Tangent<G>::Zero()};
  }};
  std::function<U(Time)> u_des_{[](Time) -> U { return Default<U>(); }};
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__MPC_HPP_
