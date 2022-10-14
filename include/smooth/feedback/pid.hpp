// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

#include <chrono>

#include <smooth/concepts/lie_group.hpp>
#include <smooth/spline/spline.hpp>

#include "time.hpp"

namespace smooth::feedback {

/**
 * Parameters for the PID controller.
 */
struct PIDParams
{
  /// Maximal absolute value for integral states
  double windup_limit = std::numeric_limits<double>::infinity();
};

/**
 * @brief Proportional-Integral-Derivative controller for Lie groups.
 *
 * @tparam T Time type
 * @tparam G LieGroup state space type
 *
 * This controller is designed for a system
 * \f[
 * \begin{aligned}
 *   \mathrm{d}^r \mathbf{x}_t & = \mathbf{v}, \quad \mathbf{x} \in \mathbb{G}, \mathbf{v} \in
 * \mathbb{R}^{\dim \mathbb{G}} \\ \frac{\mathrm{d}}{\mathrm{d}t} {\mathbf{v}} & = \mathbf{u}, \quad
 * \mathbf{u} \in \mathbb{R}^{\dim \mathbb{G}} \end{aligned} \f] i.e. the input is the body
 * acceleration.
 */
template<Time T, LieGroup G>
  requires(Dof<G> > 0)
class PID
{
public:
  /// Desired trajectory consists of position, velocity, and acceleration
  using TrajectoryReturnT = std::tuple<G, Tangent<G>, Tangent<G>>;

  /**
   * @brief Create a PID controller
   *
   * @param prm parameters
   *
   * At construction the proportional and derivative gains are set to 1, and the integral gains are
   * set to 0.
   */
  inline PID(const PIDParams & prm = PIDParams{}) noexcept : prm_(prm) {}
  /// Default copy constructor
  PID(const PID &) = default;
  /// Default move constructor
  PID(PID &&) = default;
  /// Default copy assignment
  PID & operator=(const PID &) = default;
  /// Default move assignment
  PID & operator=(PID &&) = default;
  /// Default destructor
  ~PID() = default;

  /**
   * @brief Calculate control input
   *
   * @param t current time
   * @param x current state
   * @param v current body velocity
   *
   * @return input proportional to desired body acceleration
   */
  inline Tangent<G> operator()(const T & t, const G & x, const Tangent<G> & v)
  {
    const auto [g_des, v_des, a_des] = x_des_(t);
    const Tangent<G> g_err           = g_des - x;

    if (t_last && t > t_last.value()) {
      // update integral state
      i_err_ += time_trait<T>::minus(t, t_last.value()) * g_err;
      i_err_ = i_err_.cwiseMax(-prm_.windup_limit).cwiseMin(prm_.windup_limit);
    }
    t_last = t;

    return a_des + kp_.cwiseProduct(g_err) + kd_.cwiseProduct(v_des - v) + ki_.cwiseProduct(i_err_);
  }

  /**
   * @brief Set all proportional gains to kp.
   */
  inline void set_kp(double kp) { kp_.setConstant(kp); }

  /**
   * @brief Set proportional gains.
   */
  template<typename Derived>
  inline void set_kp(const Eigen::MatrixBase<Derived> & kp)
  {
    kp_ = kp;
  }

  /**
   * @brief Set all derivative gains to kd.
   */
  inline void set_kd(double kd) { kd_.setConstant(kd); }

  /**
   * @brief Set derivative gains.
   */
  template<typename Derived>
  inline void set_kd(const Eigen::MatrixBase<Derived> & kd)
  {
    kd_ = kd;
  }

  /**
   * @brief Set all integral gains to ki.
   */
  inline void set_ki(double ki) { ki_.setConstant(ki); }

  /**
   * @brief Set derivative gains.
   */
  template<typename Derived>
  inline void set_ki(const Eigen::MatrixBase<Derived> & ki)
  {
    ki_ = ki;
  }

  /**
   * @brief Reset integral state to zero.
   */
  inline void reset_integral() { i_err_.setZero(); }

  /**
   * @brief Set desired trajectory as a smooth::Spline
   *
   * @param c desired trajectory as a smooth::Spline
   * @param t0 curve initial time s.t. the desired position at time t is equal to c(t - t0)
   */
  template<int K>
  inline void set_xdes(T t0, const smooth::Spline<K, G> & c)
  {
    set_xdes(t0, smooth::Spline<K, G>(c));
  }

  /**
   * @brief Set desired trajectory as a smooth::Spline (rvalue version)
   */
  template<int K>
  inline void set_xdes(T t0, smooth::Spline<K, G> && c)
  {
    set_xdes([t0 = std::move(t0), c = std::move(c)](T t) -> TrajectoryReturnT {
      Tangent<G> vel, acc;
      G x = c(time_trait<T>::minus(t, t0), vel, acc);
      return TrajectoryReturnT(std::move(x), std::move(vel), std::move(acc));
    });
  }

  /**
   * @brief Set desired trajectory.
   *
   * The trajectory is a function from time to (position, velocity, acceleration). To track a
   * time-dependent trajectory consider using \p smooth::Spline and \ref set_xdes(T, const
   * smooth::Spline<K, G> &) to set the desired trajectory.
   *
   * For a constant reference target the velocity and acceleration should be constantly zero.
   *
   * @note For best performance the trajectory should be dynamically consistent, i.e.
   * \f[
   *   \mathrm{d}^r \mathbf{x}_t = \mathbf{v}, \\
   *   \frac{\mathrm{d}}{\mathrm{d}t} \mathbf{v} = \mathbf{a},
   * \f]
   * where (x, v, a) is the (position, velocity, acceleration)-tuple returned by the trajectory.
   */
  inline void set_xdes(const std::function<TrajectoryReturnT(T)> & f)
  {
    auto f_copy = f;
    set_xdes(std::move(f_copy));
  }

  /**
   * @brief Set desired trajectory (rvalue version).
   */
  inline void set_xdes(std::function<TrajectoryReturnT(T)> && f) { x_des_ = std::move(f); }

private:
  PIDParams prm_;

  // gains
  Tangent<G> kd_ = Tangent<G>::Ones();
  Tangent<G> kp_ = Tangent<G>::Ones();
  Tangent<G> ki_ = Tangent<G>::Zero();

  // integral state
  std::optional<T> t_last;
  Tangent<G> i_err_ = Tangent<G>::Zero();

  // desired trajectory
  std::function<TrajectoryReturnT(T)> x_des_ = [](T) -> TrajectoryReturnT {
    return TrajectoryReturnT(Identity<G>(), Tangent<G>::Zero(), Tangent<G>::Zero());
  };
};

}  // namespace smooth::feedback
