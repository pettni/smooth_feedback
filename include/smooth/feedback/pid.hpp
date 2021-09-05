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

#ifndef SMOOTH__FEEDBACK__PID_HPP_
#define SMOOTH__FEEDBACK__PID_HPP_

#include <smooth/lie_group.hpp>
#include <smooth/spline/curve.hpp>

#include <chrono>

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
 * @tparam Time time type, must be a std::chrono::duration-like
 * @tparam G state space LieGroup type
 *
 * This controller is designed for a system
 * \f[
 * \begin{aligned}
 *   \mathrm{d}^r \mathbf{x}_t & = \mathbf{v}, \quad \mathbf{x} \in \mathbb{G}, \mathbf{v} \in
 * \mathbb{R}^{\dim \mathbb{G}} \\ \frac{\mathrm{d}}{\mathrm{d}t} {\mathbf{v}} & = \mathbf{u}, \quad
 * \mathbf{u} \in \mathbb{R}^{\dim \mathbb{G}} \end{aligned} \f] i.e. the input is the body
 * acceleration.
 */
template<typename Time, LieGroup G>
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
   * @brief Set desired trajectory as a smooth::Curve
   *
   * @param c desired trajectory as a smooth::Curve
   * @param t0 curve initial time s.t. the desired position at time t is equal to c(t - t0)
   */
  inline void set_xdes(Time t0, const smooth::Curve<G> & c) { set_xdes(t0, smooth::Curve<G>(c)); }

  /**
   * @brief Set desired trajectory as a smooth::Curve (rvalue version)
   */
  inline void set_xdes(Time t0, smooth::Curve<G> && c)
  {
    set_xdes([t0 = std::move(t0), c = std::move(c)](Time t) -> TrajectoryReturnT {
      Tangent<G> vel, acc;
      double t_curve = std::chrono::duration_cast<std::chrono::duration<double>>(t - t0).count();
      G x            = c(t_curve, vel, acc);
      return TrajectoryReturnT(std::move(x), std::move(vel), std::move(acc));
    });
  }

  /**
   * @brief Set desired trajectory.
   *
   * The trajectory is a function from time to (position, velocity, acceleration). To track a
   * time-dependent trajectory consider using \p smooth::Curve and \ref set_xdes(Time, const
   * smooth::Curve<G> &) to set the desired trajectory.
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
  inline void set_xdes(const std::function<TrajectoryReturnT(Time)> & f)
  {
    auto f_copy = f;
    set_xdes(std::move(f_copy));
  }

  /**
   * @brief Set desired trajectory (rvalue version).
   */
  inline void set_xdes(std::function<TrajectoryReturnT(Time)> && f) { x_des_ = std::move(f); }

  /**
   * @brief Calculate control input
   *
   * @param t current time
   * @param x current state
   * @param v current body velocity
   */
  inline Tangent<G> operator()(const Time & t, const G & x, const Tangent<G> & v)
  {
    const auto [g_des, v_des, a_des] = x_des_(t);
    const Tangent<G> g_err           = g_des - x;

    if (t_last && t > t_last.value()) {
      // update integral state
      double dt =
        std::chrono::duration_cast<std::chrono::duration<double>>(t - t_last.value()).count();
      i_err_ += dt * g_err;
      i_err_ = i_err_.cwiseMax(-prm_.windup_limit).cwiseMin(prm_.windup_limit);
    }
    t_last = t;

    return a_des + kp_.cwiseProduct(g_err) + kd_.cwiseProduct(v_des - v) + ki_.cwiseProduct(i_err_);
  }

private:
  PIDParams prm_;

  // gains
  Tangent<G> kd_ = Tangent<G>::Ones();
  Tangent<G> kp_ = Tangent<G>::Ones();
  Tangent<G> ki_ = Tangent<G>::Zero();

  // integral state
  std::optional<Time> t_last;
  Tangent<G> i_err_ = Tangent<G>::Zero();

  // desired trajectory
  std::function<TrajectoryReturnT(Time)> x_des_ = [](Time) -> TrajectoryReturnT {
    return TrajectoryReturnT(Identity<G>(), Tangent<G>::Zero(), Tangent<G>::Zero());
  };
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__PID_HPP_
