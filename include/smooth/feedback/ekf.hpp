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

#ifndef SMOOTH__FEEDBACK__EKF_HPP_
#define SMOOTH__FEEDBACK__EKF_HPP_

#include <boost/numeric/odeint.hpp>

#include <Eigen/Cholesky>

#include <smooth/compat/odeint.hpp>
#include <smooth/concepts.hpp>
#include <smooth/diff.hpp>

namespace smooth::feedback {

/**
 * @brief Extended Kalman filter on Lie groups.
 *
 * The primary methods are predict() and update().
 * - predict(): propagates filter state through a dynamical model.
 * - update(): Bayesian update of filter state from a measurement.
 *
 * Use this class for information fusion (without dynamics) by solely using update().
 *
 * @tparam G smooth::LieGroup type.
 * @tparam DiffType smooth::diff::Type method for calculating derivatives.
 * @tparam Stepper boost::numeric::odeint templated stepper type (e.g. euler / runge_kutta4).
 * Defaults to euler.
 */
template<LieGroup G,
  smooth::diff::Type DiffType            = smooth::diff::Type::NUMERICAL,
  template<typename...> typename Stepper = boost::numeric::odeint::euler>
class EKF
{
public:
  /// Scalar type for computations.
  using Scalar = typename G::Scalar;
  /// Degrees of freedom.
  static constexpr Eigen::Index Nx = G::SizeAtCompileTime;
  /// Covariance type.
  using CovT = Eigen::Matrix<Scalar, G::SizeAtCompileTime, G::SizeAtCompileTime>;

  /// Default constructor initializes state and covariance to Identity.
  EKF()
  {
    g_hat_.setIdentity();
    P_.setIdentity();
  }
  /// Copy constructor
  EKF(const EKF &) = default;
  /// Move constructor
  EKF(EKF &&) = default;
  /// Copy assignment
  EKF & operator=(const EKF &) = default;
  /// Move assignment
  EKF & operator=(EKF &&) = default;
  /// Destructor
  ~EKF() = default;

  /**
   * @brief Reset the state of the EKF.
   *
   * @param g filter value
   * @param P filter covariance
   */
  void reset(const G & g, const CovT & P)
  {
    g_hat_ = g;
    P_     = P;
  }

  /**
   * @brief Access filter state estimate.
   */
  G state() const { return g_hat_; }

  /**
   * @brief Access filter covariance.
   */
  CovT cov() const { return P_; }

  /**
   * @brief Propagate EKF through dynamics \f$\mathrm{d}^r x_t = f(t, x)\f$ with covariance \f$Q\f$.
   *
   * @param f right-hand side \f$f : \mathbb{R} \times \mathbb{G} \rightarrow \mathbb{R}^{\dim
   * \mathfrak{g}}\f$ of the dynamics
   * @param Q process covariance (size \f$n_x \times n_x\f$)
   * @param tau time \f$\tau\f$ to propagate
   * @param dt maximal ODE solver step size (defaults to \p tau, i.e. one step)
   *
   * @note The time \f$t\f$ argument of \f$f(t, x)\f$ ranges over the interval \f$t \in [0,
   * \tau]\f$.
   *
   * @note The covariance \f$Q\f$ is infinitesimal, i.e. its SI unit is \f$S^2/T\f$
   * where \f$S\f$ is the unit of state and \f$T\f$ is the unit of time.
   *
   * @note Only the upper triangular part of Q is used.
   */
  template<typename F, typename QDer>
  void predict(F && f, const Eigen::MatrixBase<QDer> & Q, Scalar tau, std::optional<Scalar> dt = {})
  {
    Scalar dt_stepper = tau;
    if (dt.has_value()) { dt_stepper = dt.value(); }

    const auto cov_ode = [this, &f, &Q](const CovT & cov, CovT & dcov, Scalar t) {
      const auto [f_val, dr_f] =
        smooth::diff::dr<DiffType>(std::bind(f, t, std::placeholders::_1), smooth::wrt(g_hat_));
      const CovT A = -G::ad(f_val) + dr_f;
      dcov         = (A * cov + cov * A.transpose() + Q).template selfadjointView<Eigen::Upper>();
    };
    const auto state_ode = [&f](const G & g, typename G::Tangent & dg, Scalar t) { dg = f(t, g); };

    Scalar t  = 0;
    bool done = false;

    while (!done) {
      Scalar dt = dt_stepper;
      if (t + dt_stepper >= tau) {
        dt   = tau - t;
        done = true;
      }

      // step covariance first since it depends on g_hat_
      cov_stepper_.do_step(cov_ode, P_, t, dt);
      state_stepper_.do_step(state_ode, g_hat_, t, dt);

      t += dt;
    }
  }

  /**
   * @brief Update EKF with a measurement \f$y = h(x) + w\f$ where \f$w \sim \mathcal N(0, R)\f$.
   *
   * @param h measurement function \f$h : \mathbb{G} \rightarrow \mathbb{R}^{n_y}\f$
   * @param y measurement value (size \f$n_y \times 1\f$)
   * @param R measurement covariance (size \f$n_y \times n_y\f$)
   *
   * @note The function h must be differentiable using the desired method.
   *
   * @note Only the upper triangular part of \f$R\f$ is used
   */
  template<typename F, typename YDer, typename RDev>
  void update(F && h, const Eigen::MatrixBase<YDer> & y, const Eigen::MatrixBase<RDev> & R)
  {
    const auto [hval, H] = smooth::diff::dr<DiffType>(h, smooth::wrt(g_hat_));

    static constexpr Eigen::Index Ny = std::decay_t<decltype(hval)>::SizeAtCompileTime;

    const Eigen::Matrix<Scalar, Ny, Ny> S =
      (H * P_.template selfadjointView<Eigen::Upper>() * H.transpose() + R)
        .template triangularView<Eigen::Upper>();

    // solve for Kalman gain
    // TODO(pettni) use ldlt after requiring new eigen (Eigen < 3.3.8 has problem with cpp20)
    const Eigen::Matrix<Scalar, Nx, Ny> K =
      S.template selfadjointView<Eigen::Upper>().llt().solve(H * P_.transpose()).transpose();

    // update estimate
    g_hat_ += K * (y - hval);

    // update covariance
    P_ = ((CovT::Identity() - K * H) * P_).template selfadjointView<Eigen::Upper>();
  }

private:
  // filter estimate and covariance
  G g_hat_ = G::Identity();
  CovT P_;

  // steppers for numerical ODE solutions
  Stepper<G, Scalar, typename G::Tangent, Scalar, boost::numeric::odeint::vector_space_algebra>
    state_stepper_{};
  Stepper<CovT, Scalar, CovT, Scalar, boost::numeric::odeint::vector_space_algebra> cov_stepper_{};
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__EKF_HPP_
