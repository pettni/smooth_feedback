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

#include <Eigen/Cholesky>
#include <boost/numeric/odeint.hpp>
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
 * @tparam G \p smooth::LieGroup type.
 * @tparam DiffType \p smooth::diff::Type method for calculating derivatives.
 * @tparam Stpr \p boost::numeric::odeint templated stepper type (\p euler / \p runge_kutta4 /
 * ...). Defaults to \p euler.
 */
template<LieGroup G,
  diff::Type DiffType                 = diff::Type::DEFAULT,
  template<typename...> typename Stpr = boost::numeric::odeint::euler>
class EKF
{
public:
  //! Scalar type for computations.
  using Scalar = typename G::Scalar;
  //! Degrees of freedom.
  using CovT = Eigen::Matrix<Scalar, G::Dof, G::Dof>;

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
  G estimate() const { return g_hat_; }

  /**
   * @brief Access filter covariance.
   */
  CovT covariance() const { return P_; }

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
   * @note The covariance \f$ Q \f$ is infinitesimal, i.e. its SI unit is \f$S^2/T\f$
   * where \f$S\f$ is the unit of state and \f$T\f$ is the unit of time.
   *
   * @note Only the upper triangular part of Q is used.
   */
  template<typename F, typename QDer>
  void predict(F && f, const Eigen::MatrixBase<QDer> & Q, Scalar tau, std::optional<Scalar> dt = {})
  {
    const Scalar dt_v = dt.has_value() ? dt.value() : 2 * tau;

    const auto cov_ode = [this, &f, &Q](const CovT & cov, CovT & dcov, Scalar t) {
      const auto [fv, dr] = diff::dr<DiffType>(std::bind(f, t, std::placeholders::_1), wrt(g_hat_));
      const CovT A        = -G::ad(fv) + dr;
      dcov = (A * cov + cov * A.transpose() + Q).template selfadjointView<Eigen::Upper>();
    };
    const auto state_ode = [&f](const G & g, typename G::Tangent & dg, Scalar t) { dg = f(t, g); };

    Scalar t = 0;
    while (t + dt_v < tau) {
      // step covariance first since it depends on g_hat_
      cst_.do_step(cov_ode, P_, t, dt_v);
      sst_.do_step(state_ode, g_hat_, t, dt_v);
      t += dt_v;
    }

    // last step up to time t
    cst_.do_step(cov_ode, P_, t, tau - t);
    sst_.do_step(state_ode, g_hat_, t, tau - t);
  }

  /**
   * @brief Update EKF with a measurement \f$y = h(x) + w\f$ where \f$w \sim \mathcal N(0, R)\f$.
   *
   * @param h measurement function \f$ h : \mathbb{G} \rightarrow \mathbb{Y} \f$
   * @param y measurement value \f$ y \in \mathbb{Y} \f$
   * @param R measurement covariance (size \f$ \dim \mathbb{Y} \times \dim \mathbb{Y} \f$)
   *
   * @note The function h must be differentiable using the desired method.
   *
   * @note Only the upper triangular part of \f$R\f$ is used
   */
  template<typename F, typename RDev, Manifold Y = std::invoke_result_t<F, G>>
  void update(F && h, const Y & y, const Eigen::MatrixBase<RDev> & R)
  {
    const auto [hval, H] = diff::dr<DiffType>(h, wrt(g_hat_));

    static constexpr Eigen::Index Ny = std::decay_t<decltype(hval)>::SizeAtCompileTime;

    const Eigen::Matrix<Scalar, Ny, Ny> S =
      (H * P_.template selfadjointView<Eigen::Upper>() * H.transpose() + R)
        .template triangularView<Eigen::Upper>();

    // solve for Kalman gain
    const Eigen::Matrix<Scalar, G::Dof, Ny> K =
      S.template selfadjointView<Eigen::Upper>().ldlt().solve(H * P_).transpose();

    // update estimate and covariance
    g_hat_ += K * (y - hval);
    P_ = ((CovT::Identity() - K * H) * P_).template selfadjointView<Eigen::Upper>();
  }

private:
  // filter estimate and covariance
  G g_hat_ = G::Identity();
  CovT P_  = CovT::Identity();

  // steppers for numerical ODE solutions
  Stpr<G, Scalar, typename G::Tangent, Scalar, boost::numeric::odeint::vector_space_algebra> sst_{};
  Stpr<CovT, Scalar, CovT, Scalar, boost::numeric::odeint::vector_space_algebra> cst_{};
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__EKF_HPP_
