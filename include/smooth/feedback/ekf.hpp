// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Cholesky>
#include <boost/numeric/odeint.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/concepts/lie_group.hpp>
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
template<
  LieGroup G,
  diff::Type DiffType                 = diff::Type::Default,
  template<typename...> typename Stpr = boost::numeric::odeint::euler>
  requires(Dof<G> > 0)
class EKF
{
public:
  //! Scalar type for computations.
  //! Degrees of freedom.
  using CovT = Eigen::Matrix<Scalar<G>, Dof<G>, Dof<G>>;

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
   * @brief Propagate EKF through dynamics \f$ \mathrm{d}^r x_t = f(t, x) \f$ with covariance
   * \f$Q\f$ over a time interval \f$ [0, \tau] \f$.
   *
   * @param f right-hand side \f$ f : \mathbb{R} \times \mathbb{G} \rightarrow \mathbb{R}^{\dim
   * \mathfrak{g}} \f$ of the dynamics. The time type must be the scalar type of G.
   * @param Q process covariance (size \f$ \dim \mathfrak{g} \times \dim \mathfrak{g} \f$)
   * @param tau amount of time to propagate
   * @param dt maximal ODE solver step size (defaults to \p tau, i.e. one step)
   *
   * @note The time \f$ t \f$ argument of \f$ f(t, x) \f$ ranges over the interval \f$t  \in [0,
   * \tau] \f$.
   *
   * @note The covariance \f$ Q \f$ is infinitesimal, i.e. its SI unit is \f$ S^2/T \f$
   * where \f$S\f$ is the unit of state and \f$T\f$ is the unit of time.
   *
   * @note Only the upper triangular part of Q is used.
   */
  template<typename F, typename QDer>
  void predict(F && f, const Eigen::MatrixBase<QDer> & Q, Scalar<G> tau, std::optional<Scalar<G>> dt = {})
  {
    const auto state_ode = [&f](const G & g, Tangent<G> & dg, Scalar<G> t) { dg = f(t, g); };

    const auto cov_ode = [this, &f, &Q](const CovT & cov, CovT & dcov, Scalar<G> t) {
      const auto f_x      = [&f, &t]<typename _T>(const CastT<_T, G> & x) -> Tangent<CastT<_T, G>> { return f(t, x); };
      const auto [fv, dr] = diff::dr<1, DiffType>(f_x, wrt(g_hat_));
      const CovT A        = -ad<G>(fv) + dr;
      dcov                = (A * cov + cov * A.transpose() + Q).template selfadjointView<Eigen::Upper>();
    };

    Scalar<G> t          = 0;
    const Scalar<G> dt_v = dt.value_or(2 * tau);
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
    const auto [hval, H] = diff::dr<1, DiffType>(h, wrt(g_hat_));

    using Result = std::decay_t<decltype(hval)>;

    static_assert(Manifold<Result>, "h(x) is not a Manifold");

    static constexpr Eigen::Index Ny = Dof<Result>;

    static_assert(Ny > 0, "h(x) must be statically sized");

    const Eigen::Matrix<Scalar<G>, Ny, Ny> S =
      (H * P_.template selfadjointView<Eigen::Upper>() * H.transpose() + R).template triangularView<Eigen::Upper>();

    // solve for Kalman gain
    const Eigen::Matrix<Scalar<G>, Dof<G>, Ny> K =
      S.template selfadjointView<Eigen::Upper>().ldlt().solve(H * P_).transpose();

    // update estimate and covariance
    g_hat_ += K * (y - hval);
    P_ = ((CovT::Identity() - K * H) * P_).template selfadjointView<Eigen::Upper>();
  }

private:
  // filter estimate and covariance
  G g_hat_ = Default<G>();
  CovT P_  = CovT::Identity();

  // steppers for numerical ODE solutions
  Stpr<G, Scalar<G>, Tangent<G>, Scalar<G>, boost::numeric::odeint::vector_space_algebra> sst_{};
  Stpr<CovT, Scalar<G>, CovT, Scalar<G>, boost::numeric::odeint::vector_space_algebra> cst_{};
};

}  // namespace smooth::feedback
