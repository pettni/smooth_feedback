#include <Eigen/Core>

#include <smooth/concepts.hpp>
#include <smooth/diff.hpp>

/**
 * @file
 * @brief MPC on Lie groups.
 */

namespace smooth::feedback {

/**
 * @brief Structure holding weights for optimal control problem.
 */
template<Eigen::Index nx, Eigen::Index nu>
struct MpcWeights
{
  /// Running state cost
  Eigen::Matrix<double, nx, nx> Q = Eigen::Matrix<double, nx, nx>::Identity();
  /// Final state cost
  Eigen::Matrix<double, nx, nx> QT = Eigen::Matrix<double, nx, nx>::Identity();
  /// Running input cost
  Eigen::Matrix<double, nu, nu> R = Eigen::Matrix<double, nu, nu>::Identity();
};

/**
 * @brief Standard form of a QP problem
 *
 * \f[
 * \begin{cases}
 *  \min_{x} & \frac{1}{2} x^T P x + q^T x \\
 *  \text{s.t.} & l \leq A x \leq u   \\
 * \end{cases}
 * \f]
 */
template<Eigen::Index nvar, Eigen::Index ncon>
struct QpProblem
{
  /// Quadratic cost
  Eigen::Matrix<double, nvar, nvar> P = Eigen::Matrix<double, nvar, nvar>::Zero();
  /// Linear cost
  Eigen::Matrix<double, nvar, 1> q = Eigen::Matrix<double, nvar, 1>::Zero();

  /// Inequality matrix
  Eigen::Matrix<double, ncon, nvar> A = Eigen::Matrix<double, ncon, nvar>::Zero();
  /// Inequality lower bound
  Eigen::Matrix<double, ncon, 1> l = Eigen::Matrix<double, ncon, 1>::Zero();
  /// Inequality upper bound
  Eigen::Matrix<double, ncon, 1> u = Eigen::Matrix<double, ncon, 1>::Zero();
};

/**
 * @brief Convert optimal control problem into a quadratic program.
 *
 * The optimal control problem is
 *
 * \f[
 *   \begin{cases}
 *    \min_{u(\cdot)} & \int_{0}^T (x(t) \ominus x_{des}(t))^T Q (x(t) \ominus x_{des}(t))^T
 *     + (u(t) \ominus u_{des}(t))^T R (u(t) \ominus u_{des}(t))^T + (u(T) \ominus u_{des}(T))^T Q_T
 * (u(T) \ominus u_{des}(T)) \\
 *     \text{s.t.}                 & \mathrm{d}^r x_t = f(t, x, u)  \\
 *                                 & x(0) = x_0
 *   \end{cases}
 * \f]
 *
 * This is encoded into a QpProblem via linearization around \f$(x_{lin}(t), u_{lin}(t))\f$ followed
 * by time discretization. The variables of the QP are
 * \f[
 *   \begin{bmatrix} u_0 & u_1 & \ldots & u_{K - 1} & x_1 & x_2 & \ldots & x_K \end{bmatrix},
 * \f]
 * where the discrete time index \f$k\f$ corresponds to time \f$t_k = k \frac{T}{K - 1} \f$.
 *
 * @note No state or input constraints are added.
 *
 * @tparam K number of discretization steps
 * @tparam G problem state type
 * @tparam U problem input type
 *
 * @param f dynamical model of form \f$ \mathrm{d}^r x_t = f(t, x, u) \f$
 * @param x0 initial state
 * @param T problem horizon
 * @param xlin state trajectory to linearize around \f$x_{lin}(t)\f$
 * @param ulin input trajectory to liearize around \f$u_{lin}(t)\f$
 * @param xdes desired state trajectory \f$x_{des}(t)\f$
 * @param udes desired input trajectory \f$u_{des}(t)\f$
 * @param weights problem weights
 *
 * @return QpProblem modeling the input optimal control problem.
 */
template<std::size_t K,
  LieGroup G,
  typename U,
  typename Dyn,
  typename XLin,
  typename ULin,
  typename XDes,
  typename UDes>
auto mpc(Dyn && f,
  const G & x0,
  double T,
  XLin && xlin,
  ULin && ulin,
  XDes && xdes,
  UDes && udes,
  const MpcWeights<G::SizeAtCompileTime, U::SizeAtCompileTime> & weights)
{
  // problem info
  static constexpr int nx = G::SizeAtCompileTime;
  static constexpr int nu = U::SizeAtCompileTime;

  static constexpr int nX = (K - 1) * nx;
  static constexpr int nU = (K - 1) * nu;

  static constexpr int nvar = (K - 1) * (nx + nu);
  static constexpr int ncon = (K - 1) * nx;

  using AT = Eigen::Matrix<double, nx, nx>;
  using BT = Eigen::Matrix<double, nx, nu>;
  using ET = Eigen::Matrix<double, nx, 1>;

  const double dt = T / static_cast<double>(K - 1);

  QpProblem<nvar, ncon> ret;

  // variable layout
  // u_0 u_1 u_2 ... u_{K-2} x_1 x_2 ... x_{K-1}

  // x_0 = x(0) - xlin(0)
  ret.A.template block<nx, nx>(0, nU).setIdentity();
  ret.u.template segment<nx>(0) = x0 - xlin(0);
  ret.l.template segment<nx>(0) = ret.u.template segment<nx>(0);

  for (auto k = 0u; k < K - 2; ++k) {
    const double t = k * dt;

    // LINEARIZATION

    auto [xl, dxl] = smooth::diff::dr(
      [&](auto & v) { return xlin(v(0)); }, smooth::wrt(Eigen::Matrix<double, 1, 1>(t)));
    auto ul = ulin(t);

    const auto [f1, df_x] =
      smooth::diff::dr([&](auto & v) { return f(t, v, ul); }, smooth::wrt(xl));
    const auto [f2, df_u] =
      smooth::diff::dr([&](auto & v) { return f(t, xl, v); }, smooth::wrt(ul));

    // cltv system \dot x = At x(t) + Bt u(t) + Et
    const AT At = (-0.5 * G::ad(f1) - 0.5 * G::ad(dxl) + df_x);
    const BT Bt = df_u;
    const ET Et = f1 - dxl;

    // TIME DISCRETIZATION

    const AT At2     = At * At;
    const AT At3     = At2 * At;
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;

    // Cltv system x^+ = At x + Bt u + Et
    const AT Ak = Eigen::Matrix<double, nx, nx>::Identity() + At * dt + At2 * dt2 / 2.;
    const BT Bk = Bt * dt + At * Bt * dt2 / 2. + At2 * Bt * dt3 / 6.;
    const ET Ek = Et * dt + At * Et * dt2 / 2. + At2 * Et * dt3 / 6.;

    // DYNAMICS CONStRANTS

    ret.A.template block<nx, nx>(nx * (k + 1), nU + nx * k) = -Ak;
    ret.A.template block<nx, nx>(nx * (k + 1), nU + nx * (k + 1)).setIdentity();
    ret.A.template block<nx, nu>(nx * (k + 1), nu * k) = -Bk;
    ret.u.template segment<nx>(nx * (k + 1))           = -Ek;
    ret.l.template segment<nx>(nx * (k + 1))           = -Ek;
  }

  // INPUT COSTS
  for (auto k = 0u; k < K - 1; ++k) {
    ret.P.template block<nu, nu>(k * nu, k * nu) = weights.R * dt;
    ret.q.template segment<nu>(k * nu)           = weights.R * (ulin(k * dt) - udes(k * dt));
  }

  // STATE COSTS
  for (auto k = 1u; k < K - 1; ++k) {
    ret.P.template block<nx, nx>(nU + (k - 1) * nx, nU + (k - 1) * nx) = weights.Q * dt;
    ret.q.template segment<nx>((k - 1) * nx) = weights.Q * (xlin(k * dt) - xdes(k * dt));
  }
  ret.P.template block<nx, nx>(nU + (K - 2) * nx, nU + (K - 2) * nx) = weights.QT;
  ret.q.template segment<nx>((K - 2) * nx) = weights.QT * (xlin(T) - xdes(T));

  return ret;
}

}  // namespace smooth::feedback
