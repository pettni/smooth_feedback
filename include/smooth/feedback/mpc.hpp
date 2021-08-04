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
#include <smooth/spline/bezier.hpp>
#include <smooth/tn.hpp>

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
 *
 * Default constructor linearizes around Identity (LieGroup) or Zero (non-LieGroup Manifold)
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

  /**
   * @brief Domain of validity of state linearization
   *
   *  Defines an upper bound \f$ \bar a \f$ s.t. the linearization is valid for \f$ g \f$ s.t.
   * \f[
   *   \left\| g \ominus_r g_{lin} \right \| \leq \bar a.
   * \f]
   */
  Eigen::Matrix<double, G::SizeAtCompileTime, 1> g_domain =
    Eigen::Matrix<double, G::SizeAtCompileTime, 1>::Constant(
      std::numeric_limits<double>::infinity());

  /**
   * @brief Domain of validity of input linearization
   *
   *  Defines an upper bound \f$ \bar b \f$ s.t. the linearization is valid for \f$ u \f$ s.t.
   * \f[
   *   \left\| u \ominus_r u_{lin} \right \| \leq \bar b.
   * \f]
   */
  Eigen::Matrix<double, U::SizeAtCompileTime, 1> u_domain =
    Eigen::Matrix<double, U::SizeAtCompileTime, 1>::Constant(
      std::numeric_limits<double>::infinity());
};

/**
 * @brief Calculate row-wise sparsity pattern of QP matrix A in MPC problem.
 */
template<std::size_t K, std::size_t nx, std::size_t nu>
constexpr std::array<int, K *(2 * nx + nu)> mpc_nnz_pattern_A()
{
  std::array<int, K *(2 * nx + nu)> ret{};
  for (std::size_t i = 0u; i != nx; ++i) { ret[i] = 1 + nu; }
  for (std::size_t i = nx; i != K * nx; ++i) { ret[i] = 1 + nx + nu; }
  for (std::size_t i = K * nx; i != K * (2 * nx + nu); ++i) { ret[i] = 1; }
  return ret;
}

/**
 * @brief Calculate column-wise sparsity pattern of QP matrix P in MPC problem.
 */
template<std::size_t K, std::size_t nx, std::size_t nu>
constexpr std::array<int, K *(nu + nx)> mpc_nnz_pattern_P()
{
  std::array<int, K *(nx + nu)> ret{};
  for (std::size_t i = 0u; i != K * nu; ++i) { ret[i] = nu; }
  for (std::size_t i = K * nu; i != K * (nu + nx); ++i) { ret[i] = nx; }
  return ret;
}

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
 * The QP inequalities constrain the variables so that the QP solution stays within the domain of
 * validity of the linearization.
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
QuadraticProgramSparse<double> ocp_to_qp(const OptimalControlProblem<G, U> & pbm,
  Dyn && f,
  const LinearizationInfo<G, U> & lin = LinearizationInfo<G, U>{})
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
  ret.q.resize(nvar);
  ret.A.resize(ncon, nvar);
  ret.l.resize(ncon);
  ret.u.resize(ncon);

  // SET SPARSITY PATTERNS

  static constexpr auto Ap = mpc_nnz_pattern_A<K, nx, nu>();
  static constexpr auto Pp = mpc_nnz_pattern_P<K, nx, nu>();
  ret.A.reserve(Eigen::Map<const Eigen::Matrix<int, ncon, 1>>(Ap.data()));
  ret.P.reserve(Eigen::Map<const Eigen::Matrix<int, nvar, 1>>(Pp.data()));

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
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;

    // dltv system x^+ = Ak x + Bk u + Ek by truncated taylor expansion of the matrix exponential
    const AT Ak = AT::Identity() + At * dt + At2 * dt2 / 2. + At2 * At * dt3 / 6.;
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
      // x(k+1) - A x(k) - B u(k) = E

      // identity matrix on x(k+1)
      for (auto i = 0u; i != nx; ++i) { ret.A.insert(nx * k + i, nU + nx * k + i) = 1; }

      // A matrix on x(k)
      for (auto i = 0u; i != nx; ++i) {
        for (auto j = 0u; j != nx; ++j) {
          ret.A.insert(nx * k + i, nU + nx * (k - 1) + j) = -Ak(i, j);
        }
      }

      // B matrix on u(k)
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

  // linearization bounds
  for (auto k = 0u; k < K; ++k) {
    ret.l.template segment<nu>(n_eq + k * nu) =
      ret.l.template segment<nu>(n_eq + k * nu).cwiseMax(-lin.u_domain);
    ret.u.template segment<nu>(n_eq + k * nu) =
      ret.u.template segment<nu>(n_eq + k * nu).cwiseMin(lin.u_domain);
  }

  // STATE CONSTRAINTS

  for (auto i = 0u; i != n_x_iq; ++i) { ret.A.insert(n_eq + n_u_iq + i, nU + i) = 1; }

  if (pbm.gmin) {
    for (auto k = 1u; k < K + 1; ++k) {
      ret.l.template segment<nx>(n_eq + n_u_iq + (k - 1) * nx) = pbm.gmin.value() - lin.g(k * dt);
    }
  } else {
    ret.l.template segment<nX>(n_eq + n_u_iq).setConstant(-std::numeric_limits<double>::infinity());
  }
  if (pbm.gmax) {
    for (auto k = 1u; k < K + 1; ++k) {
      ret.u.template segment<nx>(n_eq + n_u_iq + (k - 1) * nx) = pbm.gmax.value() - lin.g(k * dt);
    }
  } else {
    ret.u.template segment<nX>(n_eq + n_u_iq).setConstant(std::numeric_limits<double>::infinity());
  }

  // linearization bounds
  for (auto k = 1u; k < K + 1; ++k) {
    ret.l.template segment<nx>(n_eq + n_u_iq + (k - 1) * nx) =
      ret.l.template segment<nx>(n_eq + n_u_iq + (k - 1) * nx).cwiseMax(-lin.g_domain);
    ret.u.template segment<nx>(n_eq + n_u_iq + (k - 1) * nx) =
      ret.u.template segment<nx>(n_eq + n_u_iq + (k - 1) * nx).cwiseMin(lin.g_domain);
  }

  // INPUT COSTS

  for (auto k = 0u; k < K; ++k) {
    for (auto i = 0u; i != nu; ++i) {
      for (auto j = 0u; j != nu; ++j) { ret.P.insert(k * nu + i, k * nu + j) = pbm.R(i, j) * dt; }
    }
    ret.q.template segment<nu>(k * nu) = pbm.R * (lin.u(k * dt) - pbm.udes(k * dt)) * dt;
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

struct MPCParams
{
  /// MPC horizon (seconds)
  double T{1};

  /**
   * @brief Enable warmstarting.
   *
   * @warning MPCParams::warmstart should not be enabled together with a low value for
   * MPCParams::relinearization_interval, since a relinearized problem is approximately solved by
   * the zero solution.
   */
  bool warmstart{true};

  /**
   * @brief Relinearize problem at even intervals.
   *
   * Set to zero to disable relinearization.
   */
  uint32_t relinearization_interval{0};

  /**
   * @brief Automatically re-linearize problem when set_xudes() is called.
   */
  bool relinearize_on_new_desired{false};

  /**
   * @brief Iterative relinearization.
   *
   * If this is set to a positive number an iterative procedure is used:
   *
   *  1. Solve problem at current linearization
   *  2. If solution does not touch linearization bounds g_bound / u_bound, stop
   *  3. Else relinearize around solution and go to 1
   *
   * This process is repeated at most MPCParams::iterative_relinearization times.
   *
   * @note Requires LinearizationInfo::g_domain and LinearizationInfo::u_domain to be
   * set to appropriate values.
   */
  uint32_t iterative_relinearization{0};
};

/**
 * @brief Model-Predictive Control (MPC) on Lie groups.
 *
 * @tparam K number of MPC discretization steps (seee ocp_to_qp).
 * @tparam T time type, must be a std::chrono::duration-like
 * @tparam G state space LieGroup type
 * @tparam U input space Manifold type
 * @tparam Dyn callable type that represents dynamics
 *
 * This MPC class keeps and repeatedly solves an internal OptimalControlProblem that is updated to
 * track a time-dependent trajectory defined through set_xudes().
 *
 * @note If the MPC problem is nonlinear a good (re-)linearization policy is required for good
 * performance. See MPCParams for various options. If none of the options are enabled it is up to
 * the user to update the linearizaiton in an appropriate way, or make sure that the problem is
 * linear. The linearization point can be manually updated via the linearization() member functions.
 */
template<std::size_t K, typename T, LieGroup G, Manifold U, typename Dyn>
class MPC
{
public:
  /**
   * @brief Create an MPC instance.
   *
   * @param f callable object that represents dynamics \f$ \mathrm{d}^r f_t = f(x, u) \f$.
   * @param t time horizon
   * @param qp_prm optional parameters for the QP solver (see solve_qp())
   */
  MPC(Dyn && f, const MPCParams & prm, const SolverParams & qp_prm = SolverParams{})
      : prm_(prm), dyn_(std::forward<Dyn>(f)), qp_prm_(qp_prm)
  {}

  /// See constructor above
  MPC(const Dyn & f, const MPCParams & prm, const SolverParams & qp_prm = SolverParams{})
      : MPC(Dyn(f), prm, qp_prm)
  {}
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
   * @brief Solve MPC problem and return input.
   *
   * @param[out] u resulting input
   * @param[in] t current time
   * @param[in] g current state
   * @param[out] x_traj optionally return MPC trajectory
   *
   * @return solver exit code
   *
   * If MPCParams::relinearization_interval is set to a value \f$ I > 0 \f$, an internal counter is
   * maintained that relinearizes the problem at every \f$ I \f$ calls to this function.
   *
   * If MPCParams::iterative_relinearization is set to a value \f$ I > 0 \f$, this function checks
   * up to \f$ I \f$ times whether the solution touches the linearization domain, and relinearizes
   * and re-solves if that is the case.
   */
  ExitCode operator()(U & u,
    const T & t,
    const G & g,
    std::optional<std::reference_wrapper<std::vector<G>>> x_traj = std::nullopt)
  {
    using std::chrono::nanoseconds;

    static constexpr int nx = G::SizeAtCompileTime;
    static constexpr int nu = U::SizeAtCompileTime;
    static constexpr int nU = K * nu;

    ocp_.x0   = g;
    ocp_.T    = prm_.T;
    ocp_.gdes = [this, &t](double t_loc) -> G {
      return x_des_(t + duration_cast<nanoseconds>(std::chrono::duration<double>(t_loc)));
    };
    ocp_.udes = [this, &t](double t_loc) -> U {
      return u_des_(t + duration_cast<nanoseconds>(std::chrono::duration<double>(t_loc)));
    };

    if (relinearize_) {
      // linearize about desired trajectory
      lin_.u  = ocp_.udes;
      lin_.g  = ocp_.gdes;
      lin_.dg = [this](double t) -> typename G::Tangent {
        // need numerical derivative here since gdes is not templated...
        Eigen::Matrix<double, 1, 1> tvec(t);
        auto [gval, dg] = smooth::diff::dr<smooth::diff::Type::NUMERICAL>(
          [this](const auto & v) { return lin_.g(v.x()); }, smooth::wrt(tvec));
        return dg;
      };
      lin_cntr_ = 0;
    }

    const double dt = ocp_.T / static_cast<double>(K);

    const auto qp = smooth::feedback::ocp_to_qp<K>(ocp_, dyn_, lin_);
    auto sol      = smooth::feedback::solve_qp(qp, qp_prm_, warmstart_);

    for (auto i = 0u; i < prm_.iterative_relinearization; ++i) {
      // check if solution touches linearization domain
      bool touches = false;
      for (int k = 0; !touches && k != K; ++k) {
        // clang-format off
        if (((1. - 1e-6) * lin_.u_domain - sol.primal.template segment<nu>(     k * nu).cwiseAbs()).minCoeff() < 0) { touches = true; }
        if (((1. - 1e-6) * lin_.g_domain - sol.primal.template segment<nx>(nU + k * nx).cwiseAbs()).minCoeff() < 0) { touches = true; }
        // clang-format on
      }
      if (touches) {
        // relinearize and solve again
        relinearize(sol);
        lin_cntr_ = 0;

        const auto qp = smooth::feedback::ocp_to_qp<K>(ocp_, dyn_, lin_);
        sol           = smooth::feedback::solve_qp(qp, qp_prm_);
      } else {
        // solution seems fine
        break;
      }
    }

    // output result
    u = lin_.u(0) + sol.primal.template head<nu>();

    // output solution trajectory
    if (x_traj.has_value()) {
      x_traj.value().get().resize(K + 1);
      x_traj.value().get()[0] = ocp_.x0;
      for (auto i = 1u; i < K + 1; ++i) {
        x_traj.value().get()[i] =
          lin_.g(i * dt) + sol.primal.template segment<nx>(nU + (i - 1) * nx);
      }
    }

    if (sol.code == ExitCode::Optimal) {
      if (prm_.relinearization_interval > 0 && lin_cntr_ >= prm_.relinearization_interval) {
        relinearize(sol);  // relinearize problem about current solution
        lin_cntr_ = 0;
      } else {
        ++lin_cntr_;
      }
      if (prm_.warmstart) warmstart_ = sol;  // store successful solution for later warmstart
    }

    return sol.code;
  }

  /**
   * @brief Set input bounds.
   */
  void set_ulim(const U & umin, const U & umax)
  {
    ocp_.umin = umin;
    ocp_.umax = umax;
  }

  /**
   * @brief Set the desired state and input trajectories (lvalue version).
   *
   * @note If MPCParams::relinearize_on_new_desired is set this triggers a relinearization around
   * the desired trajectories at the next call to operator()().
   *
   * @param x_des desired state trajectory \f$ g_{des} (t) \f$ as function \f$ T \rightarrow G \f$
   * @param u_des desired input trajectory \f$ u_{des} (t) \f$ as function \f$ T \rightarrow U \f$
   */
  void set_xudes(const std::function<G(T)> & x_des, const std::function<U(T)> & u_des)
  {
    set_xudes(std::function<G(T)>(x_des), std::function<U(T)>(u_des));
  };

  /**
   * @brief Set the desired state and input trajectories (rvalue version).
   *
   * @note If MPCParams::relinearize_on_new_desired is set this triggers a relinearization around
   * the desired trajectories at the next call to operator()().
   *
   * @param x_des desired state trajectory \f$ g_{des} (t) \f$ as function \f$ T \rightarrow G \f$
   * @param u_des desired input trajectory \f$ u_{des} (t) \f$ as function \f$ T \rightarrow U \f$
   */
  void set_xudes(std::function<G(T)> && x_des, std::function<U(T)> && u_des)
  {
    x_des_ = std::move(x_des);
    u_des_ = std::move(u_des);

    if (prm_.relinearize_on_new_desired) { relinearize_ = true; }
  };

  /**
   * @brief Access the linearization.
   */
  LinearizationInfo<G, U> & linearization() { return lin_; }

  /**
   * @brief Const access the linearization.
   */
  const LinearizationInfo<G, U> & linearization() const { return lin_; }

  /**
   * @brief Reset initial guess for next iteration to zero.
   */
  void reset_warmstart() { warmstart_ = {}; }

  /**
   * @brief Set the running state cost in the internal OptimalControlProblem.
   *
   * @param Q positive definite matrix of size \f$ n_x \times n_x \f$
   */
  template<typename D1>
  void set_running_state_cost(const Eigen::MatrixBase<D1> & Q)
  {
    ocp_.Q = Q;
  }

  /**
   * @brief Set the final state cost in the internal OptimalControlProblem.
   *
   * @param QT positive definite matrix of size \f$ n_x \times n_x \f$
   */
  template<typename D1>
  void set_final_state_cost(const Eigen::MatrixBase<D1> & QT)
  {
    ocp_.QT = QT;
  }

  /**
   * @brief Set the running input cost in the internal OptimalControlProblem.
   *
   * @param R positive definite matrix of size \f$ n_u \times n_u \f$
   */
  template<typename D1>
  void set_input_cost(const Eigen::MatrixBase<D1> & R)
  {
    ocp_.R = R;
  }

  /**
   * @brief Relinearize problem around a solution.
   */
  void relinearize(const Solution<-1, -1, double> & sol)
  {
    static constexpr int nx = G::SizeAtCompileTime;
    static constexpr int nu = U::SizeAtCompileTime;
    static constexpr int nU = K * nu;
    const double dt         = ocp_.T / static_cast<double>(K);

    // STATE LINEARIZATION
    std::vector<double> tt(K + 1);
    std::vector<G> gg(K + 1);
    tt[0] = 0;
    gg[0] = ocp_.x0;

    for (auto k = 1u; k != K + 1; ++k) {
      tt[k] = dt * k;
      gg[k] = lin_.g(k * dt) + sol.primal.template segment<nx>(nU + (k - 1) * nx);
    }

    auto g_spline = smooth::fit_cubic_bezier(tt, gg);
    lin_.g        = [g_spline = g_spline](double t) -> G { return g_spline.eval(t); };
    lin_.dg       = [g_spline = std::move(g_spline)](double t) -> typename G::Tangent {
      typename G::Tangent ret;
      g_spline.eval(t, ret);
      return ret;
    };

    // INPUT LINEARIZATION
    if constexpr (LieGroup<U>) {
      std::vector<double> tt(K);
      std::vector<U> uu(K);

      for (auto k = 0u; k != K; ++k) {
        tt[k] = k * dt;
        uu[k] = lin_.u(k * dt) + sol.primal.template segment<nu>(k * nu);
      }

      auto u_spline = smooth::fit_linear_bezier(tt, uu);
      lin_.u        = [u_spline = std::move(u_spline)](double t) -> U { return u_spline.eval(t); };
    } else {
      std::vector<double> tt(K);
      std::vector<smooth::Tn<U::SizeAtCompileTime, double>> uu(K);

      for (auto k = 0u; k != K; ++k) {
        tt[k] = k * dt;
        uu[k] = lin_.u(k * dt) + sol.primal.template segment<nu>(k * nu);
      }

      auto u_spline = smooth::fit_linear_bezier(tt, uu);
      lin_.u = [u_spline = std::move(u_spline)](double t) -> U { return u_spline.eval(t).rn(); };
    }
  }

private:
  MPCParams prm_{};

  Dyn dyn_{};

  SolverParams qp_prm_{};
  OptimalControlProblem<G, U> ocp_{};
  std::optional<Solution<-1, -1, double>> warmstart_{};

  bool relinearize_{false};
  uint32_t lin_cntr_{0};
  LinearizationInfo<G, U> lin_{};

  std::function<G(T)> x_des_ = [](T) -> G { return G::Identity(); };
  std::function<U(T)> u_des_ = [](T) -> U {
    if constexpr (LieGroup<U>)
      return U::Identity();
    else
      return U::Zero();
  };
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__MPC_HPP_
