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
#include <smooth/diff.hpp>
#include <smooth/lie_group.hpp>
#include <smooth/spline/bezier.hpp>

#include "qp.hpp"

namespace smooth::feedback {

/**
 * @brief Bounds for OptimalControlProblem.
 *
 * Bounds are of the form
 *
 * \f[
 *   l \leq  A ( m \ominus e_M ) \leq u.
 * \f]
 */
template<Manifold M>
struct OptimalControlBounds
{
  /// Dimensionality
  static constexpr int N = Dof<M>;

  /// Transformation matrix
  Eigen::Matrix<double, -1, N> A = Eigen::Matrix<double, -1, N>::Identity(N, N);
  /// Lower bound
  Eigen::Matrix<double, -1, 1> l;
  /// Upper bound
  Eigen::Matrix<double, -1, 1> u;
};

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
 *     \text{s.t.}    & g(0) = x_0,                  \\
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
  static constexpr Eigen::Index Nx = Dof<G>;
  /// Input tangent dimension
  static constexpr Eigen::Index Nu = Dof<U>;

  /// Initial state
  G x0 = Identity<G>();

  /// Time horizon
  double T{1};

  /// Desired state trajectory
  std::function<G(double)> gdes = [](double) { return Identity<G>(); };
  /// Desired input trajectory
  std::function<U(double)> udes = [](double) { return Identity<U>(); };

  /// Input bounds
  std::optional<OptimalControlBounds<U>> ulim;

  /// State bounds
  std::optional<OptimalControlBounds<G>> glim;

  /// Running state cost
  Eigen::Matrix<double, Nx, Nx> Q = Eigen::Matrix<double, Nx, Nx>::Identity();
  /// Final state cost
  Eigen::Matrix<double, Nx, Nx> QT = Eigen::Matrix<double, Nx, Nx>::Identity();
  /// Running input cost
  Eigen::Matrix<double, Nu, Nu> R = Eigen::Matrix<double, Nu, Nu>::Identity();
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
  std::function<G(double)> g = [](double) -> G { return Identity<G>(); };
  /// state trajectory derivative \f$\mathrm{d}^r (g_{lin})_t\f$ to linearize around as a function
  /// \f$ \mathbb{R} \rightarrow \mathbb{R}^{\dim \mathfrak g} \f$
  std::function<Tangent<G>(double)> dg = [](double) -> Tangent<G> { return Tangent<G>::Zero(); };
  /// input trajectory \f$u_{lin}(t)\f$ to linearize around as a function \f$ \mathbb{R} \rightarrow
  /// G \f$
  std::function<U(double)> u = [](double) -> U { return Identity<U>(); };

  /**
   * @brief Domain of validity of state linearization
   *
   *  Defines an upper bound \f$ \bar a \f$ s.t. the linearization is valid for \f$ g \f$ s.t.
   * \f[
   *   \left\| g \ominus_r g_{lin} \right \| \leq \bar a.
   * \f]
   */
  Eigen::Matrix<double, Dof<G>, 1> g_domain =
    Eigen::Matrix<double, Dof<G>, 1>::Constant(std::numeric_limits<double>::infinity());

  /**
   * @brief Domain of validity of input linearization
   *
   *  Defines an upper bound \f$ \bar b \f$ s.t. the linearization is valid for \f$ u \f$ s.t.
   * \f[
   *   \left\| u \ominus_r u_{lin} \right \| \leq \bar b.
   * \f]
   */
  Eigen::Matrix<double, Dof<U>, 1> u_domain =
    Eigen::Matrix<double, Dof<U>, 1>::Constant(std::numeric_limits<double>::infinity());
};

/**
 * @brief Calculate column-wise sparsity pattern of QP matrix P in MPC problem.
 */
template<std::size_t K, std::size_t Nx, std::size_t Nu>
constexpr std::array<int, K *(Nu + Nx)> mpc_nnz_pattern_P()
{
  std::array<int, K *(Nx + Nu)> ret{};
  for (std::size_t i = 0u; i != K * Nu; ++i) { ret[i] = Nu; }
  for (std::size_t i = K * Nu; i != K * (Nu + Nx); ++i) { ret[i] = Nx; }
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
 * @tparam K number of discretization points. More points create a larger QP, but the distance \f$ T
 * / K \f$ between points should be smaller than the smallest system time constant for adequate
 * performance.
 * @tparam G problem state group type \f$ \mathbb{G} \f$
 * @tparam U problem input group type \f$ \mathbb{U} \f$
 * @tparam Dyn dynamics functor type
 * @tparam DiffType differentiation method to utilize
 *
 * @param pbm optimal control problem
 * @param f dynamics \f$ f : \mathbb{R} \times \mathbb{G} \times \mathbb{U} \rightarrow
 * \mathbb{R}^{\dim \mathfrak g}\f$ s.t. \f$ \mathrm{d}^r g_t = f(t, g, u) \f$
 * @param lin linearization point, default is to linearize around Identity (LieGroup) / Zero
 * (non-LieGroup manifold)
 *
 * @return QuadraticProgramSparse modeling the input optimal control problem.
 *
 * @note Given a solution \f$(x^*, \mu^*)\f$ to the QuadraticProgram, the corresponding
 * solution to the OptimalControlProblem is \f$ u^*(t) = u_{lin}(t) \oplus \mu^*(t) \f$ and the
 * optimal trajectory is \f$ g^*(t) = g_{lin}(t) \oplus x^*(t) \f$.
 *
 * @note Constraints are added as \f$ A x \leq b - A \log g_{lin} \f$ and similarly for the input.
 * Beware of using constraints on non-Euclidean spaces.
 *
 * @note \p f must be differentiable w.r.t. \f$ g \f$ and \f$ u \f$  with the default \p
 * smooth::diff method (check \p smooth::diff::DefaultType). If using an automatic differentiation
 * method this means that it must be templated on the scalar type.
 */
template<std::size_t K,
  LieGroup G,
  Manifold U,
  typename Dyn,
  smooth::diff::Type DiffType = smooth::diff::Type::DEFAULT>
QuadraticProgramSparse<double> ocp_to_qp(const OptimalControlProblem<G, U> & pbm,
  const Dyn & f,
  const LinearizationInfo<G, U> & lin = LinearizationInfo<G, U>{})
{
  using std::placeholders::_1;

  // problem info
  static constexpr int Nx = Dof<G>;
  static constexpr int Nu = Dof<U>;

  static_assert(Nx > 0, "State space dimension must be static");
  static_assert(Nu > 0, "Input space dimension must be static");

  static constexpr int NX   = K * Nx;
  static constexpr int NU   = K * Nu;
  static constexpr int nvar = NX + NU;

  static constexpr int n_eq = NX;  // equality constraints from dynamics

  uint32_t n_u_iq = pbm.ulim ? pbm.ulim.value().A.rows() * K : 0;
  uint32_t n_g_iq = pbm.glim ? pbm.glim.value().A.rows() * K : 0;

  const uint32_t ncon = n_eq + n_u_iq + n_g_iq;

  QuadraticProgramSparse ret;
  ret.P.resize(nvar, nvar);
  ret.q.resize(nvar);

  ret.A.resize(ncon, nvar);
  ret.l.resize(ncon);
  ret.u.resize(ncon);

  // SET SPARSITY PATTERNS

  static constexpr auto Pp = mpc_nnz_pattern_P<K, Nx, Nu>();
  ret.P.reserve(Eigen::Map<const Eigen::Matrix<int, nvar, 1>>(Pp.data()));

  Eigen::Matrix<int, -1, 1> Ap(ncon);
  int Arow = 0;
  Ap.segment(Arow, Nx).setConstant(1 + Nu);
  Arow += Nx;
  Ap.segment(Arow, (K - 1) * Nx).setConstant(1 + Nu + Nx);
  Arow += (K - 1) * Nx;
  if (pbm.ulim) {
    Ap.segment(Arow, n_u_iq).setConstant(Nu);
    Arow += n_u_iq;
  }
  if (pbm.glim) {
    Ap.tail(n_g_iq).setConstant(Nx);
    Arow += n_g_iq;
  }
  ret.A.reserve(Ap);

  const double dt = pbm.T / static_cast<double>(K);

  // DYNAMICS CONSTRAINTS

  Arow = 0;

  for (auto k = 0u; k != K; ++k) {
    using AT = Eigen::Matrix<double, Nx, Nx>;
    using BT = Eigen::Matrix<double, Nx, Nu>;
    using ET = Eigen::Matrix<double, Nx, 1>;

    const double t = k * dt;

    // LINEARIZATION

    const auto xl  = lin.g(t);
    const auto dxl = lin.dg(t);
    const auto ul  = lin.u(t);

    const auto [flin, df_xu] = diff::dr<DiffType>(
      [&f, &t]<typename T>(const CastT<T, G> & vx,
        const CastT<T, U> & vu) -> Eigen::Matrix<T, Nx, 1> { return f(t, vx, vu); },
      wrt(xl, ul));

    // cltv system \dot x = At x(t) + Bt u(t) + Et
    const AT At = (-0.5 * ad<G>(flin) - 0.5 * ad<G>(dxl) + df_xu.template leftCols<Nx>());
    const BT Bt = df_xu.template rightCols<Nu>();
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
      for (auto i = 0u; i != Nx; ++i) { ret.A.insert(Nx * k + i, NU + Nx * k + i) = 1; }

      // B matrix on u(0)
      for (auto i = 0u; i != Nx; ++i) {
        for (auto j = 0u; j != Nu; ++j) { ret.A.insert(Nx * k + i, Nu * k + j) = -Bk(i, j); }
      }

      ret.u.template segment<Nx>(Nx * k) = Ak * (pbm.x0 - lin.g(t)) + Ek;
      ret.l.template segment<Nx>(Nx * k) = ret.u.template segment<Nx>(Nx * k);

    } else {
      // x(k+1) - A x(k) - B u(k) = E

      // identity matrix on x(k+1)
      for (auto i = 0u; i != Nx; ++i) { ret.A.insert(Nx * k + i, NU + Nx * k + i) = 1; }

      // A matrix on x(k)
      for (auto i = 0u; i != Nx; ++i) {
        for (auto j = 0u; j != Nx; ++j) {
          ret.A.insert(Nx * k + i, NU + Nx * (k - 1) + j) = -Ak(i, j);
        }
      }

      // B matrix on u(k)
      for (auto i = 0u; i != Nx; ++i) {
        for (auto j = 0u; j != Nu; ++j) { ret.A.insert(Nx * k + i, Nu * k + j) = -Bk(i, j); }
      }

      ret.u.template segment<Nx>(Nx * k) = Ek;
      ret.l.template segment<Nx>(Nx * k) = Ek;
    }
  }

  Arow += K * Nx;

  // INPUT CONSTRAINTS

  if (pbm.ulim) {
    for (auto k = 0u; k < K; ++k) {
      for (auto i = 0u; i != pbm.ulim.value().A.rows(); ++i) {
        for (auto j = 0u; j != Nu; ++j) {
          ret.A.insert(Arow + k * Nu + i, k * Nu + j) = pbm.ulim.value().A(i, j);
        }
      }
      ret.l.template segment<Nu>(Arow + k * Nu) =
        pbm.ulim.value().l - pbm.ulim.value().A * (lin.u(k * dt) - Identity<U>());
      ret.u.template segment<Nu>(Arow + k * Nu) =
        pbm.ulim.value().u - pbm.ulim.value().A * (lin.u(k * dt) - Identity<U>());
    }
    Arow += n_u_iq;
  }

  // linearization bounds
  /* for (auto k = 0u; k < K; ++k) {
    ret.l.template segment<Nu>(n_eq + k * Nu) =
      ret.l.template segment<Nu>(n_eq + k * Nu).cwiseMax(-lin.u_domain);
    ret.u.template segment<Nu>(n_eq + k * Nu) =
      ret.u.template segment<Nu>(n_eq + k * Nu).cwiseMin(lin.u_domain);
  } */

  // STATE CONSTRAINTS

  if (pbm.glim) {
    for (auto k = 1u; k != K + 1; ++k) {
      for (auto i = 0u; i != pbm.glim.value().A.rows(); ++i) {
        for (auto j = 0u; j != Nx; ++j) {
          ret.A.insert(Arow + (k - 1) * Nx + i, NU + (k - 1) * Nx + j) = pbm.glim.value().A(i, j);
        }
      }
      ret.l.template segment<Nx>(Arow + (k - 1) * Nx) =
        pbm.glim.value().l - pbm.glim.value().A * (lin.g(k * dt) - Identity<G>());
      ret.u.template segment<Nx>(Arow + (k - 1) * Nx) =
        pbm.glim.value().u - pbm.glim.value().A * (lin.g(k * dt) - Identity<G>());
    }
    Arow += n_g_iq;
  }

  // linearization bounds
  /* for (auto k = 1u; k < K + 1; ++k) {
    ret.l.template segment<Nx>(n_eq + n_u_iq + (k - 1) * Nx) =
      ret.l.template segment<Nx>(n_eq + n_u_iq + (k - 1) * Nx).cwiseMax(-lin.g_domain);
    ret.u.template segment<Nx>(n_eq + n_u_iq + (k - 1) * Nx) =
      ret.u.template segment<Nx>(n_eq + n_u_iq + (k - 1) * Nx).cwiseMin(lin.g_domain);
  } */

  // INPUT COSTS

  for (auto k = 0u; k < K; ++k) {
    for (auto i = 0u; i != Nu; ++i) {
      for (auto j = 0u; j != Nu; ++j) { ret.P.insert(k * Nu + i, k * Nu + j) = pbm.R(i, j) * dt; }
    }
    ret.q.template segment<Nu>(k * Nu) = pbm.R * (lin.u(k * dt) - pbm.udes(k * dt)) * dt;
  }

  // STATE COSTS

  // intermediate states x(1) ... x(K-1)
  for (auto k = 1u; k < K; ++k) {
    for (auto i = 0u; i != Nx; ++i) {
      for (auto j = 0u; j != Nx; ++j) {
        ret.P.insert(NU + (k - 1) * Nx + i, NU + (k - 1) * Nx + j) = pbm.Q(i, j) * dt;
      }
    }
    ret.q.template segment<Nx>(NU + (k - 1) * Nx) = pbm.Q * (lin.g(k * dt) - pbm.gdes(k * dt)) * dt;
  }

  // last state x(K) ~ x(T)
  for (auto i = 0u; i != Nx; ++i) {
    for (auto j = 0u; j != Nx; ++j) {
      ret.P.insert(NU + (K - 1) * Nx + i, NU + (K - 1) * Nx + j) = pbm.QT(i, j);
    }
  }
  ret.q.template segment<Nx>(NU + (K - 1) * Nx) = pbm.QT * (lin.g(pbm.T) - pbm.gdes(pbm.T));

  ret.A.makeCompressed();
  ret.P.makeCompressed();

  return ret;
}

/**
 * @brief Parameters for MPC
 */
struct MPCParams
{
  /// MPC horizon (seconds)
  double T{1};

  /**
   * @brief Enable warmstarting.
   *
   * @warning MPCParams::warmstart should not be enabled together with frequent relinearization.
   */
  bool warmstart{true};

  /**
   * @brief Relinearize the problem around state solution after each solve.
   */
  bool relinearize_state_around_sol{false};

  /**
   * @brief Relinearize the problem around input solution after each solve.
   */
  bool relinearize_input_around_sol{false};

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

  /**
   * @brief QP solvers parameters.
   */
  QPSolverParams qp{};
};

/**
 * @brief Model-Predictive Control (MPC) on Lie groups.
 *
 * @tparam K number of MPC discretization steps (seee ocp_to_qp).
 * @tparam T time type, must be a std::chrono::duration-like
 * @tparam G state space LieGroup type
 * @tparam U input space Manifold type
 * @tparam Dyn callable type that represents dynamics
 * @tparam DiffType differentiation method
 *
 * This MPC class keeps and repeatedly solves an internal OptimalControlProblem that is updated to
 * track a time-dependent trajectory defined through set_xudes().
 *
 * @note If the MPC problem is nonlinear a good (re-)linearization policy is required for good
 * performance. See MPCParams for various options. If none of the options are enabled it is up to
 * the user to update the linearizaiton in an appropriate way, or make sure that the problem is
 * linear. The linearization point can be manually updated via the linearization() member functions.
 */
template<std::size_t K,
  typename T,
  LieGroup G,
  Manifold U,
  typename Dyn,
  smooth::diff::Type DiffType = smooth::diff::Type::DEFAULT>
class MPC
{
public:
  /**
   * @brief Create an MPC instance.
   *
   * @param f callable object that represents dynamics \f$ \mathrm{d}^r x_t = f(f, x, u) \f$ as a
   * function \f$ f : T \times G \times U \rightarrow \mathbb{R}^{\dim \mathfrak{g}} \f$.
   * @param prm MPC parameters
   *
   * @note \f$ f \f$ is copied/moved into the class. In order to modify the dynamics from the
   * outside the type Dyn can be created to contain references to outside objects that are
   * updated by the user.
   */
  MPC(Dyn && f, MPCParams && prm) : prm_(std::move(prm)), dyn_(std::move(f)) {}

  /// Same as above but for lvalues
  MPC(const Dyn & f, const MPCParams & prm) : MPC(Dyn(f), MPCParams(prm)) {}

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
   * @param[out] u_traj (optional) return MPC input solution \f$ \{ \mu_k \mid k = 0, \ldots, K-1 \}
   * \f$.
   * @param[out] x_traj (optional) return MPC state solution \f$ \{ x_k \mid k = 0, \ldots, K \}
   * \f$.
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
  QPSolutionStatus operator()(U & u,
    const T & t,
    const G & g,
    std::optional<std::reference_wrapper<std::vector<U>>> u_traj = std::nullopt,
    std::optional<std::reference_wrapper<std::vector<G>>> x_traj = std::nullopt)
  {
    using std::chrono::duration, std::chrono::nanoseconds;

    static constexpr int Nx = Dof<G>;
    static constexpr int Nu = Dof<U>;
    static constexpr int NU = K * Nu;

    // update problem with funcitons defined in "MPC time"
    ocp_.x0   = g;
    ocp_.T    = prm_.T;
    ocp_.gdes = [this, &t](double t_loc) -> G {
      return x_des_(t + duration_cast<nanoseconds>(duration<double>(t_loc)));
    };
    ocp_.udes = [this, &t](double t_loc) -> U {
      return u_des_(t + duration_cast<nanoseconds>(duration<double>(t_loc)));
    };

    // linearize around desired trajectory
    if (!prm_.relinearize_input_around_sol || relinearize_around_desired_) { lin_.u = ocp_.udes; }
    if (!prm_.relinearize_state_around_sol || relinearize_around_desired_) {
      lin_.g  = ocp_.gdes;
      lin_.dg = [this](double t) -> Tangent<G> {
        // need numerical derivative here since gdes is not templated...
        auto [gv, dg] = diff::dr<diff::Type::NUMERICAL>(
          [this]<typename S>(const S & t_var) -> CastT<S, G> { return lin_.g(t_var); }, wrt(t));
        return dg;
      };
    }

    relinearize_around_desired_ = false;

    const double dt = ocp_.T / static_cast<double>(K);

    // define dynamics in "MPC time"
    const auto dyn = [this, &t]<typename S>(
                       double t_loc, const CastT<S, G> & vx, const CastT<S, U> & vu) {
      return dyn_(t + duration_cast<nanoseconds>(duration<double>(t_loc)), vx, vu);
    };

    const auto qp = ocp_to_qp<K, G, U, decltype(dyn), DiffType>(ocp_, dyn, lin_);
    auto sol      = solve_qp(qp, prm_.qp, warmstart_);

    for (auto i = 0u; i < prm_.iterative_relinearization; ++i) {
      // check if solution touches linearization domain
      bool touches = false;
      for (int k = 0; !touches && k != K; ++k) {
        // clang-format off
        if (((1. - 1e-6) * lin_.u_domain - sol.primal.template segment<Nu>(     k * Nu).cwiseAbs()).minCoeff() < 0) { touches = true; }
        if (((1. - 1e-6) * lin_.g_domain - sol.primal.template segment<Nx>(NU + k * Nx).cwiseAbs()).minCoeff() < 0) { touches = true; }
        // clang-format on
      }
      if (touches) {
        // relinearize around solution and solve again
        relinearize_state_around_sol(sol);
        relinearize_input_around_sol(sol);

        const auto qp = ocp_to_qp<K, G, U, decltype(dyn), DiffType>(ocp_, dyn, lin_);
        sol           = solve_qp(qp, prm_.qp, warmstart_);
      } else {
        // solution seems fine
        break;
      }
    }

    // output result
    u = lin_.u(0) + sol.primal.template head<Nu>();

    // output solution trajectories
    if (u_traj.has_value()) {
      u_traj.value().get().resize(K);
      for (auto i = 0u; i < K; ++i) {
        u_traj.value().get()[i] = lin_.u(i * dt) + sol.primal.template segment<Nu>(i * Nu);
      }
    }

    if (x_traj.has_value()) {
      x_traj.value().get().resize(K + 1);
      x_traj.value().get()[0] = ocp_.x0;
      for (auto i = 1u; i < K + 1; ++i) {
        x_traj.value().get()[i] =
          lin_.g(i * dt) + sol.primal.template segment<Nx>(NU + (i - 1) * Nx);
      }
    }

    // update linearization for next iteration
    if (sol.code == QPSolutionStatus::Optimal) {
      if (prm_.relinearize_input_around_sol) relinearize_input_around_sol(sol);
      if (prm_.relinearize_state_around_sol) relinearize_state_around_sol(sol);
    }

    // save solution for warmstart
    if (prm_.warmstart
        && (sol.code == QPSolutionStatus::Optimal || sol.code == QPSolutionStatus::MaxTime
            || sol.code == QPSolutionStatus::MaxIterations)) {
      warmstart_ = sol;  // store successful solution for later warmstart
    }

    return sol.code;
  }

  /**
   * @brief Set state bounds.
   */
  void set_glim(const OptimalControlBounds<G> & lim) { ocp_.glim = lim; }

  /**
   * @brief Set input bounds.
   */
  void set_ulim(const OptimalControlBounds<U> & lim) { ocp_.ulim = lim; }

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
   * @param x_des desired state trajectory \f$ g_{des} (t) \f$ as function \f$ T \rightarrow G \f$
   * @param u_des desired input trajectory \f$ u_{des} (t) \f$ as function \f$ T \rightarrow U \f$
   */
  void set_xudes(std::function<G(T)> && x_des, std::function<U(T)> && u_des)
  {
    x_des_                      = std::move(x_des);
    u_des_                      = std::move(u_des);
    relinearize_around_desired_ = true;
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
   * @brief Relinearize state around a solution.
   */
  void relinearize_state_around_sol(const QPSolution<-1, -1, double> & sol)
  {
    static constexpr int Nx = Dof<G>;
    static constexpr int Nu = Dof<U>;
    static constexpr int NU = K * Nu;
    const double dt         = ocp_.T / static_cast<double>(K);

    // STATE LINEARIZATION
    std::vector<double> tt(K + 1);
    std::vector<G> gg(K + 1);
    tt[0] = 0;
    gg[0] = ocp_.x0;

    for (auto k = 1u; k != K + 1; ++k) {
      tt[k] = dt * k;
      gg[k] = lin_.g(k * dt) + sol.primal.template segment<Nx>(NU + (k - 1) * Nx);
    }

    auto g_spline = smooth::fit_cubic_bezier(tt, gg);
    lin_.g        = [g_spline = g_spline](double t) -> G { return g_spline(t); };
    lin_.dg       = [g_spline = std::move(g_spline)](double t) -> Tangent<G> {
      Tangent<G> ret;
      g_spline(t, ret);
      return ret;
    };
  }

  /**
   * @brief Relinearize input around a solution.
   */
  void relinearize_input_around_sol(const QPSolution<-1, -1, double> & sol)
  {
    static constexpr int Nu = Dof<U>;
    const double dt         = ocp_.T / static_cast<double>(K);

    std::vector<double> tt(K);
    std::vector<U> uu(K);

    for (auto k = 0u; k != K; ++k) {
      tt[k] = k * dt;
      uu[k] = lin_.u(k * dt) + sol.primal.template segment<Nu>(k * Nu);
    }

    auto u_spline = smooth::fit_linear_bezier(tt, uu);
    lin_.u        = [u_spline = std::move(u_spline)](double t) -> U { return u_spline(t); };
  }

private:
  MPCParams prm_{};
  Dyn dyn_{};

  OptimalControlProblem<G, U> ocp_{};

  bool relinearize_around_desired_{false};
  LinearizationInfo<G, U> lin_{};

  std::optional<QPSolution<-1, -1, double>> warmstart_{};

  std::function<G(T)> x_des_ = [](T) -> G { return Identity<G>(); };
  std::function<U(T)> u_des_ = [](T) -> U { return Identity<U>(); };
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__MPC_HPP_
