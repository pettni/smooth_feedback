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

#include "common.hpp"
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
 *     + (g(T) \ominus g_{des}(T))^T Q_T (g(T) \ominus g_{des}(T))               \\
 *    \text{s.t.}    & g(0) = x_0,                                               \\
 *                    & l_G \leq A_G (g(t) \ominus c_G) \leq u_G,                \\
 *                    & l_U \leq A_U (u(t) \ominus c_U) \leq u_U,
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

  /// Time horizon
  double T{1};
  /// Initial state (must be defined)
  G x0;

  /// Desired input trajectory (must be defined)
  std::function<U(double)> udes;
  /// Desired state trajectory (must be defined)
  std::function<G(double)> gdes;

  /// Input bounds (optional)
  std::optional<ManifoldBounds<U>> ulim{};
  /// State bounds (optional)
  std::optional<ManifoldBounds<G>> glim{};

  /// MPC weights struct
  struct Weights
  {
    /// Running state cost
    Eigen::Matrix<double, Nx, Nx> Q = Eigen::Matrix<double, Nx, Nx>::Identity();
    /// Final state cost
    Eigen::Matrix<double, Nx, Nx> QT = Eigen::Matrix<double, Nx, Nx>::Identity();
    /// Running input cost
    Eigen::Matrix<double, Nu, Nu> R = Eigen::Matrix<double, Nu, Nu>::Identity();
  };

  /// MPC weights values
  Weights weights{};
};

/**
 * @brief Struct to define a linearization point.
 *
 * Default constructor linearizes around Identity (LieGroup) or Zero (non-LieGroup Manifold)
 */
template<LieGroup G, Manifold U>
struct LinearizationInfo
{
  /**
   * @brief state linearization trajectory with first derivative
   * \f$ g_{lin}: \mathbb{R} \rightarrow (G, T / G) \f$
   */
  std::function<std::pair<G, Tangent<G>>(double)> g;

  /**
   * @brief input linearization trajectory \f$ u_{lin}(t) :  \mathbb{R} \rightarrow / G \f$
   */
  std::function<U(double)> u;

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
};

namespace detail {
/**
 * @brief Allocate QP sparsity pattern (part 1 of ocp_to_qp()).
 *
 * Variables: [x_1, ..., x_K, u_0, ..., u_{K-1}]
 *
 * Constraints:
 *  - Dynamics constraints            (K * Nx)
 *  - Input constraints               (Nu_ineq * Nu)  [optional]
 *  - State constraints               (Nx_ineq * Nx)  [optional]
 *  - State linearization constraints (K * Nu)        [optional]
 *
 * @param pbm OptimalControlProblem definition.
 * @param K number of time discretization steps
 * @param lin_con set to true to allocate K * Nu state linearization constraints
 *
 * @returns sparse quadratic program definition with allocated matrices.
 */
template<LieGroup G, Manifold U>
QuadraticProgramSparse<double> ocp_to_qp_allocate(
  const OptimalControlProblem<G, U> & pbm, std::size_t K, bool lin_con = false)
{
  // problem info
  static constexpr int Nx = Dof<G>;
  static constexpr int Nu = Dof<U>;

  static_assert(Nx > 0, "State space dimension must be static");
  static_assert(Nu > 0, "Input space dimension must be static");

  const int n_eq      = K * Nx;
  const int n_u_iq    = pbm.ulim ? pbm.ulim.value().A.rows() * K : 0;
  const int n_g_iq    = pbm.glim ? pbm.glim.value().A.rows() * K : 0;
  const int n_glin_iq = lin_con ? K * Nu : 0;

  const int nvar = K * Nx + K * Nu;
  const int ncon = n_eq + n_u_iq + n_g_iq + n_glin_iq;

  QuadraticProgramSparse qp;

  // Matrix sizes
  qp.P.resize(nvar, nvar);
  qp.q.resize(nvar);

  qp.A.resize(ncon, nvar);
  qp.l.resize(ncon);
  qp.u.resize(ncon);

  // SPARSITY PATTERN FOR P

  Eigen::Matrix<int, -1, 1> Pp(nvar);
  for (std::size_t i = 0u; i != K * Nu; ++i) { Pp[i] = Nu; }
  for (std::size_t i = K * Nu; i != K * (Nu + Nx); ++i) { Pp[i] = Nx; }

  // SPARSITY PATTERN FOR A

  Eigen::Matrix<int, -1, 1> Ap(ncon);
  int Arow = 0;

  // state constraint k = 0
  Ap.segment(Arow, Nx).setConstant(1 + Nu);
  Arow += Nx;

  // state constraints k = 1, ... K
  Ap.segment(Arow, (K - 1) * Nx).setConstant(1 + Nu + Nx);
  Arow += (K - 1) * Nx;

  // input constraints
  if (n_u_iq > 0) { Ap.segment(Arow, n_u_iq).setConstant(Nu); }
  Arow += n_u_iq;

  // state constraints
  if (n_g_iq > 0) { Ap.segment(Arow, n_g_iq).setConstant(Nx); }
  Arow += n_g_iq;

  // state linearization constraints
  if (n_glin_iq > 0) { Ap.segment(Arow, n_glin_iq).setConstant(1); }

  qp.P.reserve(Pp);
  qp.A.reserve(Ap);

  return qp;
}

/**
 * @brief Fill QP matrices (part 2 of ocp_to_qp()).
 */
template<LieGroup G, Manifold U, typename Dyn, diff::Type DT = diff::Type::DEFAULT>
void ocp_to_qp_fill(const OptimalControlProblem<G, U> & pbm,
  std::size_t K,
  const Dyn & f,
  const LinearizationInfo<G, U> & lin,
  QuadraticProgramSparse<double> & qp)
{
  using std::placeholders::_1;

  static constexpr int Nx = Dof<G>;
  static constexpr int Nu = Dof<U>;
  const int NU            = K * Nu;

  static_assert(Nx > 0, "State space dimension must be static");
  static_assert(Nu > 0, "Input space dimension must be static");

  bool lin_con = lin.g_domain.minCoeff() < std::numeric_limits<double>::infinity();

  const uint32_t Nu_iq    = pbm.ulim ? pbm.ulim.value().A.rows() : 0;
  const uint32_t Nx_iq    = pbm.glim ? pbm.glim.value().A.rows() : 0;
  const uint32_t Nxlin_iq = lin_con ? Nx : 0;

  const double dt = pbm.T / static_cast<double>(K);

  ////////////////////
  /// FILL A, l, u ///
  ////////////////////

  int Arow = 0;

  for (auto k = 0u; k != K; ++k) {
    using AT = Eigen::Matrix<double, Nx, Nx>;
    using BT = Eigen::Matrix<double, Nx, Nu>;
    using ET = Eigen::Matrix<double, Nx, 1>;

    const double t = k * dt;

    // LINEARIZATION

    const auto [xl, dxl]     = lin.g(t);
    const auto ul            = lin.u(t);
    const auto f_t           = [&f, &t]<typename T>(const CastT<T, G> & vx,
                       const CastT<T, U> & vu) -> Eigen::Matrix<T, Nx, 1> { return f(t, vx, vu); };
    const auto [flin, df_xu] = diff::dr<DT>(f_t, wrt(xl, ul));

    // cltv system \dot x = At x(t) + Bt u(t) + Et
    const AT At = -0.5 * ad<G>(flin) - 0.5 * ad<G>(dxl) + df_xu.template leftCols<Nx>();
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
      for (auto i = 0u; i != Nx; ++i) { qp.A.coeffRef(Nx * k + i, NU + Nx * k + i) = 1; }

      // B matrix on u(0)
      for (auto i = 0u; i != Nx; ++i) {
        for (auto j = 0u; j != Nu; ++j) { qp.A.coeffRef(Nx * k + i, Nu * k + j) = -Bk(i, j); }
      }

      qp.u.template segment<Nx>(Nx * k) = Ak * rminus(pbm.x0, xl) + Ek;
      qp.l.template segment<Nx>(Nx * k) = qp.u.template segment<Nx>(Nx * k);
    } else {
      // x(k+1) - A x(k) - B u(k) = E

      // identity matrix on x(k+1)
      for (auto i = 0u; i != Nx; ++i) { qp.A.coeffRef(Nx * k + i, NU + Nx * k + i) = 1; }

      // A matrix on x(k)
      for (auto i = 0u; i != Nx; ++i) {
        for (auto j = 0u; j != Nx; ++j) {
          qp.A.coeffRef(Nx * k + i, NU + Nx * (k - 1) + j) = -Ak(i, j);
        }
      }

      // B matrix on u(k)
      for (auto i = 0u; i != Nx; ++i) {
        for (auto j = 0u; j != Nu; ++j) { qp.A.coeffRef(Nx * k + i, Nu * k + j) = -Bk(i, j); }
      }

      qp.u.template segment<Nx>(Nx * k) = Ek;
      qp.l.template segment<Nx>(Nx * k) = Ek;
    }
  }
  Arow += K * Nx;

  // INPUT CONSTRAINTS

  if (Nu_iq > 0) {
    for (auto k = 0u; k < K; ++k) {
      for (auto i = 0u; i != Nu_iq; ++i) {
        for (auto j = 0u; j != Nu; ++j) {
          qp.A.coeffRef(Arow + k * Nu_iq + i, k * Nu + j) = pbm.ulim.value().A(i, j);
        }
      }
      qp.l.template segment(Arow + k * Nu_iq, Nu_iq) =
        pbm.ulim.value().l - pbm.ulim.value().A * rminus(lin.u(k * dt), pbm.ulim.value().c);
      qp.u.template segment(Arow + k * Nu_iq, Nu_iq) =
        pbm.ulim.value().u - pbm.ulim.value().A * rminus(lin.u(k * dt), pbm.ulim.value().c);
    }
  }
  Arow += K * Nu_iq;

  // STATE CONSTRAINTS

  if (Nx_iq) {
    for (auto k = 1u; k != K + 1; ++k) {
      for (auto i = 0u; i != Nx_iq; ++i) {
        for (auto j = 0u; j != Nx; ++j) {
          qp.A.coeffRef(Arow + (k - 1) * Nx_iq + i, NU + (k - 1) * Nx + j) =
            pbm.glim.value().A(i, j);
        }
      }
      qp.l.template segment(Arow + (k - 1) * Nx_iq, Nx_iq) =
        pbm.glim.value().l - pbm.glim.value().A * rminus(lin.g(k * dt).first, pbm.glim.value().c);
      qp.u.template segment(Arow + (k - 1) * Nx_iq, Nx_iq) =
        pbm.glim.value().u - pbm.glim.value().A * rminus(lin.g(k * dt).first, pbm.glim.value().c);
    }
  }
  Arow += K * Nx_iq;

  // STATE LINEARIZATION BOUNDS

  if (Nxlin_iq > 0) {
    for (auto k = 1u; k < K + 1; ++k) {
      for (auto i = 0u; i != Nxlin_iq; ++i) {
        qp.A.coeffRef(Arow + (k - 1) * Nxlin_iq + i, NU + (k - 1) * Nx + i) = 1.;
      }
      qp.l.template segment(Arow + (k - 1) * Nxlin_iq, Nxlin_iq) = -lin.g_domain;
      qp.u.template segment(Arow + (k - 1) * Nxlin_iq, Nxlin_iq) = lin.g_domain;
    }
  }
  Arow += K * Nxlin_iq;

  ////////////////
  /// FILL P,q ///
  ////////////////

  // INPUT COSTS

  for (auto k = 0u; k < K; ++k) {
    for (auto i = 0u; i != Nu; ++i) {
      for (auto j = 0u; j != Nu; ++j) {
        qp.P.coeffRef(k * Nu + i, k * Nu + j) = pbm.weights.R(i, j) * dt;
      }
    }
    qp.q.template segment<Nu>(k * Nu) =
      pbm.weights.R * rminus(lin.u(k * dt), pbm.udes(k * dt)) * dt;
  }

  // STATE COSTS

  // intermediate states x(1) ... x(K-1)
  for (auto k = 1u; k < K; ++k) {
    for (auto i = 0u; i != Nx; ++i) {
      for (auto j = 0u; j != Nx; ++j) {
        qp.P.coeffRef(NU + (k - 1) * Nx + i, NU + (k - 1) * Nx + j) = pbm.weights.Q(i, j) * dt;
      }
    }
    qp.q.template segment<Nx>(NU + (k - 1) * Nx) =
      pbm.weights.Q * rminus(lin.g(k * dt).first, pbm.gdes(k * dt)) * dt;
  }

  // last state x(K) ~ x(T)
  for (auto i = 0u; i != Nx; ++i) {
    for (auto j = 0u; j != Nx; ++j) {
      qp.P.coeffRef(NU + (K - 1) * Nx + i, NU + (K - 1) * Nx + j) = pbm.weights.QT(i, j);
    }
  }
  qp.q.template segment<Nx>(NU + (K - 1) * Nx) =
    pbm.weights.QT * rminus(lin.g(pbm.T).first, pbm.gdes(pbm.T));
}

}  // namespace detail

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
 * performance.
 * @tparam G problem state group type \f$ \mathbb{G} \f$
 * @tparam U problem input group type \f$ \mathbb{U} \f$
 * @tparam Dyn dynamics functor type
 * @tparam DT differentiation method to utilize
 *
 * @param pbm optimal control problem
 * @param K number of discretization points. More points create a larger QP, but the distance \f$
 * T / K \f$ between points should be smaller than the smallest system time constant for adequate
 * @param f dynamics \f$ f : \mathbb{R} \times \mathbb{G} \times \mathbb{U} \rightarrow
 * \mathbb{R}^{\dim \mathfrak g}\f$ s.t. \f$ \mathrm{d}^r g_t = f(t, g, u) \f$
 * @param lin linearization point
 *
 * @return QuadraticProgramSparse modeling the input optimal control problem.
 *
 * @note Given a solution \f$(x^*, \mu^*)\f$ to the QuadraticProgramSparse, the corresponding
 * solution to the OptimalControlProblem is \f$ u^*(t) = u_{lin}(t) \oplus \mu^*(t) \f$ and the
 * optimal trajectory is \f$ g^*(t) = g_{lin}(t) \oplus x^*(t) \f$.
 *
 * @note Constraints are added as \f$ A x \leq b - A (g_{lin} - c) \f$ and similarly for the
 * input. Beware of using constraints on non-Euclidean spaces.
 *
 * @note \p f must be differentiable w.r.t. \f$ g \f$ and \f$ u \f$  with the default \p
 * smooth::diff method (check \p smooth::diff::DefaultType). If using an automatic differentiation
 * method this means that it must be templated on the scalar type.
 */
template<LieGroup G, Manifold U, typename Dyn, diff::Type DT = diff::Type::DEFAULT>
QuadraticProgramSparse<double> ocp_to_qp(const OptimalControlProblem<G, U> & pbm,
  std::size_t K,
  const Dyn & f,
  const LinearizationInfo<G, U> & lin)
{
  bool lin_con = lin.g_domain.minCoeff() < std::numeric_limits<double>::infinity();

  auto qp = detail::ocp_to_qp_allocate<G, U>(pbm, K, lin_con);
  detail::ocp_to_qp_fill<G, U>(pbm, K, f, lin, qp);

  return qp;
}

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
   * @brief State bounds (optional)
   */
  std::optional<ManifoldBounds<G>> glim{};

  /**
   * @brief Input bounds (optional)
   */
  std::optional<ManifoldBounds<U>> ulim{};

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
 * @tparam T time type, must be a std::chrono::duration-like
 * @tparam G state space LieGroup type
 * @tparam U input space Manifold type
 * @tparam Dyn callable type that represents dynamics
 * @tparam DT differentiation method
 *
 * This MPC class keeps and repeatedly solves an internal OptimalControlProblem that is updated to
 * track a time-dependent trajectory defined through set_xudes().
 *
 * @note If the MPC problem is nonlinear a good (re-)linearization policy is required for good
 * performance. See MPCParams for various options. If none of the options are enabled it is up to
 * the user to update the linearizaiton in an appropriate way, or make sure that the problem is
 * linear. The linearization point can be manually updated via the linearization() member functions.
 */
template<typename T, LieGroup G, Manifold U, typename Dyn, diff::Type DT = diff::Type::DEFAULT>
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
  MPC(Dyn && f, MPCParams<G, U> && prm = MPCParams<G, U>{})
      : prm_(std::move(prm)), dyn_(std::move(f))
  {
    ocp_.T       = prm_.T;
    ocp_.glim    = prm_.glim;
    ocp_.ulim    = prm_.ulim;
    ocp_.weights = prm_.weights;
    qp_          = detail::ocp_to_qp_allocate<G, U>(ocp_, prm_.K);
  }
  /// Same as above but for lvalues
  MPC(const Dyn & f, const MPCParams<G, U> & prm = MPCParams<G, U>{})
      : MPC(Dyn(f), MPCParams<G, U>(prm))
  {}
  /// Default constructor
  MPC() : MPC(Dyn(), MPCParams<G, U>()) {}
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
   * @warning set_xdes() and set_udes() must be called before calling this function for the first
   * time.
   *
   * @param[in] t current time
   * @param[in] g current state
   * @param[out] u_traj (optional) return MPC input solution \f$ \{ \mu_k \mid k = 0, \ldots, K-1 \}
   * \f$.
   * @param[out] x_traj (optional) return MPC state solution \f$ \{ x_k \mid k = 0, \ldots, K \}
   * \f$.
   *
   * @return {u, code}
   */
  std::pair<U, QPSolutionStatus> operator()(const T & t,
    const G & g,
    std::optional<std::reference_wrapper<std::vector<U>>> u_traj = std::nullopt,
    std::optional<std::reference_wrapper<std::vector<G>>> x_traj = std::nullopt)
  {
    using std::chrono::duration, std::chrono::nanoseconds;

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

    // linearize around new desired trajectory
    if (!prm_.relinearize_around_solution || relinearize_around_desired_) {
      lin_.u = [this, &t](double t_loc) -> U {
        return u_des_(t + duration_cast<nanoseconds>(duration<double>(t_loc)));
      };
      lin_.g = [this, &t](double t_loc) -> std::pair<G, Tangent<G>> {
        return x_des_(t + duration_cast<nanoseconds>(duration<double>(t_loc)));
      };
      relinearize_around_desired_ = false;
    }

    const double dt = ocp_.T / static_cast<double>(prm_.K);

    // define dynamics in "MPC time"
    const auto dyn = [this, &t]<typename S>(
                       double t_loc, const CastT<S, G> & vx, const CastT<S, U> & vu) {
      return dyn_(t + duration_cast<nanoseconds>(duration<double>(t_loc)), vx, vu);
    };

    detail::ocp_to_qp_fill<G, U, decltype(dyn), DT>(ocp_, prm_.K, dyn, lin_, qp_);
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
        detail::ocp_to_qp_fill<G, U, decltype(dyn), DT>(ocp_, prm_.K, dyn, lin_, qp_);
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
   * @brief Set the desired input trajectory (lvalue version).
   *
   * @note This function triggers a relinearization around the desired input and trajectory at the
   * next call to operator()().
   *
   * @param u_des desired input trajectory \f$ u_{des} (t) \f$ as function \f$ T \rightarrow U \f$
   */
  void set_udes(const std::function<U(T)> & u_des) { set_xudes(std::function<U(T)>(u_des)); };

  /**
   * @brief Set the desired input trajectory (rvalue version).
   *
   * @note This function triggers a relinearization around the desired input and trajectory at the
   * next call to operator()().
   *
   * @param u_des desired input trajectory \f$ u_{des} (t) \f$ as function \f$ T \rightarrow U \f$
   */
  void set_udes(std::function<U(T)> && u_des)
  {
    u_des_                      = std::move(u_des);
    relinearize_around_desired_ = true;
  };

  /**
   * @brief Set the desired state trajectory by providing value and derivative (lvalue version).
   *
   * @note This function triggers a relinearization around the desired input and trajectory at the
   * next call to operator()().
   *
   * @param x_des desired state trajectory \f$ g_{des} (t) \f$ as function \f$ T \rightarrow (G,
   * \mathbb{R}^{\dim G} \f$
   */
  void set_xdes(const std::function<std::pair<G, Tangent<G>>(T)> & x_des)
  {
    set_xdes(std::function<std::pair<G, Tangent<G>>(T)>(x_des));
  };

  /**
   * @brief Set the desired state trajectory by providing value and derivative (rvalue version).
   *
   * @note This function triggers a relinearization around the desired input and trajectory at the
   * next call to operator()().
   *
   * @param x_des desired state trajectory \f$ g_{des} (t) \f$ as function \f$ T \rightarrow (G,
   * \mathbb{R}^{\dim G} \f$
   */
  void set_xdes(std::function<std::pair<G, Tangent<G>>(T)> && x_des)
  {
    x_des_                      = std::move(x_des);
    relinearize_around_desired_ = true;
  };

  /**
   * @brief Reset initial guess for next iteration to zero.
   */
  void reset_warmstart() { warmstart_ = {}; }

  /**
   * @brief Relinearize state around a solution.
   */
  void relinearize_around_sol(const QPSolution<-1, -1, double> & sol)
  {
    const double dt = ocp_.T / static_cast<double>(prm_.K);

    auto g_spline =
      smooth::fit_cubic_bezier(std::views::iota(0u, prm_.K + 1)
                                 | std::views::transform([&](auto k) -> double { return dt * k; }),
        std::views::iota(0u, prm_.K + 1) | std::views::transform([&](auto k) -> G {
          if (k == 0) {
            return ocp_.x0;
          } else {
            return rplus(lin_.g(dt * k).first,
              sol.primal.template segment<Dof<G>>(prm_.K * Dof<U> + (k - 1) * Dof<G>));
          }
        }));

    lin_.g = [g_spline = std::move(g_spline)](double t) -> std::pair<G, Tangent<G>> {
      Tangent<G> dg;
      auto g = g_spline(t, dg);
      return std::make_pair(g, dg);
    };
  }

private:
  // parameters
  const MPCParams<G, U> prm_{};

  // dynamics description
  Dyn dyn_{};

  // problem description
  OptimalControlProblem<G, U> ocp_;

  // flag to keep track of linearization point, and current linearization
  bool relinearize_around_desired_{false};
  LinearizationInfo<G, U> lin_{};

  // pre-allocated QP matrices
  QuadraticProgramSparse<double> qp_;

  // store last solution for warmstarting
  std::optional<QPSolution<-1, -1, double>> warmstart_{};

  // desired state (pos + vel) and input trajectories
  std::function<std::pair<G, Tangent<G>>(T)> x_des_;
  std::function<U(T)> u_des_;
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__MPC_HPP_
