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

#ifndef SMOOTH__FEEDBACK__MPC_FUNC_HPP_
#define SMOOTH__FEEDBACK__MPC_FUNC_HPP_

/**
 * @file
 * @brief Functions for Model-Predictive Control (MPC) on Lie groups.
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
  /// Initial state
  G x0{Default<G>()};

  /// Desired input trajectory
  std::function<U(double)> udes{
    [](double) -> U { return Default<U>(); },
  };
  /// Desired state trajectory
  std::function<G(double)> gdes{
    [](double) -> G { return Default<G>(); },
  };

  /// Input bounds
  ManifoldBounds<U> ulim{};
  /// State bounds
  ManifoldBounds<G> glim{};

  /// MPC weights struct
  struct Weights
  {
    /// Running state cost
    Eigen::Matrix<double, Nx, Nx> Q{
      Eigen::Matrix<double, Nx, Nx>::Identity(),
    };
    /// Final state cost
    Eigen::Matrix<double, Nx, Nx> QT{
      Eigen::Matrix<double, Nx, Nx>::Identity(),
    };
    /// Running input cost
    Eigen::Matrix<double, Nu, Nu> R{
      Eigen::Matrix<double, Nu, Nu>::Identity(),
    };
  };

  /// MPC weights values
  Weights weights{};
};

/**
 * @brief Struct to define a linearization point.
 */
template<LieGroup G, Manifold U>
struct LinearizationInfo
{
  /**
   * @brief state linearization trajectory with first derivative
   * \f$ g_{lin}: \mathbb{R} \rightarrow (G, T / G) \f$
   */
  std::function<std::pair<G, Tangent<G>>(double)> g{
    [](double) -> std::pair<G, Tangent<G>> {
      return {Default<G>(), Tangent<G>::Zero()};
    },
  };

  /**
   * @brief input linearization trajectory \f$ u_{lin}(t) :  \mathbb{R} \rightarrow / G \f$
   */
  std::function<U(double)> u{
    [](double) -> U { return Default<U>(); },
  };

  /**
   * @brief Domain of validity of state linearization
   *
   *  Defines an upper bound \f$ \bar a \f$ s.t. the linearization is valid for \f$ g \f$ s.t.
   * \f[
   *   \left\| g \ominus_r g_{lin} \right \| \leq \bar a.
   * \f]
   */
  Eigen::Matrix<double, Dof<G>, 1> g_domain{
    Eigen::Matrix<double, Dof<G>, 1>::Constant(std::numeric_limits<double>::infinity()),
  };
};

/**
 * @brief Allocate QP sparsity pattern (part 1 of ocp_to_qp()).
 *
 * Variables: [x_1, ..., x_K, u_0, ..., u_{K-1}]
 *
 * Constraints:
 *  - Dynamics constraints            (K * Nx)
 *  - Input constraints               (Nu_ineq * Nu)
 *  - State constraints               (Nx_ineq * Nx)
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

  const uint32_t n_eq     = K * Nx;
  const uint32_t NU_iq    = K * pbm.ulim.A.rows();
  const uint32_t NX_iq    = K * pbm.glim.A.rows();
  const uint32_t NXLIN_iq = lin_con ? K * Nu : 0;

  const uint32_t nvar = K * Nx + K * Nu;
  const uint32_t ncon = n_eq + NU_iq + NX_iq + NXLIN_iq;

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
  if (NU_iq > 0) { Ap.segment(Arow, NU_iq).setConstant(Nu); }
  Arow += NU_iq;

  // state constraints
  if (NX_iq > 0) { Ap.segment(Arow, NX_iq).setConstant(Nx); }
  Arow += NX_iq;

  // state linearization constraints
  if (NXLIN_iq > 0) { Ap.segment(Arow, NXLIN_iq).setConstant(1); }

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

  const bool lin_con = lin.g_domain.minCoeff() < std::numeric_limits<double>::infinity();

  const uint32_t Nu_iq    = pbm.ulim.A.rows();
  const uint32_t Nx_iq    = pbm.glim.A.rows();
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
          qp.A.coeffRef(Arow + k * Nu_iq + i, k * Nu + j) = pbm.ulim.A(i, j);
        }
      }
      // clang-format off
      qp.l.segment(Arow + k * Nu_iq, Nu_iq) = pbm.ulim.l - pbm.ulim.A * rminus(lin.u(k * dt), pbm.ulim.c);
      qp.u.segment(Arow + k * Nu_iq, Nu_iq) = pbm.ulim.u - pbm.ulim.A * rminus(lin.u(k * dt), pbm.ulim.c);
      // clang-format on
    }
  }
  Arow += K * Nu_iq;

  // STATE CONSTRAINTS

  if (Nx_iq > 0) {
    for (auto k = 1u; k != K + 1; ++k) {
      for (auto i = 0u; i != Nx_iq; ++i) {
        for (auto j = 0u; j != Nx; ++j) {
          qp.A.coeffRef(Arow + (k - 1) * Nx_iq + i, NU + (k - 1) * Nx + j) = pbm.glim.A(i, j);
        }
      }
      // clang-format off
      qp.l.segment(Arow + (k - 1) * Nx_iq, Nx_iq) = pbm.glim.l - pbm.glim.A * rminus(lin.g(k * dt).first, pbm.glim.c);
      qp.u.segment(Arow + (k - 1) * Nx_iq, Nx_iq) = pbm.glim.u - pbm.glim.A * rminus(lin.g(k * dt).first, pbm.glim.c);
      // clang-format on
    }
  }
  Arow += K * Nx_iq;

  // STATE LINEARIZATION BOUNDS

  if (Nxlin_iq > 0) {
    for (auto k = 1u; k < K + 1; ++k) {
      for (auto i = 0u; i != Nxlin_iq; ++i) {
        qp.A.coeffRef(Arow + (k - 1) * Nxlin_iq + i, NU + (k - 1) * Nx + i) = 1.;
      }
      qp.l.segment(Arow + (k - 1) * Nxlin_iq, Nxlin_iq) = -lin.g_domain;
      qp.u.segment(Arow + (k - 1) * Nxlin_iq, Nxlin_iq) = lin.g_domain;
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
    qp.q.segment(k * Nu, Nu) = pbm.weights.R * rminus(lin.u(k * dt), pbm.udes(k * dt)) * dt;
  }

  // STATE COSTS

  // intermediate states x(1) ... x(K-1)
  for (auto k = 1u; k < K; ++k) {
    for (auto i = 0u; i != Nx; ++i) {
      for (auto j = 0u; j != Nx; ++j) {
        qp.P.coeffRef(NU + (k - 1) * Nx + i, NU + (k - 1) * Nx + j) = pbm.weights.Q(i, j) * dt;
      }
    }
    qp.q.segment(NU + (k - 1) * Nx, Nx) =
      pbm.weights.Q * rminus(lin.g(k * dt).first, pbm.gdes(k * dt)) * dt;
  }

  // last state x(K) ~ x(T)
  for (auto i = 0u; i != Nx; ++i) {
    for (auto j = 0u; j != Nx; ++j) {
      qp.P.coeffRef(NU + (K - 1) * Nx + i, NU + (K - 1) * Nx + j) = pbm.weights.QT(i, j);
    }
  }
  qp.q.segment(NU + (K - 1) * Nx, Nx) =
    pbm.weights.QT * rminus(lin.g(pbm.T).first, pbm.gdes(pbm.T));
}

/**
 * @brief Convert OptimalControlProblem on \f$ (\mathbb{G}, \mathbb{U}) \f$ into a tangent space
 * QuadraticProgramSparse on \f$ (\mathbb{R}^{\dim \mathfrak{g}}, \mathbb{R}^{\dim \mathfrak{u}})
 * \f$.
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

  auto qp = ocp_to_qp_allocate<G, U>(pbm, K, lin_con);
  ocp_to_qp_fill<G, U>(pbm, K, f, lin, qp);

  return qp;
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__MPC_FUNC_HPP_
