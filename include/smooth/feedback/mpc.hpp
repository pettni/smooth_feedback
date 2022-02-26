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

#include <smooth/algo/hessian.hpp>
#include <smooth/lie_group.hpp>

#include <memory>

#include "ocp_to_qp.hpp"
#include "time.hpp"
#include "utils/d2r_exp_sparse.hpp"
#include "utils/dr_exp_sparse.hpp"
#include "utils/sparse.hpp"

namespace smooth::feedback {

template<LieGroup X, Manifold U>
struct MPCWeights
{
  static constexpr auto Nx = Dof<X>;
  static constexpr auto Nu = Dof<U>;

  /// Running state cost
  Eigen::Matrix<double, Nx, Nx> Q = Eigen::Matrix<double, Nx, Nx>::Identity();
  /// Final state cost
  Eigen::Matrix<double, Nx, Nx> QT = Eigen::Matrix<double, Nx, Nx>::Identity();
  /// Running input cost
  Eigen::Matrix<double, Nu, Nu> R = Eigen::Matrix<double, Nu, Nu>::Identity();
};

namespace detail {

/**
 * @brief Compute hessian of function \f$ x \mapsto (x - xbar)^T * Q * (x - xbar)\f$.
 *
 * @param[out] hess output variable
 * @param[in] x point to calculate hessian at
 * @param[in] xbar
 * @param[in] Q quadratic weight coefficients
 * @param[in] r0 row in hess to insert result
 * @param[in] c0 col in hess to insert result
 */
template<LieGroup X, typename Qt>
  requires std::is_base_of_v<Eigen::EigenBase<Qt>, Qt>
void hessian_quad(
  Eigen::SparseMatrix<double> & out,
  const X & x,
  const X & xbar,
  const Qt & Q,
  Eigen::Index r0 = 0,
  Eigen::Index c0 = 0)
{
  const Tangent<X> e                           = rminus(x, xbar);
  const Eigen::RowVector<Scalar<X>, Dof<X>> Jq = e.transpose() * Q;

  // TODO avoid allocation here
  Eigen::SparseMatrix<double> Jrmin(Dof<X>, Dof<X>);
  Eigen::SparseMatrix<double> d2rexpi(Dof<X>, Dof<X> * Dof<X>);
  Eigen::SparseMatrix<double> Hrmin(Dof<X>, Dof<X> * Dof<X>);

  dr_expinv_sparse<X>(Jrmin, e);
  d2r_expinv_sparse<X>(d2rexpi, e);

  for (auto j = 0u; j < Dof<X>; ++j) {
    // TODO sparse-sparse product is expensive and allocates temporary
    Hrmin.middleCols(j * Dof<X>, Dof<X>) = d2rexpi.middleCols(j * Dof<X>, Dof<X>) * Jrmin;
  }

  d2r_fog(out, Jq, Q, Jrmin, Hrmin, r0, c0);
}

/**
 * @brief Wrapper for desired trajectory and its derivative.
 */
template<Time T, LieGroup X>
struct XDes
{
  T t0;
  std::function<X(T)> xdes           = [](T) -> X { return Default<X>(); };
  std::function<Tangent<X>(T)> dxdes = [](T) -> Tangent<X> { return Tangent<X>::Zero(); };

  X operator()(const double t_rel) const
  {
    const T t_abs = time_trait<T>::plus(t0, t_rel);
    return xdes(t_abs);
  };

  Tangent<X> jacobian(const double t_rel) const
  {
    const T t_abs = time_trait<T>::plus(t0, t_rel);
    return dxdes(t_abs);
  }
};

/**
 * @brief Wrapper for desired input and its derivative.
 */
template<Time T, Manifold U>
struct UDes
{
  T t0;
  std::function<U(T)> udes = [](T) -> U { return Default<U>(); };

  U operator()(const double t_rel) const
  {
    const T t_abs = time_trait<T>::plus(t0, t_rel);
    return udes(t_abs);
  };
};

/**
 * @brief MPC cost function.
 */
template<LieGroup X>
struct MPCObj
{
  static constexpr auto Nx = Dof<X>;

  // public members
  X xf_des                         = Default<X>();
  Eigen::Matrix<double, Nx, Nx> QT = Eigen::Matrix<double, Nx, Nx>::Identity();

  // private members
  Eigen::SparseMatrix<double> hess{1 + 2 * Nx + 1, 1 + 2 * Nx + 1};

  // functor members
  double operator()(const double, const X &, const X & xf, const Eigen::Vector<double, 1> & q) const
  {
    const Tangent<X> xf_err = rminus(xf, xf_des);
    return q.sum() + 0.5 * (QT * xf_err).dot(xf_err);
  }

  Eigen::RowVector<double, 1 + 2 * Nx + 1>
  jacobian(const double, const X &, const X & xf, const Eigen::Vector<double, 1> &)
  {
    Eigen::RowVector<double, 1 + 2 * Nx + 1> ret;
    ret.setZero();

    const Tangent<X> xf_err      = rminus(xf, xf_des);
    const TangentMap<X> drexpinv = dr_expinv<X>(xf_err);

    ret(1 + 2 * Nx)                  = 1;                                   // dtheta / dq
    ret.template segment<Nx>(1 + Nx) = xf_err.transpose() * QT * drexpinv;  // dtheta / dxf

    return ret;
  }

  const Eigen::SparseMatrix<double> &
  hessian(const double, const X &, const X & xf, const Eigen::Vector<double, 1> &)
  {
    set_zero(hess);

    hessian_quad(hess, xf, xf_des, QT, 1 + Nx, 1 + Nx);

    hess.makeCompressed();
    return hess;
  }
};

/**
 * @brief MPC dynamics (derivatives obtained via autodiff).
 */
template<LieGroup X, Manifold U, typename F, diff::Type DT>
struct MPCDyn
{
  static constexpr auto Nx = Dof<X>;
  static constexpr auto Nu = Dof<U>;

  // public members
  F f;

  // functor members
  Tangent<X> operator()(const double, const X & x, const U & u) const { return f(x, u); }

  Eigen::SparseMatrix<double> jac{Nx, 1 + Nx + Nu};

  const Eigen::SparseMatrix<double> & jacobian(const double, const X & x, const U & u)
  {
    set_zero(jac);

    const auto & [fval, df] = diff::dr<1, DT>(f, smooth::wrt(x, u));
    block_add(jac, 0, 1, df);

    jac.makeCompressed();
    return jac;
  }
};

/**
 * @brief MPC integrand.
 */
template<Time T, LieGroup X, Manifold U>
struct MPCIntegrand
{
  static constexpr auto Nx = Dof<X>;
  static constexpr auto Nu = Dof<U>;

  // public members
  std::shared_ptr<XDes<T, X>> xdes;
  std::shared_ptr<UDes<T, U>> udes;

  Eigen::Matrix<double, Nx, Nx> Q = Eigen::Matrix<double, Nx, Nx>::Identity();
  Eigen::Matrix<double, Nu, Nu> R = Eigen::Matrix<double, Nu, Nu>::Identity();

  // private members
  Eigen::SparseMatrix<double> hess{1 + Nx + Nu, 1 + Nx + Nu};

  // functor members
  Eigen::Vector<double, 1> operator()(const double t_loc, const X & x, const U & u) const
  {
    const Tangent<X> x_err = rminus(x, (*xdes)(t_loc));
    const Tangent<U> u_err = rminus(u, (*udes)(t_loc));

    return 0.5 * Eigen::Vector<double, 1>{(Q * x_err).dot(x_err) + (R * u_err).dot(u_err)};
  }

  Eigen::RowVector<double, 1 + Nx + Nu> jacobian(const double t_loc, const X & x, const U & u)
  {
    Eigen::RowVector<double, 1 + Nx + Nu> ret;
    ret.setZero();

    // dg / dx
    const Tangent<X> x_err      = rminus(x, (*xdes)(t_loc));
    ret.template segment<Nx>(1) = x_err.transpose() * Q * dr_expinv<X>(x_err);

    // dg / du
    const Tangent<U> u_err               = rminus(u, (*udes)(t_loc));
    ret.template segment<Dof<U>>(1 + Nx) = u_err.transpose() * R * dr_expinv<U>(u_err);

    return ret;
  }

  const Eigen::SparseMatrix<double> & hessian(const double t_loc, const X & x, const U & u)
  {
    set_zero(hess);

    hessian_quad(hess, x, (*xdes)(t_loc), Q, 1, 1);
    hessian_quad(hess, u, (*udes)(t_loc), R, 1 + Dof<X>, 1 + Dof<X>);

    hess.makeCompressed();
    return hess;
  }
};

/**
 * @brief MPC running constraints (jacobian obtained via automatic differentiation).
 */
template<LieGroup X, Manifold U, typename F, diff::Type DT>
struct MPCCR
{
  static constexpr auto Nx  = Dof<X>;
  static constexpr auto Nu  = Dof<U>;
  static constexpr auto Ncr = std::invoke_result_t<F, X, U>::SizeAtCompileTime;

  // public members
  F f;

  // private members
  Eigen::SparseMatrix<double> jac{Ncr, 1 + Nx + Nu};

  // functor members
  Eigen::Vector<double, Ncr> operator()(const double, const X & x, const U & u) const
  {
    return f(x, u);
  }

  const Eigen::SparseMatrix<double> & jacobian(const double, const X & x, const U & u)
  {
    set_zero(jac);

    const auto & [fval, df] = diff::dr<1, DT>(f, smooth::wrt(x, u));
    block_add(jac, 0, 1, df);

    jac.makeCompressed();
    return jac;
  }
};

/**
 * @brief MPC end constraints.
 */
template<LieGroup X>
struct MPCCE
{
  static constexpr auto Nx = Dof<X>;

  // public members
  X x0val = Default<X>();

  // private members
  Eigen::SparseMatrix<double> jac{Nx, 1 + 2 * Nx + 1};

  // functor members
  Tangent<X>
  operator()(const double, const X & x0, const X &, const Eigen::Vector<double, 1> &) const
  {
    return rminus(x0, x0val);
  }

  const Eigen::SparseMatrix<double> &
  jacobian(const double, const X & x0, const X &, const Eigen::Vector<double, 1> &)
  {
    set_zero(jac);

    dr_exp_sparse<X>(jac, rminus(x0, x0val), 0, 1);

    jac.makeCompressed();
    return jac;
  }
};

}  // namespace detail

/**
 * @brief Parameters for MPC
 */
struct MPCParams
{
  /**
   * @brief Minimum number of collocation points
   *
   * @note The actual number of points is ceil(K / Kmesh)
   */
  std::size_t K{10};

  /**
   * @brief MPC time horizon (seconds)
   */
  double tf{1};

  /**
   * @brief Enable warmstarting
   */
  bool warmstart{true};

  /**
   * @brief QP solvers parameters.
   */
  QPSolverParams qp{};
};

/**
 * @brief Model-Predictive Control (MPC) on Lie groups.
 *
 * @tparam T time type, must be a std::chrono::duration-like
 * @tparam X state space LieGroup type
 * @tparam U input space Manifold type
 * @tparam F callable type that represents dynamics
 * @tparam CR callable type that represents running constraints
 * @tparam DT differentiation method
 * @tparam Kmesh number of collocation points per mesh interval
 *
 * This MPC class keeps and repeatedly solves an internal OCP that is updated to track a
 * time-dependent trajectory defined through set_xudes().
 *
 * The dynamics F are linearized around the desired trajectory.
 */
template<
  Time T,
  LieGroup X,
  Manifold U,
  typename F,
  typename CR,
  diff::Type DT     = diff::Type::Default,
  std::size_t Kmesh = 4>
class MPC
{
  static constexpr auto Ncr = std::invoke_result_t<CR, X, U>::SizeAtCompileTime;

public:
  /**
   * @brief Create an MPC instance.
   *
   * @param f callable object that represents dynamics \f$ \mathrm{d}^r x_t = f(x, u) \f$ as a
   * function \f$ f : X \times U \rightarrow \mathbb{R}^{\dim \mathfrak{g}} \f$.
   * @param cr callable object that represents running constraints \f$ c_{rl} \leq c_r(x, u) \leq
   * c_{ru} \f$ as a function \f$ c_r : X \times U \rightarrow \mathbb{R}^{n_{c_r}} \f$.
   * @param crl, cru running constraints bounds
   * @param prm MPC parameters
   *
   * @note Allocates dynamic memory for work matrices and a sparse QP.
   */
  inline MPC(
    F && f,
    CR && cr,
    Eigen::Vector<double, Ncr> && crl,
    Eigen::Vector<double, Ncr> && cru,
    MPCParams && prm = MPCParams{})
      : xdes_{std::make_shared<detail::XDes<T, X>>()},
        udes_{std::make_shared<detail::UDes<T, U>>()},
        mesh_{
          (prm.K + Kmesh - 1) / Kmesh,
        },
        ocp_{
          .theta = {},
          .f     = {.f = std::forward<F>(f)},
          .g     = {.xdes = xdes_, .udes = udes_},
          .cr    = {.f = std::forward<CR>(cr)},
          .crl   = std::move(crl),
          .cru   = std::move(cru),
          .ce    = {},
          .cel   = Eigen::Vector<double, Dof<X>>::Zero(),
          .ceu   = Eigen::Vector<double, Dof<X>>::Zero(),
        },
        prm_{std::move(prm)}
  {
    detail::ocp_to_qp_allocate<DT>(qp_, work_, ocp_, mesh_);
    assert(test_ocp_derivatives(ocp_, 5, 1e-2));
  }
  /// Same as above but for lvalues
  inline MPC(
    const F & f,
    const CR & cr,
    const Eigen::Vector<double, Ncr> & crl,
    const Eigen::Vector<double, Ncr> & cru,
    const MPCParams & prm = MPCParams{})
      : MPC(
        F(f),
        CR(cr),
        Eigen::Vector<double, Ncr>(crl),
        Eigen::Vector<double, Ncr>(cru),
        MPCParams(prm))
  {}
  /// Default constructor
  inline MPC() = default;
  /// Default copy constructor
  inline MPC(const MPC &) = default;
  /// Default move constructor
  inline MPC(MPC &&) = default;
  /// Default copy assignment
  inline MPC & operator=(const MPC &) = default;
  /// Default move assignment
  inline MPC & operator=(MPC &&) = default;
  /// Default destructor
  inline ~MPC() = default;

  /**
   * @brief Solve MPC problem and return input.
   *
   * @param[in] t current time
   * @param[in] x current state
   * @param[out] u_traj (optional) return MPC input solution \f$ [\mu_0, \mu_1, \ldots, \mu_{K - 1}]
   * \f$
   * @param[out] x_traj (optional) return MPC state solution \f$ [x_0, x_1, \ldots, x_K] \f$
   *
   * @return {u, code}
   */
  inline std::pair<U, QPSolutionStatus> operator()(
    const T & t,
    const X & x,
    std::optional<std::reference_wrapper<std::vector<U>>> u_traj = std::nullopt,
    std::optional<std::reference_wrapper<std::vector<X>>> x_traj = std::nullopt)
  {
    static constexpr auto Nx = Dof<X>;
    static constexpr auto Nu = Dof<U>;
    const auto N             = mesh_.N_colloc();

    const auto xvar_L = Nx * (N + 1);

    const auto xvar_B = 0u;
    const auto uvar_B = xvar_L;

    // update problem
    xdes_->t0         = t;
    udes_->t0         = t;
    ocp_.ce.x0val     = x;
    ocp_.theta.xf_des = (*xdes_)(prm_.tf);

    // transcribe to QP
    ocp_to_qp_update<diff::Type::Analytic>(qp_, work_, ocp_, mesh_, prm_.tf, *xdes_, *udes_);
    qp_.A.makeCompressed();
    qp_.P.makeCompressed();

    // solve QP
    auto sol = solve_qp(qp_, prm_.qp, warmstart_);

    // output solution trajectories
    if (u_traj.has_value()) {
      u_traj.value().get().resize(N);
      for (const auto & [i, trel] : zip(std::views::iota(0u, N), mesh_.all_nodes())) {
        u_traj.value().get()[i] = (*udes_)(trel) + sol.primal.template segment<Nu>(uvar_B + i * Nu);
      }
    }
    if (x_traj.has_value()) {
      x_traj.value().get().resize(N + 1);
      for (const auto & [i, trel] : zip(std::views::iota(0u, N + 1), mesh_.all_nodes())) {
        x_traj.value().get()[i] = (*xdes_)(trel) + sol.primal.template segment<Nx>(xvar_B + i * Nx);
      }
    }

    // save solution for warmstart
    if (prm_.warmstart) {
      if (
        // clang-format off
        sol.code == QPSolutionStatus::Optimal
        || sol.code == QPSolutionStatus::MaxTime
        || sol.code == QPSolutionStatus::MaxIterations
        // clang-format n
      ) {
        warmstart_ = sol;
      }
    }

    return {rplus((*udes_)(0), sol.primal.template segment<Nu>(Nx * (N + 1))), sol.code};
  }

  /**
   * @brief Set the desired input trajectory (absolute time)
   */
  inline void set_udes(std::function<U(T)> && u_des)
  {
    udes_->udes = std::move(u_des);
  }

  /**
   * @brief Set the desired input trajectory (absolute time, rvalue version)
   */
  inline void set_udes(const std::function<U(T)> & u_des) { set_udes(std::function<U(T)>(u_des)); }

  /**
   * @brief Set the desired input trajectory (relative time).
   *
   * @param f function double -> U<double> s.t. u_des(t) = f(t - t0)
   * @param t0 absolute zero time for the desired trajectory
   */
  template<typename Fun>
    requires(std::is_same_v<std::invoke_result_t<Fun, Scalar<U>>, U>)
  inline void set_udes_rel(Fun && f, T t0 = T(0))
  {
    set_udes([t0 = t0, f = std::forward<Fun>(f)](T t) -> U {
      const double t_rel = time_trait<T>::minus(t, t0);
      return std::invoke(f, t_rel);
    });
  }

  /**
   * @brief Set the desired state trajectory and velocity (absolute time)
   */
  inline void set_xdes(std::function<X(T)> && x_des, std::function<Tangent<X>(T)> && dx_des)
  {
    xdes_->xdes = std::move(x_des);
    xdes_->dxdes = std::move(dx_des);
  }

  /**
   * @brief Set the desired state trajectory (absolute time, rvalue version)
   */
  inline void set_xdes(const std::function<X(T)> & x_des, const std::function<Tangent<X>(T)> & dx_des)
  {
    set_xdes(std::function<X(T)>(x_des), std::function<Tangent<X>(T)>(dx_des));
  }

  /**
   * @brief Set the desired state trajectry (relative time, automatic differentiation).
   *
   * Instead of providing {x(t), dx(t)} of the desired trajectory, this function differentiates a
   * function x(t) so that the derivative does not need to be specified.
   *
   * @param f function s.t. desired trajectory is x(t) = f(t - t0)
   * @param t0 absolute zero time for the desired trajectory
   */
  template<typename Fun>
    requires(std::is_same_v<std::invoke_result_t<Fun, Scalar<X>>, X>)
  inline void set_xdes_rel(Fun && f, T t0 = T(0))
  {
    std::function<X(T)> x_des = [t0 = t0, f = f](T t) -> X {
      const double t_rel = time_trait<T>::minus(t, t0);
      return std::invoke(f, t_rel);
    };

    std::function<Tangent<X>(T)> dx_des = [t0 = t0, f = f](T t) -> Tangent<X> {
      const double t_rel = time_trait<T>::minus(t, t0);
      return std::get<1>(diff::dr<1, DT>(f, wrt(t_rel)));
    };

    set_xdes(std::move(x_des), std::move(dx_des));
  }

  /**
   * @brief Update MPC weights
   */
  inline void set_weights(const MPCWeights<X, U> & weights)
  {
    ocp_.g.R      = weights.R;
    ocp_.g.Q      = weights.Q;
    ocp_.theta.QT = weights.QT;
  }

  /**
   * @brief Reset initial guess for next iteration to zero.
   */
  inline void reset_warmstart() { warmstart_ = {}; }

private:
  // linearization
  std::shared_ptr<detail::XDes<T, X>> xdes_;
  std::shared_ptr<detail::UDes<T, U>> udes_;

  // collocation mesh
  Mesh<Kmesh, Kmesh> mesh_{};

  // problem description
  using ocp_t = OCP<X, U, detail::MPCObj<X>, detail::MPCDyn<X, U, F, DT>, detail::MPCIntegrand<T, X, U>, detail::MPCCR<X, U, CR, DT>, detail::MPCCE<X>>;
  ocp_t ocp_;

  // parameters
  MPCParams prm_{};

  // pre-allocated QP matrices
  detail::OcpToQpWorkmemory work_;
  QuadraticProgramSparse<double> qp_;

  // store last solution for warmstarting
  std::optional<QPSolution<-1, -1, double>> warmstart_{};
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__MPC_HPP_
