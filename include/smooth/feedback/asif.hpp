// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

#include <cassert>
#include <utility>

#include "asif_func.hpp"
#include "qp_solver.hpp"
#include "time.hpp"

namespace smooth::feedback {

/**
 * @brief ASIFilter filter parameters
 */
template<Manifold U>
struct ASIFilterParams
{
  /// Time horizon
  double T{1};
  /// Number of barrier constraints (must agree with output dimensionality of h)
  std::size_t nh{1};
  /// Weights on desired input
  Eigen::Matrix<double, Dof<U>, 1> u_weight{Eigen::Matrix<double, Dof<U>, 1>::Ones()};
  /// Input bounds
  ManifoldBounds<U> ulim{};
  /// ASIFilter algorithm parameters
  ASIFtoQPParams asif{};
  /// solve_qp() parameters
  QPSolverParams qp{};
};

/**
 * @brief ASI Filter
 *
 * Thin wrapper around asif_to_qp() and solve_qp() that keeps track of the most
 * recent solution for warmstarting, and facilitates working with time-varying
 * problems.
 */
template<LieGroup G, Manifold U, typename Dyn, diff::Type DT = diff::Type::Default>
class ASIFilter
{
public:
  /**
   * @brief Construct an ASI filter
   *
   * @param f dynamics as function \f$ \mathbb{R} \times \mathbb{G} \times \mathbb{U} \rightarrow
   * \mathbb{R}^{\dim \mathbb{G}} \f$
   * @param prm filter parameters
   */
  ASIFilter(const Dyn & f, const ASIFilterParams<U> & prm = ASIFilterParams<U>{})
      : ASIFilter(Dyn(f), ASIFilterParams<U>(prm))
  {}

  /**
   * @brief Construct an ASI filter (rvalue version).
   */
  ASIFilter(Dyn && f, ASIFilterParams<U> && prm = ASIFilterParams<U>{}) : f_(std::move(f)), prm_(std::move(prm))
  {
    const int nu_ineq = prm_.ulim.A.rows();
    asif_to_qp_allocate<G, U>(qp_, prm_.asif.K, nu_ineq, prm_.nh);
  }

  /**
   * @brief Filter an input
   *
   * @param t current global time
   * @param g current state
   * @param u_des nominal (desired) control input
   * @param h safety set definition as function \f$ \mathbb{R} \times \mathbb{G} \rightarrow
   * \mathbb{R}^{n_h} \f$
   * @param bu backup controller as function \f$ \mathbb{R} \times \mathbb{G} \rightarrow
   * \mathbb{U} \f$
   *
   * @note h and bu are defined w.r.t the current time. That is, the safety set at global time
   * \f$\tau\f$ is \f$ S(\tau) = \{ h(\tau - t) \geq 0 \} \f$, and the backup controll action is \f$
   * u(\tau, x) = bu(\tau - t, x ) \f$.
   *
   * @returns {u, code}: safe control input and QP solver code
   */
  std::pair<U, QPSolutionStatus> operator()(const G & g, const U & u_des, auto && h, auto && bu)
  {
    using std::chrono::duration, std::chrono::duration_cast, std::chrono::nanoseconds;

    assert((prm_.nh == std::invoke_result_t<decltype(h), Scalar<G>, G>::RowsAtCompileTime));

    ASIFProblem<G, U> pbm{
      .T     = prm_.T,
      .x0    = g,
      .u_des = u_des,
      .W_u   = prm_.u_weight,
      .ulim  = prm_.ulim,
    };

    asif_to_qp_update<G, U, DT>(qp_, pbm, prm_.asif, f_, std::forward<decltype(h)>(h), std::forward<decltype(bu)>(bu));
    auto sol = feedback::solve_qp(qp_, prm_.qp, warmstart_);

    if (sol.code == QPSolutionStatus::Optimal) { warmstart_ = sol; }

    return {rplus(u_des, sol.primal.template head<Dof<U>>()), sol.code};
  }

private:
  Dyn f_;

  QuadraticProgram<-1, -1, double> qp_;
  ASIFilterParams<U> prm_;
  std::optional<QPSolution<-1, -1, double>> warmstart_;
};

}  // namespace smooth::feedback
