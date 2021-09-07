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

#ifndef SMOOTH__FEEDBACK__ASIF_HPP_
#define SMOOTH__FEEDBACK__ASIF_HPP_

#include "asif_func.hpp"

namespace smooth::feedback {

/**
 * @brief ASIFilter filter parameters
 */
template<Manifold U>
struct ASIFilterParams
{
  /// Time horizon
  double T{1};
  /// Weights on desired input
  Eigen::Matrix<double, Dof<U>, 1> u_weight{Eigen::Matrix<double, Dof<U>, 1>::Ones()};
  /// Input bounds
  ManifoldBounds<U> ulim{};
  /// ASIFilter algorithm parameters
  ASIFtoQPParams asif{};
  /// QP solver parameters
  QPSolverParams qp{};
};

/**
 * @brief ASI Filter
 *
 * Thin wrapper around asif_to_qp() and solve_qp() that keeps track of the most
 * recent solution for warmstarting, and facilitates working with time-varying
 * problems.
 */
template<LieGroup G,
  Manifold U,
  typename Dyn,
  typename SS,
  typename BU,
  diff::Type DT = diff::Type::DEFAULT>
class ASIFilter
{
public:
  /**
   * @brief Construct an ASI filter
   *
   * @param f dynamics as function \f$ \mathbb{R} \times \mathbb{G} \times \mathbb{U} \rightarrow
   * \mathbb{R}^{\dim \mathbb{G}} \f$
   * @param h safety set definition as function \f$ \mathbb{G} \rightarrow \mathbb{R}^{n_h} \f$
   * @param bu backup controller as function \f$ \mathbb{R} \times \mathbb{G} \rightarrow \mathbb{U}
   * \f$
   * @param prm filter parameters
   *
   * @note These functions are defined in global time. As opposed to MPC the time
   * variable must be a \p double since differentiation of bu w.r.t. t is required.
   */
  ASIFilter(const Dyn & f,
    const SS & h,
    const BU & bu,
    const ASIFilterParams<U> & prm = ASIFilterParams<U>{})
      : ASIFilter(Dyn(f), SS(h), BU(bu), ASIFilterParams<U>(prm))
  {}

  /**
   * @brief Construct an ASI filter (rvalue version).
   */
  ASIFilter(Dyn && f, SS && h, BU && bu, ASIFilterParams<U> && prm = ASIFilterParams<U>{})
      : f_(std::move(f)), h_(std::move(h)), bu_(std::move(bu)), prm_(std::move(prm))
  {
    static constexpr int nh = std::invoke_result_t<SS, double, G>::SizeAtCompileTime;
    const int nu_ineq       = prm_.ulim.A.rows();
    qp_                     = asif_to_qp_allocate<G, U>(prm_.asif.K, nu_ineq, nh);
  }

  /**
   * @brief Filter an input
   *
   * @param t current global time
   * @param g current state
   * @param u_des nominal (desired) control input
   *
   * @returns {u, code}: safe control input and QP solver code
   */
  std::pair<U, QPSolutionStatus> operator()(double t, const G & g, const U & u_des)
  {
    using std::chrono::duration, std::chrono::nanoseconds;

    auto f = [this, &t]<typename T>(T t_loc, const CastT<T, G> & vx, const CastT<T, U> & vu) {
      return f_(T(t) + t_loc, vx, vu);
    };
    auto h = [this, &t]<typename T>(
               T t_loc, const CastT<T, G> & vx) { return h_(T(t) + t_loc, vx); };
    auto bu = [this, &t]<typename T>(
                T t_loc, const CastT<T, G> & vx) { return bu_(T(t) + t_loc, vx); };

    ASIFProblem<G, U> pbm{
      .T     = prm_.T,
      .x0    = g,
      .u_des = u_des,
      .W_u   = prm_.u_weight,
      .ulim  = ulim_,
    };

    asif_to_qp_fill<G, U, decltype(f), decltype(h), decltype(bu), DT>(
      pbm, prm_.asif, std::move(f), std::move(h), std::move(bu), qp_);
    auto sol = feedback::solve_qp(qp_, prm_.qp, warmstart_);

    if (sol.code == QPSolutionStatus::Optimal) { warmstart_ = sol; }

    return {rplus(u_des, sol.primal.template head<Dof<U>>()), sol.code};
  }

private:
  Dyn f_;
  SS h_;
  BU bu_;

  QuadraticProgram<-1, -1, double> qp_;
  ManifoldBounds<U> ulim_;
  ASIFilterParams<U> prm_;
  std::optional<QPSolution<-1, -1, double>> warmstart_;
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__ASIF_HPP_
