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

#ifndef SMOOTH__FEEDBACK__QP_HPP_
#define SMOOTH__FEEDBACK__QP_HPP_

/**
 * @file
 * @brief Quadratic Programming.
 */

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/LU>

#include <limits>

namespace smooth::feedback
{

/**
 * @brief Quadratic program definition.
 *
 * The quadratic program is on the form
 * \f[
 * \begin{cases}
 *  \min_{x} & \frac{1}{2} x^T P x + q^T x, \\
 *  \text{s.t.} & l \leq A x \leq u.
 * \end{cases}
 * \f]
 */
template<Eigen::Index Nvar, Eigen::Index Ncon>
struct QuadraticProgram
{
  /// Quadratic cost
  Eigen::Matrix<double, Nvar, Nvar> P = Eigen::Matrix<double, Nvar, Nvar>::Zero();
  /// Linear cost
  Eigen::Matrix<double, Nvar, 1> q = Eigen::Matrix<double, Nvar, 1>::Zero();

  /// Inequality matrix
  Eigen::Matrix<double, Ncon, Nvar> A = Eigen::Matrix<double, Ncon, Nvar>::Zero();
  /// Inequality lower bound
  Eigen::Matrix<double, Ncon, 1> l = Eigen::Matrix<double, Ncon, 1>::Zero();
  /// Inequality upper bound
  Eigen::Matrix<double, Ncon, 1> u = Eigen::Matrix<double, Ncon, 1>::Zero();
};

struct QuadraticProgramSparse
{
  /// Quadratic cost
  Eigen::SparseMatrix<double> P;
  /// Linear cost
  Eigen::SparseMatrix<double> q;

  /// Inequality matrix
  Eigen::SparseMatrix<double> A;
  /// Inequality lower bound
  Eigen::SparseMatrix<double> l;
  /// Inequality upper bound
  Eigen::SparseMatrix<double> u;
};

enum class ExitCode : int
{
  Optimal,
  PrimalInfeasible,
  DualInfeasible,
  MaxIterations,
  Unknown
};

template<Eigen::Index Nvar, Eigen::Index Ncon>
struct Solution
{
  ExitCode code = ExitCode::Unknown;
  Eigen::Matrix<double, Nvar, 1> primal;
  Eigen::Matrix<double, Ncon, 1> dual;

};

struct SolverParams
{
  double rho = 0.1;
  double sigma = 1e-6;
  double alpha = 1.6;

  double eps_primal_inf = 1e-4;
  double eps_dual_inf = 1e-4;
  double eps_abs = 1e-3;
  double eps_rel = 1e-3;

  uint64_t max_iter = std::numeric_limits<uint64_t>::max();
};

template<Eigen::Index Nvar, Eigen::Index Ncon>
Solution<Nvar, Ncon> solveQP(
  const QuadraticProgram<Nvar, Ncon> & problem,
  const SolverParams & params)
{
  using Rvar = Eigen::Matrix<double, Nvar, 1>;
  using Rcon = Eigen::Matrix<double, Ncon, 1>;

  Eigen::Index nvar = problem.A.cols();
  Eigen::Index ncon = problem.A.rows();
  // initial
  Rvar x = Rvar::Zero(nvar);
  Rcon z = Rcon::Zero(ncon);
  Rcon y = Rcon::Zero(ncon);

  static constexpr Eigen::Index Nh = (Nvar == -1 || Ncon == -1) ? Eigen::Index(-1) : Nvar + Ncon;
  const Eigen::Index nh = nvar + ncon;

  Eigen::Matrix<double, Nh, Nh> H(nh, nh);
  Eigen::Matrix<double, Nh, 1> h(nh);

  H.template topLeftCorner<Nvar, Nvar>(
    nvar,
    nvar) = problem.P;
  H.template topLeftCorner<Nvar, Nvar>(nvar, nvar) +=
    Rvar::Constant(nvar, params.sigma).asDiagonal();
  H.template topRightCorner<Nvar, Ncon>(nvar, ncon) = problem.A.transpose();
  H.template bottomLeftCorner<Ncon, Nvar>(ncon, nvar) = problem.A;
  H.template bottomRightCorner<Ncon, Ncon>(ncon, ncon) = Rcon::Constant(
    ncon, -1. / params.rho).asDiagonal();

  Eigen::PartialPivLU<decltype(H)> lu(H);

  for (auto i = 0u; i < params.max_iter; ++i) {
    // solve linear system and update
    h.template head<Nvar>(nvar) = params.sigma * x - problem.q;
    h.template tail<Ncon>(ncon) = z - y / params.rho;

    Eigen::Matrix<double, Nvar + Ncon, 1> o = lu.solve(h);

    Rvar x_tilde = o.template head<Nvar>(nvar);
    Rcon nu = o.template tail<Ncon>(ncon);

    Rcon z_tilde = z + (nu - y) / params.rho;
    Rvar x_next = params.alpha * x_tilde + (1. - params.alpha) * x;
    Rcon z_next =
      (params.alpha * z_tilde + (1 - params.alpha) * z + y /
      params.rho).cwiseMax(problem.l).cwiseMin(problem.u);
    Rcon y_next = y + params.rho * (params.alpha * z_tilde + (1 - params.alpha) * z - z_next);

    Rvar dx = x_next - x;
    Rcon dy = y_next - y;
    Rcon dz = z_next - z;

    x = x_next;
    y = y_next;
    z = z_next;

    // compute residuals
    Rcon r_primal = problem.A * x - z;
    Rvar r_dual = problem.P * x + problem.q + problem.A.transpose() * y;
    // compute termination criteria
    double eps_primal = params.eps_abs + params.eps_rel * std::max(
      (problem.A * x).template lpNorm<Eigen::Infinity>(), z.template lpNorm<Eigen::Infinity>());
    double eps_dual = params.eps_abs + params.eps_abs * std::max(
      std::max(
        (problem.P * x).template lpNorm<Eigen::Infinity>(),
        (problem.A.transpose() * y).template lpNorm<Eigen::Infinity>()),
      problem.q.template lpNorm<Eigen::Infinity>());

    bool primal_reached = r_primal.template lpNorm<Eigen::Infinity>() <= eps_primal;
    bool dual_reached = r_dual.template lpNorm<Eigen::Infinity>() <= eps_dual;
    bool primal_infeasible = ((problem.A.transpose() * dy).template lpNorm<Eigen::Infinity>() <=
      params.eps_primal_inf * y.template lpNorm<Eigen::Infinity>()) &&
      ((problem.u.transpose() * dy.cwiseMax(0.) + problem.l.transpose() * dy.cwiseMin(0.))[0] <=
      params.eps_primal_inf * dy.template lpNorm<Eigen::Infinity>());

    bool dual_infeasible = true;
    Rcon Adx = problem.A * dx;
    for (auto i = 0u; i < ncon; ++i) {
      if (problem.u[i] == std::numeric_limits<double>::infinity()) {
        dual_infeasible &= (Adx[i] >= -params.eps_dual_inf);
      } else if (problem.u[i] == -std::numeric_limits<double>::infinity()) {
        dual_infeasible &= (Adx[i] <= params.eps_dual_inf);
      } else {
        dual_infeasible &= ((-params.eps_dual_inf <= Adx[i]) && (Adx[i] <= params.eps_dual_inf));
      }
    }
    dual_infeasible &=
      ((problem.P * dx).template lpNorm<Eigen::Infinity>() <=
      params.eps_dual_inf * dx.template lpNorm<Eigen::Infinity>()) &&
      (problem.q.transpose() * dx <= params.eps_dual_inf * dx.template lpNorm<Eigen::Infinity>());

    if (primal_reached && dual_reached) {
      return {.code = ExitCode::Optimal, .primal = x, .dual = y};
    }

    if (primal_infeasible) {
      return {.code = ExitCode::PrimalInfeasible};
    }

    if (dual_infeasible) {
      return {.code = ExitCode::DualInfeasible};
    }
  }

  return {.code = ExitCode::MaxIterations, .primal = x, .dual = y};
}

Solution<-1, -1> solve(const QuadraticProgramSparse & problem, const SolverParams & params)
{

}
}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__QP_HPP_
