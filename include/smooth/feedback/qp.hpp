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
#include <Eigen/LU>
#include <Eigen/Sparse>

#include <limits>

namespace smooth::feedback {

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
template<Eigen::Index Nx, Eigen::Index Ny>
struct QuadraticProgram
{
  /// Quadratic cost
  Eigen::Matrix<double, Nx, Nx> P;
  /// Linear cost
  Eigen::Matrix<double, Nx, 1> q;

  /// Inequality matrix
  Eigen::Matrix<double, Ny, Nx> A;
  /// Inequality lower bound
  Eigen::Matrix<double, Ny, 1> l;
  /// Inequality upper bound
  Eigen::Matrix<double, Ny, 1> u;
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

enum class ExitCode : int { Optimal, PrimalInfeasible, DualInfeasible, MaxIterations, Unknown };

template<Eigen::Index Nx, Eigen::Index Ny>
struct Solution
{
  ExitCode code = ExitCode::Unknown;
  Eigen::Matrix<double, Nx, 1> primal;
  Eigen::Matrix<double, Ny, 1> dual;
};

struct SolverParams
{
  double rho   = 0.1;
  double sigma = 1e-6;
  double alpha = 1.6;

  double eps_primal_inf = 1e-4;
  double eps_dual_inf   = 1e-4;
  double eps_abs        = 1e-3;
  double eps_rel        = 1e-3;

  uint64_t max_iter = std::numeric_limits<uint64_t>::max();
};

template<Eigen::Index Nx, Eigen::Index Ny>
Solution<Nx, Ny> solveQP(const QuadraticProgram<Nx, Ny> & pbm, const SolverParams & prm)
{
  static constexpr auto Inf = Eigen::Infinity;
  using RN                  = Eigen::Matrix<double, Nx, 1>;
  using RM                  = Eigen::Matrix<double, Ny, 1>;

  // static sizes
  static constexpr Eigen::Index Np = (Nx == -1 || Ny == -1) ? Eigen::Index(-1) : Nx + Ny;

  // dynamic sizes
  const Eigen::Index n  = pbm.A.cols();
  const Eigen::Index m  = pbm.A.rows();
  const Eigen::Index np = n + m;

  // matrix factorization
  Eigen::Matrix<double, Np, Np> H(np, np);
  H.template topLeftCorner<Nx, Nx>(n, n) = pbm.P;
  H.template topLeftCorner<Nx, Nx>(n, n) += RN::Constant(n, prm.sigma).asDiagonal();
  H.template topRightCorner<Nx, Ny>(n, m)    = pbm.A.transpose();
  H.template bottomLeftCorner<Ny, Nx>(m, n)  = pbm.A;  // TODO ldlt only reads upper triangle
  H.template bottomRightCorner<Ny, Ny>(m, m) = RM::Constant(m, -1. / prm.rho).asDiagonal();

  // TODO need ldlt decomposition for indefinite matrices
  Eigen::PartialPivLU<decltype(H)> lu(H);

  // initialization
  // TODO what's a good strategy here?
  RN x = RN::Zero(n);
  RM z = RM::Zero(m);
  RM y = RM::Zero(m);

  for (auto i = 0u; i < prm.max_iter; ++i) {

    // ADMM ITERATION

    Eigen::Matrix<double, Np, 1> h(np);
    h.template head<Nx>(n) = prm.sigma * x - pbm.q;
    h.template tail<Ny>(m) = z - y / prm.rho;

    // solve linear system H p = h
    const Eigen::Matrix<double, Np, 1> p = lu.solve(h);

    const RM z_tilde = z + (p.template tail<Ny>(m) - y) / prm.rho;
    const RN x_next  = prm.alpha * p.template head<Nx>(n) + (1. - prm.alpha) * x;
    const RM z_next =
      (prm.alpha * z_tilde + (1. - prm.alpha) * z + y / prm.rho).cwiseMax(pbm.l).cwiseMin(pbm.u);
    const RM y_next = y + prm.rho * (prm.alpha * z_tilde + (1. - prm.alpha) * z - z_next);

    const RN dx = x_next - x, dy = y_next - y;
    x = x_next, y = y_next, z = z_next;

    const double r_primal_norm = (pbm.A * x - z).template lpNorm<Inf>();
    const double r_dual_norm   = (pbm.P * x + pbm.q + pbm.A.transpose() * y).template lpNorm<Inf>();
    double dx_norm = dx.template lpNorm<Inf>(), dy_norm = dy.template lpNorm<Inf>();

    // OPTIMALITY

    // clang-format off
    double eps_primal = prm.eps_abs + prm.eps_rel * std::max(
      (pbm.A * x).template lpNorm<Inf>(),
      z.template lpNorm<Inf>()
    );
    double eps_dual = prm.eps_abs + prm.eps_abs * std::max({
      (pbm.P * x).template lpNorm<Inf>(),
      (pbm.A.transpose() * y).template lpNorm<Inf>(),
      pbm.q.template lpNorm<Inf>()
    });
    // clang-format on

    if (r_primal_norm <= eps_primal && r_dual_norm <= eps_dual) {
      return {.code = ExitCode::Optimal, .primal = x, .dual = y};
    }

    // PRIMAL INFEASIBILITY

    bool primal_infeasible =
      ((pbm.A.transpose() * dy).template lpNorm<Inf>() <= prm.eps_primal_inf * dy_norm)
      && (pbm.u.dot(dy.cwiseMax(0.)) + pbm.l.dot(dy.cwiseMin(0.)) <= prm.eps_primal_inf * dy_norm);

    if (primal_infeasible) {
      return {.code = ExitCode::PrimalInfeasible, .primal = {}, .dual = {}};
    }

    // DUAL INFEASIBILITY

    bool dual_infeasible = ((pbm.P * dx).template lpNorm<Inf>() <= prm.eps_dual_inf * dx_norm)
                        && (pbm.q.dot(dx) <= prm.eps_dual_inf * dx_norm);
    const RM Adx = pbm.A * dx;
    for (auto i = 0u; i < m && dual_infeasible; ++i) {
      if (pbm.u(i) == std::numeric_limits<double>::infinity()) {
        dual_infeasible &= (Adx(i) >= -prm.eps_dual_inf * dx_norm);
      } else if (pbm.u(i) == -std::numeric_limits<double>::infinity()) {
        dual_infeasible &= (Adx(i) <= prm.eps_dual_inf * dx_norm);
      } else {
        dual_infeasible &= ((-prm.eps_dual_inf <= Adx(i)) && (Adx(i) <= prm.eps_dual_inf));
      }
    }

    if (dual_infeasible) { return {.code = ExitCode::DualInfeasible, .primal = {}, .dual = {}}; }
  }

  return {.code = ExitCode::MaxIterations, .primal = x, .dual = y};
}

Solution<-1, -1> solve(const QuadraticProgramSparse &, const SolverParams &) { return {}; }

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__QP_HPP_
