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
 * @tparam M number of constraints
 * @tparam N number of variables
 *
 * The quadratic program is on the form
 * \f[
 * \begin{cases}
 *  \min_{x} & \frac{1}{2} x^T P x + q^T x, \\
 *  \text{s.t.} & l \leq A x \leq u.
 * \end{cases}
 * \f]
 */
template<Eigen::Index M, Eigen::Index N>
struct QuadraticProgram
{
  /// Quadratic cost
  Eigen::Matrix<double, N, N> P;
  /// Linear cost
  Eigen::Matrix<double, N, 1> q;

  /// Inequality matrix
  Eigen::Matrix<double, M, N> A;
  /// Inequality lower bound
  Eigen::Matrix<double, M, 1> l;
  /// Inequality upper bound
  Eigen::Matrix<double, M, 1> u;
};

/**
 * @brief Sparse quadratic program definition.
 *
 * The quadratic program is on the form
 * \f[
 * \begin{cases}
 *  \min_{x} & \frac{1}{2} x^T P x + q^T x, \\
 *  \text{s.t.} & l \leq A x \leq u.
 * \end{cases}
 * \f]
 */
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

/// Solver exit codes
enum class ExitCode : int { Optimal, PrimalInfeasible, DualInfeasible, MaxIterations, Unknown };

/// Solver solution
template<Eigen::Index M, Eigen::Index N>
struct Solution
{
  /// Exit code
  ExitCode code = ExitCode::Unknown;
  /// Primal vector
  Eigen::Matrix<double, N, 1> primal;
  /// Dual vector
  Eigen::Matrix<double, M, 1> dual;
};

/**
 * @brief Options for solveQP
 */
struct SolverParams
{
  /// relaxation parameter
  double alpha = 1.6;
  /// first dul step size
  double rho = 0.1;
  /// second dual step length
  double sigma = 1e-6;

  /// absolute threshold for convergence
  double eps_abs = 1e-3;
  /// relative threshold for convergence
  double eps_rel = 1e-3;
  /// threshold for primal infeasibility
  double eps_primal_inf = 1e-4;
  /// threshold for dual infeasibility
  double eps_dual_inf = 1e-4;

  /// max number of iterations
  uint64_t max_iter = std::numeric_limits<uint64_t>::max();

  /// iterations between checking stopping criterion
  uint64_t stop_check_iter = 10;
};

/**
 * @brief Solve a quadratic program using the operator splitting approach.
 *
 * @tparam M number of constraints
 * @tparam N number of variables
 *
 * @param pbm problem formulation
 * @param prm solver options
 * @return Problem solution as Solution<M, N>
 *
 * @note dynamic problem sizes (`M == -1 || N == -1`) are supported
 *
 * This is a third-party implementation of the algorithm described in the following paper:
 * * Stellato, B., Banjac, G., Goulart, P. et al.
 * **OSQP: an operator splitting solver for quadratic programs.**
 * *Math. Prog. Comp.* 12, 637–672 (2020).
 * https://doi.org/10.1007/s12532-020-00179-2
 *
 * For the official C implementation, see https://osqp.org/.
 */
template<Eigen::Index M, Eigen::Index N>
Solution<M, N> solveQP(const QuadraticProgram<M, N> & pbm, const SolverParams & prm)
{
  static constexpr auto Inf = Eigen::Infinity;

  // static sizes
  static constexpr Eigen::Index K = (N == -1 || M == -1) ? Eigen::Index(-1) : N + M;
  using Rn                         = Eigen::Matrix<double, N, 1>;
  using Rm                         = Eigen::Matrix<double, M, 1>;
  using Rk                         = Eigen::Matrix<double, K, 1>;

  // dynamic sizes
  const Eigen::Index n = pbm.A.cols(), m = pbm.A.rows(), k = n + m;

  // check that feasible set is nonempty
  if ((pbm.u - pbm.l).minCoeff() < 0. || pbm.l.maxCoeff() == std::numeric_limits<double>::infinity()
      || pbm.u.minCoeff() == -std::numeric_limits<double>::infinity()) {
    return {.code = ExitCode::PrimalInfeasible, .primal = {}, .dual = {}};
  }

  // TODO problem scaling / conditioning

  // matrix factorization
  Eigen::Matrix<double, K, K> H(k, k);
  H.template topLeftCorner<N, N>(n, n) = pbm.P;
  H.template topLeftCorner<N, N>(n, n) += Rn::Constant(n, prm.sigma).asDiagonal();
  H.template topRightCorner<N, M>(n, m)    = pbm.A.transpose();
  H.template bottomLeftCorner<M, N>(m, n)  = pbm.A;  // TODO ldlt only reads upper triangle
  H.template bottomRightCorner<M, M>(m, m) = Rm::Constant(m, -1. / prm.rho).asDiagonal();

  // TODO need ldlt decomposition for indefinite matrices
  Eigen::PartialPivLU<decltype(H)> lu(H);

  // initialization
  // TODO what's a good strategy here?
  Rn x = Rn::Zero(n);
  Rm z = Rm::Zero(m);
  Rm y = Rm::Zero(m);

  for (auto i = 0u; i != prm.max_iter; ++i) {

    // ADMM ITERATION

    // solve linear system H p = h
    const Rk h = (Rk(k) << prm.sigma * x - pbm.q, z - y / prm.rho).finished();
    const Rk p = lu.solve(h);

    const Rm z_tilde  = z + (p.tail(m) - y) / prm.rho;
    const Rn x_next   = prm.alpha * p.head(n) + (1. - prm.alpha) * x;
    const Rm z_interp = prm.alpha * z_tilde + (1. - prm.alpha) * z;
    const Rm z_next   = (z_interp + y / prm.rho).cwiseMax(pbm.l).cwiseMin(pbm.u);
    const Rm y_next   = y + prm.rho * (z_interp - z_next);

    // CHECK STOPPING CRITERIA

    if (i % prm.stop_check_iter == prm.stop_check_iter - 1) {
      const Rn dx                = x_next - x;
      const Rm dy                = y_next - y;
      const Rn Px                = pbm.P * x;
      const Rm Ax                = pbm.A * x;
      const Rn Aty               = pbm.A.transpose() * y;
      const double r_primal_norm = (Ax - z).template lpNorm<Inf>();
      const double r_dual_norm   = (Px + pbm.q + Aty).template lpNorm<Inf>();
      const double dx_norm       = dx.template lpNorm<Inf>();
      const double dy_norm       = dy.template lpNorm<Inf>();

      // OPTIMALITY

      // clang-format off
      const double eps_primal = prm.eps_abs + prm.eps_rel * std::max<double>(
        Ax.template lpNorm<Inf>(),
        z.template lpNorm<Inf>()
      );
      const double eps_dual = prm.eps_abs + prm.eps_abs * std::max<double>({
        Px.template lpNorm<Inf>(),
        Aty.template lpNorm<Inf>(),
        pbm.q.template lpNorm<Inf>()
      });
      // clang-format on

      if (r_primal_norm <= eps_primal && r_dual_norm <= eps_dual) {
        return {.code = ExitCode::Optimal, .primal = std::move(x), .dual = std::move(y)};
      }

      // PRIMAL INFEASIBILITY

      const double At_dy_norm = (pbm.A.transpose() * dy).template lpNorm<Inf>();
      double u_dyp_plus_l_dyn = 0;
      for (auto i = 0u; i != m; ++i) {
        if (pbm.u(i) != std::numeric_limits<double>::infinity()) {
          u_dyp_plus_l_dyn += pbm.u(i) * std::max<double>(0, dy(i));
        } else if (dy(i) > prm.eps_primal_inf * dy_norm) {
          // contributes +inf to sum --> no certificate
          u_dyp_plus_l_dyn = std::numeric_limits<double>::infinity();
          break;
        }
        if (pbm.l(i) != -std::numeric_limits<double>::infinity()) {
          u_dyp_plus_l_dyn += pbm.l(i) * std::min<double>(0, dy(i));
        } else if (dy(i) < -prm.eps_primal_inf * dy_norm) {
          // contributes +inf to sum --> no certificate
          u_dyp_plus_l_dyn = std::numeric_limits<double>::infinity();
          break;
        }
      }

      if (std::max<double>(At_dy_norm, u_dyp_plus_l_dyn) < prm.eps_primal_inf * dy_norm) {
        return {.code = ExitCode::PrimalInfeasible, .primal = {}, .dual = {}};
      }

      // DUAL INFEASIBILITY

      bool dual_infeasible = ((pbm.P * dx).template lpNorm<Inf>() <= prm.eps_dual_inf * dx_norm)
                          && (pbm.q.dot(dx) <= prm.eps_dual_inf * dx_norm);
      const Rm Adx = pbm.A * dx;
      for (auto i = 0u; i != m && dual_infeasible; ++i) {
        if (pbm.u(i) == std::numeric_limits<double>::infinity()) {
          dual_infeasible &= (Adx(i) >= -prm.eps_dual_inf * dx_norm);
        } else if (pbm.l(i) == -std::numeric_limits<double>::infinity()) {
          dual_infeasible &= (Adx(i) <= prm.eps_dual_inf * dx_norm);
        } else {
          dual_infeasible &= std::abs(Adx(i)) < prm.eps_dual_inf * dx_norm;
        }
      }

      if (dual_infeasible) { return {.code = ExitCode::DualInfeasible, .primal = {}, .dual = {}}; }
    }

    x = x_next, y = y_next, z = z_next;
  }

  return {.code = ExitCode::MaxIterations, .primal = std::move(x), .dual = std::move(y)};
}

// Solution<-1, -1> solve(const QuadraticProgramSparse &, const SolverParams &) { return {}; }

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__QP_HPP_
