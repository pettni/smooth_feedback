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

#include <limits>

#include "internal/ldlt_lapack.hpp"
#include "internal/ldlt_sparse.hpp"

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
template<Eigen::Index M, Eigen::Index N, typename Scalar = double>
struct QuadraticProgram
{
  /// Quadratic cost
  Eigen::Matrix<Scalar, N, N> P;
  /// Linear cost
  Eigen::Matrix<Scalar, N, 1> q;

  /// Inequality matrix
  Eigen::Matrix<Scalar, M, N> A;
  /// Inequality lower bound
  Eigen::Matrix<Scalar, M, 1> l;
  /// Inequality upper bound
  Eigen::Matrix<Scalar, M, 1> u;
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
template<typename Scalar = double>
struct QuadraticProgramSparse
{
  /// Quadratic cost
  Eigen::SparseMatrix<Scalar> P;
  /// Linear cost
  Eigen::Matrix<Scalar, -1, 1> q;

  /// Inequality matrix
  Eigen::SparseMatrix<Scalar> A;
  /// Inequality lower bound
  Eigen::Matrix<Scalar, -1, 1> l;
  /// Inequality upper bound
  Eigen::Matrix<Scalar, -1, 1> u;
};

/// Solver exit codes
enum class ExitCode : int { Optimal, PrimalInfeasible, DualInfeasible, MaxIterations, Unknown };

/// Solver solution
template<Eigen::Index M, Eigen::Index N, typename Scalar>
struct Solution
{
  /// Exit code
  ExitCode code = ExitCode::Unknown;
  /// Primal vector
  Eigen::Matrix<Scalar, N, 1> primal;
  /// Dual vector
  Eigen::Matrix<Scalar, M, 1> dual;
};

/**
 * @brief Options for solveQP
 */
struct SolverParams
{
  /// relaxation parameter
  float alpha = 1.6;
  /// first dul step size
  float rho = 0.1;
  /// second dual step length
  float sigma = 1e-6;

  /// absolute threshold for convergence
  float eps_abs = 1e-3;
  /// relative threshold for convergence
  float eps_rel = 1e-3;
  /// threshold for primal infeasibility
  float eps_primal_inf = 1e-4;
  /// threshold for dual infeasibility
  float eps_dual_inf = 1e-4;

  /// max number of iterations
  uint64_t max_iter = std::numeric_limits<uint64_t>::max();

  /// iterations between checking stopping criterion
  uint64_t stop_check_iter = 10;

  /// run solution polishing (uses dynamic memory)
  bool polish = true;
  /// number of iterations to refine polish
  uint64_t polish_iter = 5;
  /// regularization parameter for polishing
  float delta = 1e-6;
};

namespace detail {
template<typename T>
struct qp_traits;

template<typename Scalar, Eigen::Index M, Eigen::Index N>
struct qp_traits<QuadraticProgram<M, N, Scalar>>
{
  static constexpr Eigen::Index K = (N == -1 || M == -1) ? Eigen::Index(-1) : N + M;

  using sol_t = Solution<M, N, Scalar>;
  using fac_t = LDLTLapack<Scalar, K>;
};

template<typename Scalar>
struct qp_traits<QuadraticProgramSparse<Scalar>>
{
  using sol_t = Solution<-1, -1, Scalar>;
  using fac_t = LDLTSparse<double>;
};
}  // namespace detail

/**
 * @brief Solve a quadratic program using the operator splitting approach.
 *
 * @tparam M number of constraints
 * @tparam N number of variables
 *
 * @param pbm problem formulation
 * @param prm solver options
 * @param hotstart provide initial guess for primal and dual variables
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
template<typename Problem>
typename detail::qp_traits<Problem>::sol_t solveQP(const Problem & pbm,
  const SolverParams & prm,
  std::optional<std::reference_wrapper<const typename detail::qp_traits<Problem>::sol_t>> hotstart =
    {})
{
  using AmatT                     = decltype(Problem::A);
  static constexpr bool is_sparse = std::is_base_of_v<Eigen::SparseMatrixBase<AmatT>, AmatT>;

  // static sizes
  static constexpr Eigen::Index M = AmatT::RowsAtCompileTime;
  static constexpr Eigen::Index N = AmatT::ColsAtCompileTime;
  static constexpr Eigen::Index K = (N == -1 || M == -1) ? Eigen::Index(-1) : N + M;

  // typedefs
  using Scalar = typename AmatT::Scalar;
  using Rn     = Eigen::Matrix<Scalar, N, 1>;
  using Rm     = Eigen::Matrix<Scalar, M, 1>;
  using Rk     = Eigen::Matrix<Scalar, K, 1>;

  // dynamic sizes
  const Eigen::Index n = pbm.A.cols(), m = pbm.A.rows(), k = n + m;

  const auto norm = [](auto && t) -> Scalar { return t.template lpNorm<Eigen::Infinity>(); };
  static constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();

  // cast parameters to scalar type
  const Scalar rho   = static_cast<Scalar>(prm.rho);
  const Scalar alpha = static_cast<Scalar>(prm.alpha);
  const Scalar sigma = static_cast<Scalar>(prm.sigma);

  // check that feasible set is nonempty
  if ((pbm.u - pbm.l).minCoeff() < Scalar(0.) || pbm.l.maxCoeff() == inf
      || pbm.u.minCoeff() == -inf) {
    return {.code = ExitCode::PrimalInfeasible, .primal = {}, .dual = {}};
  }

  // TODO problem scaling / conditioning

  // define system matrix
  std::conditional_t<is_sparse, Eigen::SparseMatrix<double>, Eigen::Matrix<Scalar, K, K>> H(k, k);

  if constexpr (is_sparse) {
    // TODO preallocate pattern

    for (auto col = 0u; col != n; ++col) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(pbm.P, col); it; ++it) {
        H.coeffRef(it.index(), col) = it.value();
      }
      H.coeffRef(col, col) += sigma;
    }

    for (auto col = 0u; col != m; ++col) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(pbm.A, col); it; ++it) {
        H.coeffRef(col, n + it.index()) = it.value();
      }
      H.coeffRef(n + col, n + col) -= Scalar(1) / rho;
    }
  } else {
    H.template topLeftCorner<N, N>(n, n) = pbm.P;
    H.template topLeftCorner<N, N>(n, n) += Rn::Constant(n, sigma).asDiagonal();
    H.template topRightCorner<N, M>(n, m)  = pbm.A.transpose();
    H.template bottomRightCorner<M, M>(m, m) = Rm::Constant(m, Scalar(-1.) / rho).asDiagonal();
  }

  typename detail::qp_traits<Problem>::fac_t ldlt(H);

  if (ldlt.info()) { return {.code = ExitCode::Unknown, .primal = {}, .dual = {}}; }

  // initialization
  Rn x(n);
  Rm z(m);
  Rm y(m);

  if (hotstart.has_value()) {
    x = hotstart.value().get().primal;
    z = pbm.A * x;
    y = hotstart.value().get().dual;
  } else {
    // TODO what's a good choice here?
    x.setZero();
    y.setZero();
    z.setZero();
  }

  for (auto i = 0u; i != prm.max_iter; ++i) {

    // ADMM ITERATION

    // solve linear system H p = h
    const Rk h = (Rk(k) << sigma * x - pbm.q, z - y / rho).finished();
    const Rk p = ldlt.solve(h);

    const Rm z_tilde  = z + (p.tail(m) - y) / rho;
    const Rn x_next   = alpha * p.head(n) + (Scalar(1.) - alpha) * x;
    const Rm z_interp = alpha * z_tilde + (Scalar(1.) - alpha) * z;
    const Rm z_next   = (z_interp + y / rho).cwiseMax(pbm.l).cwiseMin(pbm.u);
    const Rm y_next   = y + rho * (z_interp - z_next);

    // CHECK STOPPING CRITERIA

    if (i % prm.stop_check_iter == prm.stop_check_iter - 1) {

      // OPTIMALITY

      const Rn Px  = pbm.P * x;
      const Rm Ax  = pbm.A * x;
      const Rn Aty = pbm.A.transpose() * y;

      const Scalar primal_scale = std::max<Scalar>(norm(Ax), norm(z));
      const Scalar dual_scale   = std::max<Scalar>({norm(Px), norm(pbm.q), norm(Aty)});

      if (norm(Ax - z) <= prm.eps_abs + prm.eps_rel * primal_scale
          && norm(Px + pbm.q + Aty) <= prm.eps_abs + prm.eps_abs * dual_scale) {
        if constexpr (!is_sparse) {
          if (prm.polish) { polishQP<M, N, Scalar>(x, y, pbm, prm); }
        }
        return {.code = ExitCode::Optimal, .primal = std::move(x), .dual = std::move(y)};
      }

      // PRIMAL INFEASIBILITY

      const Rn dx             = x_next - x;
      const Rm dy             = y_next - y;
      const Scalar dx_norm    = norm(dx);
      const Scalar dy_norm    = norm(dy);
      const Scalar At_dy_norm = norm(pbm.A.transpose() * dy);

      Scalar u_dyp_plus_l_dyn = Scalar(0);
      for (auto i = 0u; i != m; ++i) {
        if (pbm.u(i) != inf) {
          u_dyp_plus_l_dyn += pbm.u(i) * std::max<Scalar>(Scalar(0), dy(i));
        } else if (dy(i) > prm.eps_primal_inf * dy_norm) {
          // contributes +inf to sum --> no certificate
          u_dyp_plus_l_dyn = inf;
          break;
        }
        if (pbm.l(i) != -inf) {
          u_dyp_plus_l_dyn += pbm.l(i) * std::min<Scalar>(Scalar(0), dy(i));
        } else if (dy(i) < -prm.eps_primal_inf * dy_norm) {
          // contributes +inf to sum --> no certificate
          u_dyp_plus_l_dyn = inf;
          break;
        }
      }

      if (std::max<Scalar>(At_dy_norm, u_dyp_plus_l_dyn) < prm.eps_primal_inf * dy_norm) {
        return {.code = ExitCode::PrimalInfeasible, .primal = {}, .dual = {}};
      }

      // DUAL INFEASIBILITY

      bool dual_infeasible = (norm(pbm.P * dx) <= prm.eps_dual_inf * dx_norm)
                          && (pbm.q.dot(dx) <= prm.eps_dual_inf * dx_norm);
      const Rm Adx = pbm.A * dx;
      for (auto i = 0u; i != m && dual_infeasible; ++i) {
        if (pbm.u(i) == inf) {
          dual_infeasible &= (Adx(i) >= -prm.eps_dual_inf * dx_norm);
        } else if (pbm.l(i) == -inf) {
          dual_infeasible &= (Adx(i) <= prm.eps_dual_inf * dx_norm);
        } else {
          dual_infeasible &= std::abs(Adx(i)) < prm.eps_dual_inf * dx_norm;
        }
      }

      if (dual_infeasible) { return {.code = ExitCode::DualInfeasible, .primal = {}, .dual = {}}; }

      // TODO print solver info (if verbose flag)
    }

    x = x_next, y = y_next, z = z_next;
  }

  return {.code = ExitCode::MaxIterations, .primal = std::move(x), .dual = std::move(y)};
}

/**
 * @brief Polish solution of quadratic program
 *
 * @tparam M number of constrains
 * @tparam N number of variables
 * @tparam Scalar scalar type for linear algebra
 *
 * @param[out] primal primal solution to qp problem
 * @param[in,out] dual dual solution to qp problem
 * @param[in] pbm problem formulation
 * @param[in] prm solver options
 *
 * @return \p true if polish succeeded, \p false otherwise
 */
template<Eigen::Index M, Eigen::Index N, typename Scalar>
bool polishQP(Eigen::Matrix<Scalar, N, 1> & primal,
  Eigen::Matrix<Scalar, M, 1> & dual,
  const QuadraticProgram<M, N, Scalar> & pbm,
  const SolverParams & prm)
{
  const Eigen::Index n = pbm.A.cols(), m = pbm.A.rows();

  // FIND ACTIVE CONTRAINT SETS

  Eigen::Index nl = 0, nu = 0;
  for (Eigen::Index idx = 0; idx < m; ++idx) {
    if (dual[idx] < 0) { nl++; }
    if (dual[idx] > 0) { nu++; }
  }

  Eigen::Matrix<Eigen::Index, -1, 1> L_indices(nl), U_indices(nu);
  nl = 0, nu = 0;
  for (Eigen::Index idx = 0; idx < m; ++idx) {
    if (dual[idx] < 0) { L_indices(nl++) = idx; }
    if (dual[idx] > 0) { U_indices(nu++) = idx; }
  }
  const Eigen::Index nb = n + nl + nu;

  // FORM REDUCED SYSTEMS (27) AND (30)

  Eigen::Matrix<Scalar, -1, -1> H(nb, nb);
  H.setZero();
  H.template topLeftCorner<N, N>(n, n) = pbm.P;
  for (auto i = 0u; i != nl; ++i) { H.col(n + i).head(n) = pbm.A.row(L_indices(i)); }
  for (auto i = 0u; i != nu; ++i) { H.col(n + nl + i).head(n) = pbm.A.row(U_indices(i)); }

  Eigen::Matrix<Scalar, -1, -1> H_per = H;
  H.topLeftCorner(n, n) += Eigen::Matrix<Scalar, -1, 1>::Constant(n, prm.delta).asDiagonal();
  H.bottomRightCorner(nl + nu, nl + nu) -=
    Eigen::Matrix<Scalar, -1, 1>::Constant(nl + nu, prm.delta).asDiagonal();

  Eigen::Matrix<Scalar, -1, 1> h(nb);
  h << -pbm.q, L_indices.head(nl).unaryExpr(pbm.l), U_indices.head(nu).unaryExpr(pbm.u);

  // ITERATIVE REFINEMENT

  detail::LDLTLapack<Scalar, -1> ldlt(H_per);
  if (ldlt.info()) { return false; }  // polishing failed

  Eigen::Matrix<Scalar, -1, 1> t_hat = Eigen::Matrix<Scalar, -1, 1>::Zero(nb);
  for (auto i = 0u; i != prm.polish_iter; ++i) {
    t_hat += ldlt.solve(h - H.template selfadjointView<Eigen::Upper>() * t_hat);
  }

  // UPDATE SOLUTION

  primal = t_hat.template head<N>(n);
  for (Eigen::Index i = 0; i < nl; ++i) { dual(L_indices(i)) = t_hat(n + i); }
  for (Eigen::Index i = 0; i < nu; ++i) { dual(U_indices(i)) = t_hat(n + nl + i); }

  // TODO print polishing info (if verbose flag)

  return true;
}

// Solution<-1, -1> solve(const QuadraticProgramSparse &, const SolverParams &) { return {}; }

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__QP_HPP_
