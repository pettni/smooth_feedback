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

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <limits>

#include "internal/ldlt_lapack.hpp"
#include "internal/ldlt_sparse.hpp"

namespace smooth::feedback {

using detail::LDLTSparse, detail::LDLTLapack;

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
 *  \text{s.t.} & l \leq A x \leq u,
 * \end{cases}
 * \f]
 * where \f$ P \in \mathbb{R}^{n \times n}, q \in \mathbb{R}^n, l, u \in \mathbb{R}^m, A \in
 * \mathbb{R}^{m \times n} \f$.
 */
template<Eigen::Index M, Eigen::Index N, typename Scalar = double>
struct QuadraticProgram
{
  /// Positive semi-definite square cost (only upper trianglular part is used)
  Eigen::Matrix<Scalar, N, N> P;
  /// Linear cost
  Eigen::Matrix<Scalar, N, 1> q;

  /// Inequality matrix (only upper triangular part is used)
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
 *  \text{s.t.} & l \leq A x \leq u,
 * \end{cases}
 * \f]
 * where \f$ P \in \mathbb{R}^{n \times n}, q \in \mathbb{R}^n, l, u \in \mathbb{R}^m, A \in
 * \mathbb{R}^{m \times n} \f$.
 */
template<typename Scalar = double>
struct QuadraticProgramSparse
{
  /// Positive semi-definite square cost (only upper trianglular part is used)
  Eigen::SparseMatrix<Scalar> P;
  /// Linear cost
  Eigen::Matrix<Scalar, -1, 1> q;

  /**
   * @brief Inequality matrix (only upper triangular part is used)
   *
   * @note The constraint matrix is stored in row-major format,
   * i.e. coefficients for each constraint are contiguous in memory
   */
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor> A;
  /// Inequality lower bound
  Eigen::Matrix<Scalar, -1, 1> l;
  /// Inequality upper bound
  Eigen::Matrix<Scalar, -1, 1> u;
};

/// Solver exit codes
enum class ExitCode : int {
  Optimal,
  PolishFailed,
  PrimalInfeasible,
  DualInfeasible,
  MaxIterations,
  Unknown
};

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

// Traits to figure solution type from problem type
// \cond
template<typename T>
struct QpSol;

template<typename Scalar, Eigen::Index M, Eigen::Index N>
struct QpSol<QuadraticProgram<M, N, Scalar>>
{
  using type = Solution<M, N, Scalar>;
};

template<typename Scalar>
struct QpSol<QuadraticProgramSparse<Scalar>>
{
  using type = Solution<-1, -1, Scalar>;
};

template<typename T>
using QpSol_t = typename QpSol<T>::type;
// \endcond

template<typename Pbm>
auto scaleQp(const Pbm & pbm)
{
  using AmatT                  = decltype(Pbm::A);
  using Scalar                 = typename AmatT::Scalar;
  static constexpr bool sparse = std::is_base_of_v<Eigen::SparseMatrixBase<AmatT>, AmatT>;

  Pbm ret = pbm;

  static constexpr Eigen::Index M = AmatT::RowsAtCompileTime;
  static constexpr Eigen::Index N = AmatT::ColsAtCompileTime;
  static constexpr Eigen::Index K = (N == -1 || M == -1) ? Eigen::Index(-1) : N + M;

  const Eigen::Index n = ret.A.cols(), m = ret.A.rows(), k = n + m;
  const auto norm = [](auto && t) -> Scalar { return t.template lpNorm<Eigen::Infinity>(); };

  Eigen::Matrix<Scalar, K, 1> scale = Eigen::Matrix<Scalar, K, 1>::Ones(k);  // scaling
  Eigen::Matrix<Scalar, K, 1> d_scale(k);                                    // incremental scaling

  // find "norm" of cost function
  if constexpr (sparse) {
    for (auto i = 0u; i != n; ++i) {
      d_scale(i) = 0;
      // traverse each col of P
      for (Eigen::SparseMatrix<double>::InnerIterator it(ret.P, i); it; ++it) {
        d_scale(i) = std::max(d_scale(i), std::abs(it.value()));
      }
    }
  } else {
    d_scale.head(n) = ret.P.colwise().template lpNorm<Eigen::Infinity>();
  }

  // scale cost function
  Scalar c = Scalar(1) / std::max({1e-3, d_scale.head(n).mean(), norm(ret.q)});
  ret.P *= c;
  ret.q *= c;

  int it = 0;

  // calculate norm for every column of [P A' ; A 0]
  do {
    if constexpr (sparse) {
      // A is stored row-wise
      // P is stored col-wise
      Eigen::SparseMatrix<Scalar> Acol = ret.A;
      for (auto i = 0u; i != n; ++i) {
        d_scale(i) = 0;
        // traverse col i of P
        for (Eigen::SparseMatrix<double>::InnerIterator it(ret.P, i); it; ++it) {
          d_scale(i) = std::max(d_scale(i), std::abs(it.value()));
        }
        // traverse col i of A
        for (Eigen::SparseMatrix<double>::InnerIterator it(Acol, i); it; ++it) {
          d_scale(i) = std::max(d_scale(i), std::abs(it.value()));
        }
      }
      for (auto i = 0u; i != m; ++i) {
        d_scale(n + i) = 0;
        // traverse row i of A
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(ret.A, i); it; ++it) {
          d_scale(n + i) = std::max(d_scale(n + i), std::abs(it.value()));
        }
      }
    } else {
      d_scale.head(n) = ret.P.colwise().template lpNorm<Eigen::Infinity>().cwiseMax(
        ret.A.colwise().template lpNorm<Eigen::Infinity>());
      d_scale.tail(m) = ret.A.rowwise().template lpNorm<Eigen::Infinity>();
    }

    // make sure we are not doing anything stupid
    d_scale = d_scale.cwiseMax(1e-3).cwiseInverse().cwiseSqrt();

    // perform scaling
    ret.P = d_scale.head(n).asDiagonal() * ret.P * d_scale.head(n).asDiagonal();
    ret.q.applyOnTheLeft(d_scale.head(n).asDiagonal());
    ret.A = d_scale.tail(m).asDiagonal() * ret.A * d_scale.head(n).asDiagonal();
    ret.l.applyOnTheLeft(d_scale.tail(m).asDiagonal());
    ret.u.applyOnTheLeft(d_scale.tail(m).asDiagonal());

    scale.applyOnTheLeft(d_scale.asDiagonal());

  } while (it++ < 10 && (d_scale.array() - 1).abs().maxCoeff() > 0.1);

  return std::make_tuple(ret, scale, c);
}

/**
 * @brief Solve a quadratic program using the operator splitting approach.
 *
 * @tparam Pbm problem type (QuadraticProgram or QuadraticProgramSparse)
 *
 * @param pbm problem formulation
 * @param prm solver options
 * @param hotstart provide initial guess for primal and dual variables
 * @return solution as Solution<M, N>
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
template<typename Pbm>
QpSol_t<Pbm> solveQP(const Pbm & pbm,
  const SolverParams & prm,
  std::optional<std::reference_wrapper<const QpSol_t<Pbm>>> hotstart = {})
{
  using AmatT                  = decltype(Pbm::A);
  static constexpr bool sparse = std::is_base_of_v<Eigen::SparseMatrixBase<AmatT>, AmatT>;

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

  // scale problem
  // new problem is s.t.
  //  Pnew = c * S_x * P * S_x
  //  qnew = c * q * S_x
  //  Enew = S_e * E * S_x
  //  l = S_e * l
  //  u = S_e * u
  // where S_x = diag(s[0:n])  and S_e = diag(s[n:n+m])
  const auto [spbm, s, c] = scaleQp(pbm);
  const Rn sd             = s.head(n);
  const Rm se             = s.tail(m);

  // check that feasible set is nonempty
  if ((spbm.u - spbm.l).minCoeff() < Scalar(0.) || spbm.l.maxCoeff() == inf
      || spbm.u.minCoeff() == -inf) {
    return {.code = ExitCode::PrimalInfeasible, .primal = {}, .dual = {}};
  }

  // square symmetric system matrix
  std::conditional_t<sparse, Eigen::SparseMatrix<Scalar>, Eigen::Matrix<Scalar, K, K>> H(k, k);

  if constexpr (sparse) {
    // preallocate nonzeros
    Eigen::Matrix<int, -1, 1> nnz(k);
    for (auto i = 0u; i != n; ++i) {
      nnz(i) = spbm.P.outerIndexPtr()[i + 1] - spbm.P.outerIndexPtr()[i] + 1;
    }
    for (auto i = 0u; i != m; ++i) {
      nnz(n + i) = spbm.A.outerIndexPtr()[i + 1] - spbm.A.outerIndexPtr()[i] + 1;
    }
    H.reserve(nnz);

    using PIter = typename Eigen::SparseMatrix<Scalar, Eigen::ColMajor>::InnerIterator;
    using AIter = typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator;

    for (Eigen::Index col = 0u; col != n; ++col) {
      for (PIter it(spbm.P, col); it && it.index() <= col; ++it) {
        H.insert(it.index(), col) = it.value();
      }
      H.coeffRef(col, col) += sigma;
    }

    for (auto row = 0u; row != m; ++row) {
      for (AIter it(spbm.A, row); it; ++it) { H.insert(it.index(), n + row) = it.value(); }
      H.insert(n + row, n + row) = -Scalar(1) / rho;
    }

    H.makeCompressed();
  } else {
    H.template topLeftCorner<N, N>(n, n) = spbm.P;
    H.template topLeftCorner<N, N>(n, n) += Rn::Constant(n, sigma).asDiagonal();
    H.template topRightCorner<N, M>(n, m)    = spbm.A.transpose();
    H.template bottomRightCorner<M, M>(m, m) = Rm::Constant(m, Scalar(-1.) / rho).asDiagonal();
  }

  // factorize H
  std::conditional_t<sparse, detail::LDLTSparse<Scalar>, detail::LDLTLapack<Scalar, K>> ldlt(
    std::move(H));

  if (ldlt.info()) { return {.code = ExitCode::Unknown, .primal = {}, .dual = {}}; }

  // initialization
  Rn x;
  Rm z, y;
  if (hotstart.has_value()) {
    x = hotstart.value().get().primal;
    x.applyOnTheLeft(sd.cwiseInverse().asDiagonal());
    y = hotstart.value().get().dual;
    y.applyOnTheLeft(se.asDiagonal());
    z.noalias() = spbm.A * x;
  } else {
    x.setZero(n);
    y.setZero(m);
    z.setZero(m);
  }

  // working arrays
  Rn x_next(n), Px(n), Aty(n), dx(n);
  Rm z_interp(m), z_next(m), y_next(m), Ax(m), dy(m);
  Rk p(k);

  for (auto i = 0u; i != prm.max_iter; ++i) {

    // ADMM ITERATION

    // solve linear system
    p.head(n) = sigma * x - spbm.q;
    p.tail(m) = z - y / rho;
    ldlt.solve_inplace(p);

    EIGEN_ASM_COMMENT("admm");
    x_next   = alpha * p.head(n) + x - alpha * x;
    z_interp = alpha * z + (alpha / rho) * p.tail(m) - (alpha / rho) * y + z - alpha * z;
    z_next   = (z_interp + y / rho).cwiseMax(spbm.l).cwiseMin(spbm.u);
    y_next   = y + rho * z_interp - rho * z_next;
    EIGEN_ASM_COMMENT("/admm");

    // CHECK STOPPING CRITERIA

    if (i % prm.stop_check_iter == prm.stop_check_iter - 1) {

      // OPTIMALITY

      Ax.noalias() = spbm.A * x;
      Ax.applyOnTheLeft(se.cwiseInverse().asDiagonal());

      // clang-format off
      // check primal
      bool optimal = true;
      const Scalar primal_val   = norm(Ax - se.cwiseInverse().cwiseProduct(z));
      const Scalar primal_scale = std::max<Scalar>(norm(Ax), norm(se.cwiseInverse().cwiseProduct(z)));
      if (primal_val > prm.eps_abs + prm.eps_rel * primal_scale) {
        optimal = false;
      }
      if (optimal) {
        // primal succeeded, check dual
        Px.noalias() = spbm.P * x;
        Px.applyOnTheLeft(sd.cwiseInverse().asDiagonal());
        Aty.noalias() = spbm.A.transpose() * y;
        Aty.applyOnTheLeft(sd.cwiseInverse().asDiagonal());
        const Scalar dual_val = norm(Px + sd.cwiseInverse().cwiseProduct(spbm.q) + Aty);
        const Scalar dual_scale = std::max<Scalar>({norm(Px), norm(sd.cwiseInverse().cwiseProduct(spbm.q)), norm(Aty)});
        if (dual_val > c * prm.eps_abs + prm.eps_rel * dual_scale) { optimal = false; }
      }
      if (optimal) {
        QpSol_t<Pbm> sol{.code = ExitCode::Optimal, .primal = std::move(x), .dual = std::move(y)};
        if (prm.polish) { polishQP(spbm, sol, prm); }
        sol.primal.applyOnTheLeft(sd.asDiagonal());
        sol.dual.applyOnTheLeft(se.cwiseInverse().asDiagonal());
        return sol;
      }
      // clang-format on

      // PRIMAL INFEASIBILITY

      dx = x_next - x;
      dy = y_next - y;

      const Scalar Edy_norm      = norm(se.cwiseProduct(dy));
      const Scalar Di_At_dy_norm = norm(sd.cwiseInverse().asDiagonal() * spbm.A.transpose() * dy);

      Scalar u_dyp_plus_l_dyn = Scalar(0);
      for (auto i = 0u; i != m; ++i) {
        if (spbm.u(i) != inf) {
          u_dyp_plus_l_dyn += spbm.u(i) * std::max<Scalar>(Scalar(0), dy(i));
        } else if (dy(i) > prm.eps_primal_inf * Edy_norm) {
          // contributes +inf to sum --> no certificate
          u_dyp_plus_l_dyn = inf;
          break;
        }
        if (spbm.l(i) != -inf) {
          u_dyp_plus_l_dyn += spbm.l(i) * std::min<Scalar>(Scalar(0), dy(i));
        } else if (dy(i) < -prm.eps_primal_inf * Edy_norm) {
          // contributes +inf to sum --> no certificate
          u_dyp_plus_l_dyn = inf;
          break;
        }
      }

      if (std::max<Scalar>(Di_At_dy_norm, u_dyp_plus_l_dyn) < prm.eps_primal_inf * Edy_norm) {
        return {.code = ExitCode::PrimalInfeasible, .primal = {}, .dual = {}};
      }

      // DUAL INFEASIBILITY

      const Scalar D_dx_norm = norm(sd.cwiseProduct(dx));  // holds n( D * dx )
      Ax.noalias()           = spbm.A * dx;                // note new value Einv * A * dx
      Ax.applyOnTheLeft(se.cwiseInverse().asDiagonal());
      Px.noalias() = spbm.P * dx;  // note new value Dinv * P * dx
      Px.applyOnTheLeft(sd.cwiseInverse().asDiagonal());

      bool dual_infeasible = (norm(Px) <= c * prm.eps_dual_inf * D_dx_norm)
                          && (spbm.q.dot(dx) <= c * prm.eps_dual_inf * D_dx_norm);
      for (auto i = 0u; i != m && dual_infeasible; ++i) {
        if (spbm.u(i) == inf) {
          dual_infeasible &= (Ax(i) >= -prm.eps_dual_inf * D_dx_norm);
        } else if (spbm.l(i) == -inf) {
          dual_infeasible &= (Ax(i) <= prm.eps_dual_inf * D_dx_norm);
        } else {
          dual_infeasible &= std::abs(Ax(i)) < prm.eps_dual_inf * D_dx_norm;
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
 * @tparam Pbm problem type
 *
 * @param[in] spbm problem formulation
 * @param[in, out] sol solution to polish
 * @param[in] prm solver options
 *
 * @warning This function allocates heap memory even for static-sized problems.
 */
template<typename Pbm>
void polishQP(const Pbm & pbm, QpSol_t<Pbm> & sol, const SolverParams & prm)
{
  using AmatT                  = decltype(Pbm::A);
  using Scalar                 = typename AmatT::Scalar;
  using VecX                   = Eigen::Matrix<Scalar, -1, 1>;
  static constexpr bool sparse = std::is_base_of_v<Eigen::SparseMatrixBase<AmatT>, AmatT>;

  const Eigen::Index n = pbm.A.cols(), m = pbm.A.rows();

  // FIND ACTIVE CONSTRAINT SETS

  // TODO would make sense to add small margins here

  Eigen::Index nl = 0, nu = 0;
  for (Eigen::Index idx = 0; idx < m; ++idx) {
    if (sol.dual[idx] < 0) { nl++; }
    if (sol.dual[idx] > 0) { nu++; }
  }

  Eigen::Matrix<Eigen::Index, -1, 1> LU_idx(nl + nu);
  for (Eigen::Index idx = 0, lcntr = 0, ucntr = 0; idx < m; ++idx) {
    if (sol.dual[idx] < 0) { LU_idx(lcntr++) = idx; }
    if (sol.dual[idx] > 0) { LU_idx(nl + ucntr++) = idx; }
  }

  // FORM REDUCED SYSTEMS (27) AND (30)

  // square symmetric system matrix
  using HT = std::conditional_t<sparse, Eigen::SparseMatrix<Scalar>, Eigen::Matrix<Scalar, -1, -1>>;
  HT H(n + nl + nu, n + nl + nu);

  // fill up H
  if constexpr (sparse) {
    // preallocate nonzeros (extra 1 for H_per)
    Eigen::Matrix<int, -1, 1> nnz(n + nl + nu);
    for (auto i = 0u; i != n; ++i) {
      nnz(i) = pbm.P.outerIndexPtr()[i + 1] - pbm.P.outerIndexPtr()[i] + 1;
    }
    for (auto i = 0u; i != nl + nu; ++i) {
      nnz(n + i) = pbm.A.outerIndexPtr()[LU_idx(i) + 1] - pbm.A.outerIndexPtr()[LU_idx(i)] + 1;
    }
    H.reserve(nnz);

    using PIter = typename Eigen::SparseMatrix<Scalar, Eigen::ColMajor>::InnerIterator;
    using AIter = typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator;

    // fill P in top left block
    for (Eigen::Index p_col = 0u; p_col != n; ++p_col) {
      for (PIter it(pbm.P, p_col); it && it.index() <= p_col; ++it) {
        H.insert(it.index(), p_col) = it.value();
      }
    }

    // fill selected rows of A in top right block
    for (auto a_row = 0u; a_row != nl + nu; ++a_row) {
      for (AIter it(pbm.A, LU_idx(a_row)); it; ++it) {
        H.insert(it.index(), n + a_row) = it.value();
      }
    }
  } else {
    H.setZero();
    H.topLeftCorner(n, n) = pbm.P;
    for (auto i = 0u; i != nl + nu; ++i) { H.col(n + i).head(n) = pbm.A.row(LU_idx(i)); }
  }

  HT Hp = H;

  // add perturbing diagonal elements to Hp
  if constexpr (sparse) {
    for (auto i = 0u; i != n; ++i) { Hp.coeffRef(i, i) += prm.delta; }
    for (auto i = 0u; i != nl + nu; ++i) { Hp.coeffRef(n + i, n + i) -= prm.delta; }
    H.makeCompressed();
    Hp.makeCompressed();
  } else {
    Hp.topLeftCorner(n, n) += VecX::Constant(n, prm.delta).asDiagonal();
    Hp.bottomRightCorner(nl + nu, nl + nu) -= VecX::Constant(nl + nu, prm.delta).asDiagonal();
  }

  VecX h(n + nl + nu);
  h << -pbm.q, LU_idx.head(nl).unaryExpr(pbm.l), LU_idx.tail(nu).unaryExpr(pbm.u);

  // ITERATIVE REFINEMENT

  // factorize Hp
  std::conditional_t<sparse, LDLTSparse<Scalar>, LDLTLapack<Scalar, -1>> ldlt(std::move(Hp));

  if (ldlt.info()) {
    sol.code = ExitCode::PolishFailed;
    return;
  }

  VecX t_hat = VecX::Zero(n + nl + nu);
  for (auto i = 0u; i != prm.polish_iter; ++i) {
    t_hat += ldlt.solve(h - H.template selfadjointView<Eigen::Upper>() * t_hat);
  }

  // UPDATE SOLUTION

  sol.primal = t_hat.head(n);
  for (Eigen::Index i = 0; i < nl; ++i) { sol.dual(LU_idx(i)) = t_hat(n + i); }
  for (Eigen::Index i = 0; i < nu; ++i) { sol.dual(LU_idx(nl + i)) = t_hat(n + nl + i); }

  // TODO print polishing info (if verbose flag)
}

// Solution<-1, -1> solve(const QuadraticProgramSparse &, const SolverParams &) { return {}; }

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__QP_HPP_
