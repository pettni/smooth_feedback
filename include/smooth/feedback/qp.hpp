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
  Optimal,           /// Solution satisifes optimality condition. Solution is polished if
                     /// `SolverParams::polish = true`.
  PolishFailed,      /// Solution satisfies optimality condition but is not polished
  PrimalInfeasible,  /// A certificate of primal infeasibility was found, no solution returned
  DualInfeasible,    /// A certificate of dual infeasibility was found, no solution returned
  MaxIterations,     /// Max number of iterations was reached, returned solution is not optimal
  Unknown            /// Solution is useless because of other reasons, no solution returned
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
 * @brief Options for solve_qp
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

/**
 * @brief Re-scale a QuadraticProgram
 *
 * @param pbm problem \f$ (P, q, A, l, u) \f$ to rescale.
 *
 * @returns tuple `(spbm, s, c)` where `spbm` is a scaled problem \f$ (\bar P, \bar q, \bar A, \bar
 * l, \bar u)\f$.
 *
 * The scaled problem is defined as
 *
 * * \f$ \bar P = c S_x P S_x \f$,
 * * \f$ \bar q = c q S_x \f$,
 * * \f$ \bar A = S_e A S_x \f$,
 * * \f$ \bar l = S_e l \f$,
 * * \f$ \bar u = S_e u \f$,
 *
 * where \f$ S_x = diag(s_{0:n}), S_e = diag(s_{n:n+m}) \f$.
 *
 * The relation between scaled variables and original variables are
 *
 * * Primal: \f$ \bar x = S_x^{-1} x \f$,
 * * Dual: \f$ \bar y = c E_x^{-1} x \f$.
 *
 * The objective of the rescaling is the make the columns of
 * \f[
 *   \begin{bmatrix} \bar P & \bar A^T \\ \bar A & 0 \end{bmatrix}
 * \f]
 * have similar \f$ l_\infty \f$ norm, and similarly for
 * the columns of
 * \f[
 *  \begin{bmatrix} \bar P & \bar q \end{bmatrix}.
 * \f]
 */
template<typename Pbm>
auto scale_qp(const Pbm & pbm)
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
    d_scale.template head<N>(n) = ret.P.colwise().template lpNorm<Eigen::Infinity>();
  }

  // scale cost function
  Scalar c = Scalar(1) / std::max({1e-3, d_scale.template head<N>(n).mean(), norm(ret.q)});
  ret.P *= c;
  ret.q *= c;

  int iter = 0;

  // calculate norm for every column of [P A' ; A 0]
  do {
    if constexpr (sparse) {
      // P is stored col wise
      for (auto i = 0u; i != n; ++i) {
        d_scale(i) = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(ret.P, i); it; ++it) {
          // upper left block of H
          d_scale(i) = std::max(d_scale(i), std::abs(it.value()));
        }
      }
      // A is stored row wise
      for (auto i = 0u; i != m; ++i) {
        d_scale(n + i) = 0;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(ret.A, i); it; ++it) {
          // bottom left block of H
          d_scale(it.index()) = std::max(d_scale(it.index()), std::abs(it.value()));
          // upper right block of H
          d_scale(n + i) = std::max(d_scale(n + i), std::abs(it.value()));
        }
      }
    } else {
      d_scale.template head<N>(n) = ret.P.colwise().template lpNorm<Eigen::Infinity>().cwiseMax(
        ret.A.colwise().template lpNorm<Eigen::Infinity>());
      d_scale.template segment<M>(n, m) = ret.A.rowwise().template lpNorm<Eigen::Infinity>();
    }

    // make sure we are not doing anything stupid
    d_scale = d_scale.cwiseMax(1e-3).cwiseInverse().cwiseSqrt();

    // perform scaling
    if constexpr (sparse) {
      ret.P =
        d_scale.template head<N>(n).asDiagonal() * ret.P * d_scale.template head<N>(n).asDiagonal();
      ret.A = d_scale.template segment<M>(n, m).asDiagonal() * ret.A
            * d_scale.template head<N>(n).asDiagonal();
    } else {
      ret.P.applyOnTheLeft(d_scale.template head<N>(n).asDiagonal());
      ret.P.applyOnTheRight(d_scale.template head<N>(n).asDiagonal());
      ret.A.applyOnTheLeft(d_scale.template segment<M>(n, m).asDiagonal());
      ret.A.applyOnTheRight(d_scale.template head<N>(n).asDiagonal());
    }
    ret.q.applyOnTheLeft(d_scale.template head<N>(n).asDiagonal());
    ret.l.applyOnTheLeft(d_scale.template segment<M>(n, m).asDiagonal());
    ret.u.applyOnTheLeft(d_scale.template segment<M>(n, m).asDiagonal());

    scale.applyOnTheLeft(d_scale.asDiagonal());
  } while (iter++ < 10 && (d_scale.array() - 1).abs().maxCoeff() > 0.1);

  return std::make_tuple(std::move(ret), std::move(scale), c);
}

/**
 * @brief Polish solution of quadratic program
 *
 * @tparam Pbm problem type
 *
 * @param[in] pbm problem formulation
 * @param[in, out] sol solution to polish
 * @param[in] prm solver options
 *
 * @warning This function allocates heap memory even for static-sized problems.
 */
template<typename Pbm>
bool polish_qp(const Pbm & pbm, QpSol_t<Pbm> & sol, const SolverParams & prm)
{
  using AmatT                  = decltype(Pbm::A);
  using Scalar                 = typename AmatT::Scalar;
  using VecX                   = Eigen::Matrix<Scalar, -1, 1>;
  static constexpr bool sparse = std::is_base_of_v<Eigen::SparseMatrixBase<AmatT>, AmatT>;
  using LDLT =
    std::conditional_t<sparse, detail::LDLTSparse<Scalar>, detail::LDLTLapack<Scalar, -1>>;

  static constexpr Eigen::Index N = AmatT::ColsAtCompileTime;
  const Eigen::Index n = pbm.A.cols(), m = pbm.A.rows();

  // FIND ACTIVE CONSTRAINT SETS

  Eigen::Index nl = 0, nu = 0;
  for (Eigen::Index idx = 0; idx < m; ++idx) {
    if (sol.dual[idx] < -10 * std::numeric_limits<Scalar>::epsilon()) { nl++; }
    if (sol.dual[idx] > 10 * std::numeric_limits<Scalar>::epsilon()) { nu++; }
  }

  Eigen::Matrix<Eigen::Index, -1, 1> LU_idx(nl + nu);
  for (Eigen::Index idx = 0, lcntr = 0, ucntr = 0; idx < m; ++idx) {
    if (sol.dual[idx] < -10 * std::numeric_limits<Scalar>::epsilon()) { LU_idx(lcntr++) = idx; }
    if (sol.dual[idx] > 10 * std::numeric_limits<Scalar>::epsilon()) { LU_idx(nl + ucntr++) = idx; }
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
    for (auto i = 0u; i != nl + nu; ++i) {
      H.col(n + i).template head<N>(n) = pbm.A.row(LU_idx(i));
    }
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
  LDLT ldlt(std::move(Hp));
  if (ldlt.info()) { return false; }

  VecX t_hat = VecX::Zero(n + nl + nu);
  for (auto i = 0u; i != prm.polish_iter; ++i) {
    t_hat += ldlt.solve(h - H.template selfadjointView<Eigen::Upper>() * t_hat);
  }

  // UPDATE SOLUTION

  sol.primal = t_hat.template head<N>(n);
  for (Eigen::Index i = 0; i < nl; ++i) { sol.dual(LU_idx(i)) = t_hat(n + i); }
  for (Eigen::Index i = 0; i < nu; ++i) { sol.dual(LU_idx(nl + i)) = t_hat(n + nl + i); }

  return true;
}

/**
 * @brief Check stopping criterion for QP solver.
 */
template<typename Pbm, typename D1, typename D2, typename D3, typename D4, typename D5>
std::optional<ExitCode> qp_check_stopping(const Pbm & pbm,
  const Eigen::MatrixBase<D1> & x,
  const Eigen::MatrixBase<D2> & y,
  const Eigen::MatrixBase<D3> & z,
  const Eigen::MatrixBase<D4> & dx,
  const Eigen::MatrixBase<D5> & dy,
  const SolverParams & prm)
{
  using Scalar = typename decltype(Pbm::A)::Scalar;

  const Eigen::Index n = pbm.A.cols(), m = pbm.A.rows();
  static constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
  const auto norm = [](auto && t) -> Scalar { return t.template lpNorm<Eigen::Infinity>(); };

  // working memory
  Eigen::Matrix<Scalar, decltype(Pbm::A)::ColsAtCompileTime, 1> Px(n), Aty(n);
  Eigen::Matrix<Scalar, decltype(Pbm::A)::RowsAtCompileTime, 1> Ax(m);

  // OPTIMALITY

  // check primal
  Ax.noalias() = pbm.A * x;
  if (norm(Ax - z) <= prm.eps_abs + prm.eps_rel * std::max<Scalar>(norm(Ax), norm(z))) {
    // primal succeeded, check dual
    Px.noalias()            = pbm.P * x;
    Aty.noalias()           = pbm.A.transpose() * y;
    const Scalar dual_scale = std::max<Scalar>({norm(Px), norm(pbm.q), norm(Aty)});
    if (norm(Px + pbm.q + Aty) <= prm.eps_abs + prm.eps_rel * dual_scale) {
      return ExitCode::Optimal;
    }
  }

  // PRIMAL INFEASIBILITY

  Aty.noalias()         = pbm.A.transpose() * dy;  // note new value A' * dy
  const Scalar Edy_norm = norm(dy);

  Scalar u_dyp_plus_l_dyn = Scalar(0);
  for (auto i = 0u; i != m; ++i) {
    if (pbm.u(i) != inf) {
      u_dyp_plus_l_dyn += pbm.u(i) * std::max<Scalar>(Scalar(0), dy(i));
    } else if (dy(i) > prm.eps_primal_inf * Edy_norm) {
      // contributes +inf to sum --> no certificate
      u_dyp_plus_l_dyn = inf;
      break;
    }
    if (pbm.l(i) != -inf) {
      u_dyp_plus_l_dyn += pbm.l(i) * std::min<Scalar>(Scalar(0), dy(i));
    } else if (dy(i) < -prm.eps_primal_inf * Edy_norm) {
      // contributes +inf to sum --> no certificate
      u_dyp_plus_l_dyn = inf;
      break;
    }
  }

  if (std::max<Scalar>(norm(Aty), u_dyp_plus_l_dyn) < prm.eps_primal_inf * Edy_norm) {
    return ExitCode::PrimalInfeasible;
  }

  // DUAL INFEASIBILITY

  Ax.noalias()         = pbm.A * dx;  // note new value A * dx
  const Scalar dx_norm = norm(dx);

  bool dual_infeasible = (norm(pbm.P * dx) <= prm.eps_dual_inf * dx_norm)
                      && (pbm.q.dot(dx) <= prm.eps_dual_inf * dx_norm);
  for (auto i = 0u; i != m && dual_infeasible; ++i) {
    if (pbm.u(i) == inf) {
      dual_infeasible &= (Ax(i) >= -prm.eps_dual_inf * dx_norm);
    } else if (pbm.l(i) == -inf) {
      dual_infeasible &= (Ax(i) <= prm.eps_dual_inf * dx_norm);
    } else {
      dual_infeasible &= std::abs(Ax(i)) < prm.eps_dual_inf * dx_norm;
    }
  }

  if (dual_infeasible) { return ExitCode::DualInfeasible; }

  return std::nullopt;
}

}  // namespace detail

/**
 * @brief Solve a quadratic program using the operator splitting approach.
 *
 * @tparam Pbm problem type (QuadraticProgram or QuadraticProgramSparse)
 *
 * @param pbm problem formulation
 * @param prm solver options
 * @param warmstart provide initial guess for primal and dual variables
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
detail::QpSol_t<Pbm> solve_qp(const Pbm & pbm,
  const SolverParams & prm,
  std::optional<std::reference_wrapper<const detail::QpSol_t<Pbm>>> warmstart = {})
{
  using AmatT                  = decltype(Pbm::A);
  using Scalar                 = typename AmatT::Scalar;
  static constexpr bool sparse = std::is_base_of_v<Eigen::SparseMatrixBase<AmatT>, AmatT>;

  // static sizes
  static constexpr Eigen::Index M = AmatT::RowsAtCompileTime;
  static constexpr Eigen::Index N = AmatT::ColsAtCompileTime;
  static constexpr Eigen::Index K = (N == -1 || M == -1) ? Eigen::Index(-1) : N + M;

  // factorization method
  using LDLT =
    std::conditional_t<sparse, detail::LDLTSparse<Scalar>, detail::LDLTLapack<Scalar, K>>;

  // typedefs
  using Rn = Eigen::Matrix<Scalar, N, 1>;
  using Rm = Eigen::Matrix<Scalar, M, 1>;
  using Rk = Eigen::Matrix<Scalar, K, 1>;

  // dynamic sizes
  const Eigen::Index n = pbm.A.cols(), m = pbm.A.rows(), k = n + m;

  static constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();

  // cast parameters to scalar type
  const Scalar rho        = static_cast<Scalar>(prm.rho);
  const Scalar alpha      = static_cast<Scalar>(prm.alpha);
  const Scalar alpha_comp = Scalar(1) - alpha;
  const Scalar sigma      = static_cast<Scalar>(prm.sigma);

  // return code: when set algorithm is finished
  std::optional<ExitCode> ret_code = std::nullopt;

  // scale problem
  const auto [spbm, S, c] = detail::scale_qp(pbm);

  if ((spbm.u - spbm.l).minCoeff() < Scalar(0.) || spbm.l.maxCoeff() == inf
      || spbm.u.minCoeff() == -inf) {
    // feasible set is trivially empty
    ret_code = ExitCode::PrimalInfeasible;
  }

  // fill square symmetric system matrix H
  std::conditional_t<sparse, Eigen::SparseMatrix<Scalar>, Eigen::Matrix<Scalar, K, K>> H(k, k);
  if constexpr (sparse) {
    // preallocate nonzeros in H
    Eigen::Matrix<int, -1, 1> nnz(k);
    for (auto i = 0u; i != n; ++i) {
      nnz(i) = spbm.P.outerIndexPtr()[i + 1] - spbm.P.outerIndexPtr()[i] + 1;
    }
    for (auto i = 0u; i != m; ++i) {
      nnz(n + i) = spbm.A.outerIndexPtr()[i + 1] - spbm.A.outerIndexPtr()[i] + 1;
    }
    H.reserve(nnz);

    // fill nonzeros in H
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
  LDLT ldlt(std::move(H));
  if (ldlt.info()) { ret_code = ExitCode::Unknown; }

  // initialize solver variables
  Rn x;
  Rm z, y;
  if (warmstart.has_value()) {
    // warmstart variables must be scaled
    x = warmstart.value().get().primal;
    x.applyOnTheLeft(S.template head<N>(n).cwiseInverse().asDiagonal());
    y = warmstart.value().get().dual;
    y.applyOnTheLeft(S.template segment<M>(n, m).cwiseInverse().asDiagonal());
    y *= c;
    z.noalias() = spbm.A * x;
  } else {
    x.setZero(n);
    y.setZero(m);
    z.setZero(m);
  }

  // allocate working arrays
  Rn x_us(n), dx_us(n);
  Rm z_next(m), y_us(m), z_us(m), dy_us(m);
  Rk p(k);

  // main optimization loop
  for (auto iter = 0u; iter != prm.max_iter && !ret_code; ++iter) {
    p.template head<N>(n)       = sigma * x - spbm.q;
    p.template segment<M>(n, m) = z - y / rho;
    ldlt.solve_inplace(p);

    if (iter % prm.stop_check_iter == prm.stop_check_iter - 1) {
      // termination checking requires difference, store old scaled values
      dx_us = x, dy_us = y;
    }

    x      = alpha * p.template head<N>(n) + alpha_comp * x;
    z_next = ((alpha / rho) * p.template segment<M>(n, m) + (alpha_comp / rho) * y + z)
               .cwiseMax(spbm.l)
               .cwiseMin(spbm.u);
    y = alpha_comp * y + alpha * p.template segment<M>(n, m) + rho * z - rho * z_next;
    z = z_next;

    if (iter % prm.stop_check_iter == prm.stop_check_iter - 1) {
      // check stopping criteria for unscaled problem and unscaled variables
      x_us     = S.template head<N>(n).cwiseProduct(x);
      y_us     = S.template segment<M>(n, m).cwiseProduct(y) / c;
      z_us     = S.template segment<M>(n, m).cwiseInverse().cwiseProduct(z);
      dx_us    = S.template head<N>(n).cwiseProduct(x - dx_us);
      dy_us    = S.template segment<M>(n, m).cwiseProduct(y - dy_us) / c;
      ret_code = detail::qp_check_stopping(pbm, x_us, y_us, z_us, dx_us, dy_us, prm);
    }
  }

  detail::QpSol_t<Pbm> sol{.code = ret_code.value_or(ExitCode::MaxIterations),
    .primal                      = std::move(x),
    .dual                        = std::move(y)};

  // polish solution
  if (sol.code == ExitCode::Optimal && prm.polish) {
    if (!detail::polish_qp(spbm, sol, prm)) { sol.code = ExitCode::PolishFailed; }
  }

  // unscale solution
  sol.primal.applyOnTheLeft(S.template head<N>(n).asDiagonal());
  sol.dual.applyOnTheLeft(S.template segment<M>(n, m).asDiagonal());
  sol.dual /= c;

  // TODO print solver summary

  return sol;
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__QP_HPP_
