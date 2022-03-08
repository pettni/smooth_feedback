// smooth_feedback: Control theory on Lie groups
// https://github.com/pettni/smooth_feedback
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson, John B. Mains
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

#ifndef SMOOTH__FEEDBACK__QP_SOLVER_HPP_
#define SMOOTH__FEEDBACK__QP_SOLVER_HPP_

/**
 * @file
 * @brief Quadratic Program solver.
 */

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>

#include "qp.hpp"
#include "utils/sparse.hpp"

namespace smooth::feedback {

/**
 * @brief Options for solve_qp
 */
struct QPSolverParams
{
  /// print solver info to stdout
  bool verbose = false;

  /// relaxation parameter
  float alpha = 1.6;
  /// first dul step size
  float rho = 0.1;
  /// second dual step length
  float sigma = 1e-6;

  /// scale problem
  bool scaling = true;

  /// absolute threshold for convergence
  float eps_abs = 1e-3;
  /// relative threshold for convergence
  float eps_rel = 1e-3;
  /// threshold for primal infeasibility
  float eps_primal_inf = 1e-4;
  /// threshold for dual infeasibility
  float eps_dual_inf = 1e-4;

  /// max number of iterations
  std::optional<uint32_t> max_iter = {};

  /// max solution time
  std::optional<std::chrono::nanoseconds> max_time = {};

  /// iterations between checking stopping criterion
  uint32_t stop_check_iter = 25;

  /// run solution polishing (uses dynamic memory)
  bool polish = true;
  /// number of iterations to refine polish
  uint32_t polish_iter = 5;
  /// regularization parameter for polishing
  float delta = 1e-6;
};

namespace detail {

template<typename Pbm>
using qp_solution_t = QPSolution<
  decltype(Pbm::A)::RowsAtCompileTime,
  decltype(Pbm::A)::ColsAtCompileTime,
  typename decltype(Pbm::A)::Scalar>;

/**
 * @brief Polish solution of quadratic program
 *
 * @tparam Pbm problem type
 *
 * @param[in] pbm problem formulation
 * @param[in, out] sol solution to polish
 * @param[in] prm solver options
 * @param[in] c cost scaling
 * @param[in] sx variable scaling
 * @param[in] sy constraint scaling
 *
 * @warning This function allocates heap memory even for static-sized problems.
 */
template<typename Pbm, typename D1, typename D2>
bool polish_qp(
  const Pbm & pbm,
  qp_solution_t<Pbm> & sol,
  const QPSolverParams & prm,
  const typename decltype(Pbm::A)::Scalar c,
  const Eigen::MatrixBase<D1> & sx,
  const Eigen::MatrixBase<D2> & sy)
{
  using AmatT                  = decltype(Pbm::A);
  using Scalar                 = typename AmatT::Scalar;
  static constexpr bool sparse = std::is_base_of_v<Eigen::SparseMatrixBase<AmatT>, AmatT>;

  static constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
  static constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

  static constexpr Eigen::Index N = AmatT::ColsAtCompileTime;
  const Eigen::Index n = pbm.A.cols(), m = pbm.A.rows();

  // FIND ACTIVE CONSTRAINT SETS

  Eigen::Index nl = 0, nu = 0;
  for (Eigen::Index idx = 0; idx < m; ++idx) {
    if (sol.dual[idx] < -100 * eps && pbm.l[idx] != -inf) { nl++; }
    if (sol.dual[idx] > 100 * eps && pbm.u[idx] != inf) { nu++; }
  }

  Eigen::VectorXi LU_idx(nl + nu);
  for (Eigen::Index idx = 0, lcntr = 0, ucntr = 0; idx < m; ++idx) {
    if (sol.dual[idx] < -100 * eps && pbm.l[idx] != -inf) { LU_idx(lcntr++) = idx; }
    if (sol.dual[idx] > 100 * eps && pbm.u[idx] != inf) { LU_idx(nl + ucntr++) = idx; }
  }

  // FORM REDUCED SYSTEMS (27) AND (30)

  // square symmetric system matrix
  using HT = std::conditional_t<sparse, Eigen::SparseMatrix<Scalar>, Eigen::Matrix<Scalar, -1, -1>>;
  HT H(n + nl + nu, n + nl + nu), Hp(n + nl + nu, n + nl + nu);

  // fill up H
  if constexpr (sparse) {
    // preallocate nonzeros
    Eigen::VectorXi nnz(n + nl + nu);
    for (auto i = 0u; i != n; ++i) {
      nnz(i) = pbm.P.outerIndexPtr()[i + 1] - pbm.P.outerIndexPtr()[i];
    }
    for (auto i = 0u; i != nl + nu; ++i) {
      nnz(n + i) = pbm.A.outerIndexPtr()[LU_idx(i) + 1] - pbm.A.outerIndexPtr()[LU_idx(i)];
    }
    H.reserve(nnz);
    Hp.reserve(nnz + Eigen::VectorXi::Ones(n + nl + nu));

    // fill P in top left block
    for (auto k = 0u; k < n; ++k) {
      for (Eigen::InnerIterator it(pbm.P, k); it; ++it) {
        const Scalar pij              = c * sx(it.col()) * sx(it.row()) * it.value();
        H.insert(it.row(), it.col())  = pij;
        Hp.insert(it.row(), it.col()) = pij;
      }
    }

    // fill selected rows of A in top right block
    for (auto a_row = 0u; a_row != nl + nu; ++a_row) {
      for (Eigen::InnerIterator it(pbm.A, LU_idx(a_row)); it; ++it) {
        const Scalar Aij               = sy(it.row()) * sx(it.col()) * it.value();
        H.insert(it.col(), n + a_row)  = Aij;
        Hp.insert(it.col(), n + a_row) = Aij;
      }
    }
  } else {
    H.setZero();
    H.topLeftCorner(n, n) = c * sx.asDiagonal() * pbm.P * sx.asDiagonal();
    for (auto i = 0u; i != nl + nu; ++i) {
      H.col(n + i).template head<N>(n) = sy(LU_idx(i)) * pbm.A.row(LU_idx(i)) * sx.asDiagonal();
    }
    Hp = H;
  }

  // add perturbing diagonal elements to Hp
  if constexpr (sparse) {
    for (auto i = 0u; i != n; ++i) { Hp.coeffRef(i, i) += prm.delta; }
    for (auto i = 0u; i != nl + nu; ++i) { Hp.coeffRef(n + i, n + i) -= prm.delta; }
    H.makeCompressed();
    Hp.makeCompressed();
  } else {
    Hp.topLeftCorner(n, n) += Eigen::VectorX<Scalar>::Constant(n, prm.delta).asDiagonal();
    Hp.bottomRightCorner(nl + nu, nl + nu) -=
      Eigen::VectorX<Scalar>::Constant(nl + nu, prm.delta).asDiagonal();
  }

  Eigen::VectorX<Scalar> h(n + nl + nu);
  h.head(n) = -c * sx.cwiseProduct(pbm.q);
  for (auto i = 0u; i != nl; ++i) { h(n + i) = sy(LU_idx(i)) * pbm.l(LU_idx(i)); }
  for (auto i = 0u; i != nu; ++i) { h(n + nl + i) = sy(LU_idx(nl + i)) * pbm.u(LU_idx(nl + i)); }

  // ITERATIVE REFINEMENT

  // factorize Hp
  std::conditional_t<
    sparse,
    Eigen::SimplicialLDLT<decltype(H), Eigen::Upper>,
    Eigen::LDLT<decltype(H), Eigen::Upper>>
    ldlt(Hp);

  if (ldlt.info()) { return false; }

  Eigen::VectorX<Scalar> t_hat = Eigen::VectorX<Scalar>::Zero(n + nl + nu);
  for (auto i = 0u; i != prm.polish_iter; ++i) {
    t_hat += ldlt.solve(h - H.template selfadjointView<Eigen::Upper>() * t_hat);
  }

  // UPDATE SOLUTION

  sol.primal = t_hat.template head<N>(n);
  for (auto i = 0u; i < nl; ++i) { sol.dual(LU_idx(i)) = t_hat(n + i); }
  for (auto i = 0u; i < nu; ++i) { sol.dual(LU_idx(nl + i)) = t_hat(n + nl + i); }

  return true;
}

}  // namespace detail

/**
 * @brief Solver for quadratic programs
 *
 * Use this class to efficiently solve many QPs with the same problem structure.
 *
 * For one-off QPs, see solve_qp().
 */
template<typename Pbm>
class QPSolver
{
  using AmatT                  = decltype(Pbm::A);
  using Scalar                 = typename AmatT::Scalar;
  static constexpr bool sparse = std::is_base_of_v<Eigen::SparseMatrixBase<AmatT>, AmatT>;

  // static sizes
  static constexpr Eigen::Index M = AmatT::RowsAtCompileTime;
  static constexpr Eigen::Index N = AmatT::ColsAtCompileTime;
  static constexpr Eigen::Index K = (N == -1 || M == -1) ? Eigen::Index(-1) : N + M;

  // typedefs
  using Rn = Eigen::Vector<Scalar, N>;
  using Rm = Eigen::Vector<Scalar, M>;
  using Rk = Eigen::Vector<Scalar, K>;
  using Ht = std::conditional_t<sparse, Eigen::SparseMatrix<Scalar>, Eigen::Matrix<Scalar, K, K>>;
  using LDLTt = std::
    conditional_t<sparse, Eigen::SimplicialLDLT<Ht, Eigen::Upper>, Eigen::LDLT<Ht, Eigen::Upper>>;

  static inline const Scalar inf = std::numeric_limits<Scalar>::infinity();

public:
  /**
   * @brief Default constructor.
   */
  QPSolver(const QPSolverParams & prm = {}) : prm_(prm) {}

  /**
   * @brief Construct and allocate working memory.
   *
   * @param pbm template problem.
   *
   * Memory is allocated for solving problems with same structure as pbm.
   */
  QPSolver(const Pbm & pbm, const QPSolverParams & prm = {}) : prm_(prm) { analyze(pbm); }

  /**
   * @brief Access most recent QP solution.
   */
  const QPSolution<M, N, Scalar> & sol() const { return sol_; }

  /**
   * @brief Prepare for solving problems.
   */
  void analyze(const Pbm & pbm)
  {
    const Eigen::Index n = pbm.A.cols(), m = pbm.A.rows(), k = n + m;

    // solution
    sol_.primal.setZero(n);
    sol_.dual.setZero(m);

    // scaling variables and working memory
    c_ = 1;
    sx_.setOnes(n);
    sy_.setOnes(m);
    sx_inc_.setZero(n);
    sy_inc_.setZero(m);

    // solve working memory
    z_.resize(m);
    z_next_.resize(m);
    rho_.resize(m);
    p_.resize(k);

    // stopping working memory
    x_us_.resize(n);
    dx_us_.resize(n);
    y_us_.resize(m);
    dy_us_.resize(m);
    z_us_.resize(m);
    Px_.resize(n);
    Aty_.resize(n);
    Ax_.resize(m);

    // preallocate nonzeros in H
    H_.resize(k, k);
    if constexpr (sparse) {
      Eigen::VectorXi nnz(k);
      for (auto i = 0u; i < n; ++i) {
        nnz(i) = pbm.P.outerIndexPtr()[i + 1] - pbm.P.outerIndexPtr()[i] + 1;
      }
      for (auto i = 0u; i < m; ++i) {
        nnz(n + i) = pbm.A.outerIndexPtr()[i + 1] - pbm.A.outerIndexPtr()[i] + 1;
      }
      H_.reserve(nnz);

      // fill nonzeros in H
      for (auto i = 0u; i < pbm.P.outerSize(); ++i) {
        for (Eigen::InnerIterator it(pbm.P, i); it; ++it) {
          if (it.col() >= it.row()) { H_.insert(it.row(), it.col()) = Scalar(1.); }
        }
      }
      block_add_identity(H_, 0, 0, n);
      for (auto i = 0u; i < pbm.A.outerSize(); ++i) {
        for (Eigen::InnerIterator it(pbm.A, i); it; ++it) {
          H_.insert(it.col(), n + it.row()) = Scalar(1.);
        }
      }
      for (auto row = 0u; row < m; ++row) { H_.insert(n + row, n + row) = Scalar(1); }

      H_.makeCompressed();
      ldlt_.analyzePattern(H_);
    }
  }

  /**
   * @brief Solve quadratic program.
   */
  const QPSolution<M, N, Scalar> & solve(
    const Pbm & pbm,
    std::optional<std::reference_wrapper<const QPSolution<M, N, Scalar>>> warmstart = {})
  {
    // update problem scaling
    if (prm_.scaling) { scale(pbm); }

    // dynamic sizes
    const Eigen::Index n = pbm.A.cols(), m = pbm.A.rows();

    // cast parameters to scalar type
    const Scalar rho_bar    = static_cast<Scalar>(prm_.rho);
    const Scalar alpha      = static_cast<Scalar>(prm_.alpha);
    const Scalar alpha_comp = Scalar(1) - alpha;
    const Scalar sigma      = static_cast<Scalar>(prm_.sigma);

    // return code: when set algorithm is finished
    std::optional<QPSolutionStatus> ret_code = std::nullopt;

    for (auto i = 0u; i != m; ++i) {
      if (pbm.l(i) == inf || pbm.u(i) == -inf || pbm.u(i) - pbm.l(i) < Scalar(0.)) {
        ret_code = QPSolutionStatus::PrimalInfeasible;  // feasible set trivially empty
      }

      // set rho depending on constraint type
      if (pbm.l(i) == -inf && pbm.u(i) == inf) {
        rho_(i) = Scalar(1e-6);  // unbounded
      } else if (sy_(i) * abs(pbm.l(i) - pbm.u(i)) < 1e-5) {
        rho_(i) = Scalar(1e3) * rho_bar;  // equality
      } else {
        rho_(i) = rho_bar;  // inequality
      }
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    // fill square symmetric system matrix H = [P A'; A 0]
    if constexpr (sparse) {
      assert(H_.isCompressed());  // pattern set in analyze()
      H_.coeffs().setZero();

      for (auto i = 0u; i < pbm.P.outerSize(); ++i) {
        for (Eigen::InnerIterator it(pbm.P, i); it; ++it) {
          if (it.col() >= it.row()) {
            H_.coeffRef(it.row(), it.col()) = c_ * sx_(it.row()) * sx_(it.col()) * it.value();
          }
        }
      }
      block_add_identity(H_, 0, 0, n, sigma);
      for (auto i = 0u; i < pbm.A.outerSize(); ++i) {
        for (Eigen::InnerIterator it(pbm.A, i); it; ++it) {
          H_.coeffRef(it.col(), n + it.row()) = sy_(it.row()) * sx_(it.col()) * it.value();
        }
      }
      for (auto row = 0u; row < m; ++row) {
        H_.coeffRef(n + row, n + row) = Scalar(-1) / rho_(row);
      }

      assert(H_.isCompressed());  // pattern set in analyze()
    } else {
      H_.setZero();

      H_.template topLeftCorner<N, N>(n, n) = c_ * sx_.asDiagonal() * pbm.P * sx_.asDiagonal();
      H_.template topLeftCorner<N, N>(n, n) +=
        Eigen::Vector<Scalar, N>::Constant(n, sigma).asDiagonal();
      H_.template topRightCorner<N, M>(n, m) =
        (sy_.asDiagonal() * pbm.A * sx_.asDiagonal()).transpose();
      H_.template bottomRightCorner<M, M>(m, m) = (-rho_).cwiseInverse().asDiagonal();
    }

    const auto t_fill = std::chrono::high_resolution_clock::now();

    if (prm_.verbose) {
      using std::cout, std::left, std::setw, std::right;
      // clang-format off
      cout << "========================= QP Solver =========================" << '\n';
      cout << "Solving " << (sparse ? "sparse" : "dense") << " QP with n=" << n << ", m=" << m << '\n';
      cout << setw(8)  << right << "ITER"
           << setw(14) << right << "OBJ"
           << setw(14) << right << "PRI_RES"
           << setw(14) << right << "DUA_RES"
           << setw(10) << right << "TIME" << '\n';
      // clang-format on
    }

    // factorize H
    if constexpr (sparse) {
      ldlt_.factorize(H_);
    } else {
      ldlt_.compute(H_);
    }

    const auto t_factor = std::chrono::high_resolution_clock::now();

    if (ldlt_.info()) { ret_code = QPSolutionStatus::Unknown; }

    // initialize solver variables
    if (warmstart.has_value()) {
      // warmstart variables must be scaled
      sol_.primal  = sx_.cwiseInverse().cwiseProduct(warmstart.value().get().primal);
      sol_.dual    = c_ * sy_.cwiseInverse().cwiseProduct(warmstart.value().get().dual);
      z_.noalias() = sy_.asDiagonal() * pbm.A * warmstart.value().get().primal;
    } else {
      sol_.primal.setZero();
      sol_.dual.setZero();
      z_.setZero();
    }

    // main optimization loop
    auto iter = 0u;
    for (; (!prm_.max_iter || iter != prm_.max_iter.value()) && !ret_code; ++iter) {
      p_.template segment<N>(0, n) = sigma * sol_.primal - c_ * sx_.asDiagonal() * pbm.q;
      p_.template segment<M>(n, m) = z_ - rho_.cwiseInverse().cwiseProduct(sol_.dual);
      if constexpr (sparse) {
        p_ = ldlt_.solve(p_);
      } else {
        ldlt_.solveInPlace(p_);
      }

      if (iter % prm_.stop_check_iter == 1) {
        // termination checking requires difference, store old scaled values
        dx_us_ = sol_.primal, dy_us_ = sol_.dual;
      }

      sol_.primal = alpha * p_.template segment<N>(0, n) + alpha_comp * sol_.primal;
      z_next_     = (alpha * rho_.cwiseInverse().cwiseProduct(p_.template segment<M>(n, m))
                 + alpha_comp * rho_.cwiseInverse().cwiseProduct(sol_.dual) + z_)
                  .cwiseMax(sy_.cwiseProduct(pbm.l))
                  .cwiseMin(sy_.cwiseProduct(pbm.u));
      sol_.dual = alpha_comp * sol_.dual + alpha * p_.template segment<M>(n, m)
                + rho_.cwiseProduct(z_) - rho_.cwiseProduct(z_next_);
      std::swap(z_, z_next_);

      if (iter % prm_.stop_check_iter == 1) {
        // unscale solution
        x_us_  = sx_.cwiseProduct(sol_.primal);
        y_us_  = sy_.cwiseProduct(sol_.dual) / c_;
        z_us_  = sy_.cwiseInverse().cwiseProduct(z_);
        dx_us_ = sx_.cwiseProduct(sol_.primal - dx_us_);
        dy_us_ = sy_.cwiseProduct(sol_.dual - dy_us_) / c_;

        // check stopping criteria for unscaled problem and unscaled variables
        ret_code = check_stopping(pbm);

        if (prm_.verbose) {
          using std::cout, std::setw, std::right, std::chrono::microseconds;
          // clang-format off
          cout << setw(7) << right << iter << ":"
            << std::scientific
            << setw(14) << right << (0.5 * pbm.P * x_us_ + pbm.q).dot(x_us_)
            << setw(14) << right << (pbm.A * x_us_ - z_us_).template lpNorm<Eigen::Infinity>()
            << setw(14) << right << (pbm.P * x_us_ + pbm.q + pbm.A.transpose() * y_us_).template lpNorm<Eigen::Infinity>()
            << setw(10) << right << duration_cast<microseconds>(std::chrono::high_resolution_clock::now() - t0).count()
            << '\n';
          // clang-format on
        }

        // check for timeout
        if (!ret_code) {
          if (
            prm_.max_time
            && std::chrono::high_resolution_clock::now() > t0 + prm_.max_time.value()) {
            ret_code = QPSolutionStatus::MaxTime;
          }
        }
      }
    }

    const auto t_iter = std::chrono::high_resolution_clock::now();

    // polish solution if optimal
    if (ret_code.has_value() && ret_code.value() == QPSolutionStatus::Optimal && prm_.polish) {
      if (detail::polish_qp(pbm, sol_, prm_, c_, sx_, sy_)) {
        if (prm_.verbose) {
          // unscale solution
          x_us_ = sx_.cwiseProduct(sol_.primal);
          y_us_ = sy_.cwiseProduct(sol_.dual) / c_;
          z_us_ = sy_.cwiseInverse().cwiseProduct(z_);

          using std::cout, std::setw, std::right, std::chrono::microseconds;
          // clang-format off
          cout << setw(8) << right << "polish:"
            << std::scientific
            << setw(14) << right << (0.5 * pbm.P * x_us_ + pbm.q).dot(x_us_)
            << setw(14) << right << (pbm.A * x_us_ - z_us_).template lpNorm<Eigen::Infinity>()
            << setw(14) << right << (pbm.P * x_us_ + pbm.q + pbm.A.transpose() * y_us_).template lpNorm<Eigen::Infinity>()
            << setw(10) << right << duration_cast<microseconds>(std::chrono::high_resolution_clock::now() - t0).count()
            << '\n';
          // clang-format on
        }

      } else {
        if (prm_.verbose) { std::cout << "Polish failed" << '\n'; }
        sol_.code = QPSolutionStatus::PolishFailed;
      }
    }

    const auto t_polish = std::chrono::high_resolution_clock::now();

    // unscale solution
    sol_.code      = ret_code.value_or(QPSolutionStatus::MaxIterations);
    sol_.primal    = sx_.cwiseProduct(sol_.primal);
    sol_.dual      = sy_.cwiseProduct(sol_.dual) / c_;
    sol_.objective = sol_.primal.dot(0.5 * pbm.P * sol_.primal + pbm.q);
    sol_.iter      = iter;

    if (prm_.verbose) {
      using std::cout, std::left, std::right, std::setw, std::chrono::microseconds;

      // clang-format off
      cout << "QP solver summary:" << '\n';
      cout << "Result " << static_cast<int>(sol_.code) << '\n';

      cout << setw(25) << left << "Iterations"        << setw(10) << right << iter - 1                                               << '\n';
      cout << setw(26) << left << "Total time (µs)"   << setw(10) << right << duration_cast<microseconds>(t_polish - t0).count()     << '\n';
      cout << setw(25) << left << "  Matrix filling"  << setw(10) << right << duration_cast<microseconds>(t_fill - t0).count()       << '\n';
      cout << setw(25) << left << "  Factorization"   << setw(10) << right << duration_cast<microseconds>(t_factor - t_fill).count() << '\n';
      cout << setw(25) << left << "  Iteration"       << setw(10) << right << duration_cast<microseconds>(t_iter - t_factor).count() << '\n';
      cout << setw(25) << left << "  Polish"          << setw(10) << right << duration_cast<microseconds>(t_polish - t_iter).count() << '\n';
      cout << "=============================================================" << '\n';
      // clang-format on
    }

    return sol_;
  }

protected:
  /**
   * @brief Check stopping criteria for solver.
   */
  std::optional<QPSolutionStatus> check_stopping(const Pbm & pbm)
  {
    const Eigen::Index m = pbm.A.rows();

    // norm function
    static const auto norm = [](auto && t) -> Scalar {
      return t.template lpNorm<Eigen::Infinity>();
    };

    // OPTIMALITY

    // check primal
    Ax_.noalias()        = pbm.A * x_us_;
    const Scalar Ax_norm = norm(Ax_);
    Ax_ -= z_us_;
    if (norm(Ax_) <= prm_.eps_abs + prm_.eps_rel * std::max<Scalar>(Ax_norm, norm(z_us_))) {
      // primal succeeded, check dual
      Px_.noalias()           = pbm.P * x_us_;
      Aty_.noalias()          = pbm.A.transpose() * y_us_;
      const Scalar dual_scale = std::max<Scalar>({norm(Px_), norm(pbm.q), norm(Aty_)});
      Px_ += pbm.q + Aty_;
      if (norm(Px_) <= prm_.eps_abs + prm_.eps_rel * dual_scale) {
        return QPSolutionStatus::Optimal;
      }
    }

    // PRIMAL INFEASIBILITY

    Aty_.noalias()        = pbm.A.transpose() * dy_us_;  // NOTE new value A' * dy
    const Scalar Edy_norm = norm(dy_us_);

    Scalar u_dyp_plus_l_dyn = Scalar(0);
    for (auto i = 0u; i != m; ++i) {
      if (pbm.u(i) != inf) {
        u_dyp_plus_l_dyn += pbm.u(i) * std::max<Scalar>(Scalar(0), dy_us_(i));
      } else if (dy_us_(i) > prm_.eps_primal_inf * Edy_norm) {
        // contributes +inf to sum --> no certificate
        u_dyp_plus_l_dyn = inf;
        break;
      }
      if (pbm.l(i) != -inf) {
        u_dyp_plus_l_dyn += pbm.l(i) * std::min<Scalar>(Scalar(0), dy_us_(i));
      } else if (dy_us_(i) < -prm_.eps_primal_inf * Edy_norm) {
        // contributes +inf to sum --> no certificate
        u_dyp_plus_l_dyn = inf;
        break;
      }
    }

    if (std::max<Scalar>(norm(Aty_), u_dyp_plus_l_dyn) < prm_.eps_primal_inf * Edy_norm) {
      return QPSolutionStatus::PrimalInfeasible;
    }

    // DUAL INFEASIBILITY

    Ax_.noalias()        = pbm.A * dx_us_;  // note new value A * dx
    const Scalar dx_norm = norm(dx_us_);
    Px_.noalias()        = pbm.P * dx_us_;

    bool dual_infeasible = (norm(Px_) <= prm_.eps_dual_inf * dx_norm)
                        && (pbm.q.dot(dx_us_) <= prm_.eps_dual_inf * dx_norm);
    for (auto i = 0u; i != m && dual_infeasible; ++i) {
      if (pbm.u(i) == inf) {
        dual_infeasible &= (Ax_(i) >= -prm_.eps_dual_inf * dx_norm);
      } else if (pbm.l(i) == -inf) {
        dual_infeasible &= (Ax_(i) <= prm_.eps_dual_inf * dx_norm);
      } else {
        dual_infeasible &= std::abs(Ax_(i)) < prm_.eps_dual_inf * dx_norm;
      }
    }

    if (dual_infeasible) { return QPSolutionStatus::DualInfeasible; }

    return std::nullopt;
  }

  /**
   * @brief Re-scale QP.
   *
   * The scaled problem is defined as
   *
   * * \f$ P_s = c S_x P S_x \f$,
   * * \f$ q_s = c q S_x \f$,
   * * \f$ A_s = S_y A S_x \f$,
   * * \f$ l_s = S_y l \f$,
   * * \f$ u_s = S_y u \f$,
   *
   * where Sx = diag(sx), Sy = diag(sy).
   *
   * The relation between scaled variables and original variables are
   *
   * * Primal: \f$ x_s = S_x^{-1} x \f$,
   * * Dual: \f$ y_s = c S_y^{-1} y \f$.
   *
   * The objective of the rescaling is the make the columns of
   * \f[
   *   \begin{bmatrix} \bar P & \bar A^T \\ \bar A & 0 \end{bmatrix}
   * \f]
   * have similar \f$ l_\infty \f$ norm, and similarly for the columns of
   * \f[
   *  \begin{bmatrix} \bar P & \bar q \end{bmatrix}.
   * \f]
   */
  void scale(const Pbm & pbm)
  {
    sx_.setOnes();
    sy_.setOnes();

    sx_inc_.setZero();

    // find "norm" of cost function
    for (auto i = 0u; i < pbm.P.outerSize(); ++i) {
      for (Eigen::InnerIterator it(pbm.P, i); it; ++it) {
        sx_inc_(it.col()) = std::max(sx_inc_(it.col()), std::abs(it.value()));
      }
    }

    // if there are "zero cols"
    for (auto i = 0u; i != sx_.size(); ++i) {
      if (sx_inc_(i) == 0) { sx_inc_(i) = 1; }
    }

    // scale cost function
    c_ = Scalar(1) / std::max({1e-6, sx_inc_.mean(), pbm.q.template lpNorm<Eigen::Infinity>()});

    int iter = 0;

    // calculate inf-norm for every column of [Ps As' ; As 0]
    do {
      sx_inc_.setZero();
      sy_inc_.setZero();
      for (auto k = 0u; k < pbm.P.outerSize(); ++k) {
        for (Eigen::InnerIterator it(pbm.P, k); it; ++it) {
          // upper left block of H
          sx_inc_(it.col()) = std::max({
            sx_inc_(it.col()),
            std::abs(c_ * sx_(it.row()) * sx_(it.col()) * it.value()),
          });
        }
      }
      for (auto k = 0u; k < pbm.A.outerSize(); ++k) {
        for (Eigen::InnerIterator it(pbm.A, k); it; ++it) {
          const Scalar Aij  = std::abs(sy_(it.row()) * sx_(it.col()) * it.value());
          sx_inc_(it.col()) = std::max(sx_inc_(it.col()), Aij);  // bottom left block of H
          sy_inc_(it.row()) = std::max(sy_inc_(it.row()), Aij);  // upper right block of H
        }
      }

      // if there are "zero cols" we don't scale
      for (auto i = 0u; i < sx_.size(); ++i) {
        if (sx_inc_(i) == 0) { sx_inc_(i) = 1; }
      }
      for (auto i = 0u; i < sy_.size(); ++i) {
        if (sy_inc_(i) == 0) { sy_inc_(i) = 1; }
      }

      sx_.applyOnTheLeft(sx_inc_.cwiseMax(1e-8).cwiseInverse().cwiseSqrt().asDiagonal());
      sy_.applyOnTheLeft(sy_inc_.cwiseMax(1e-8).cwiseInverse().cwiseSqrt().asDiagonal());
    } while (
      iter++ < 10
      && std::max((sx_inc_.array() - 1).abs().maxCoeff(), (sy_inc_.array() - 1).abs().maxCoeff())
           > 0.1);
  }

private:
  // solver parameters
  QPSolverParams prm_;

  // solution
  QPSolution<M, N, Scalar> sol_;

  // scaling variables and working memory
  Scalar c_;
  Rn sx_, sx_inc_;
  Rm sy_, sy_inc_;

  // solve working memory
  Rm z_, z_next_, rho_;
  Rk p_;

  // stopping working memory
  Rn x_us_, dx_us_;
  Rn Px_, Aty_;
  Rm Ax_;
  Rm y_us_, z_us_, dy_us_;

  // system matrix and decomposition
  Ht H_;
  LDLTt ldlt_;
};

/**
 * @brief Solve a quadratic program using the operator splitting approach.
 *
 * @tparam Pbm problem type (QuadraticProgram or QuadraticProgramSparse)
 *
 * @param pbm problem formulation
 * @param prm solver options
 * @param warmstart provide initial guess for primal and dual variables
 * @return solution as QuasraticProgramSolution<M, N>
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
detail::qp_solution_t<Pbm> solve_qp(
  const Pbm & pbm,
  const QPSolverParams & prm,
  std::optional<std::reference_wrapper<const detail::qp_solution_t<Pbm>>> warmstart = {})
{
  QPSolver<Pbm> solver(pbm, prm);
  return solver.solve(pbm, warmstart);
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__QP_SOLVER_HPP_
