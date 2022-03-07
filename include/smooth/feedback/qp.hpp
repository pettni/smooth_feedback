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

#ifndef SMOOTH__FEEDBACK__QP_HPP_
#define SMOOTH__FEEDBACK__QP_HPP_

/**
 * @file
 * @brief Quadratic Programming.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>

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
  /// Positive semi-definite square cost (only upper triangular part is used)
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
   * @brief Inequality matrix
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

/// @brief Solver exit codes
enum class QPSolutionStatus {
  Optimal,           /// @brief Solution satisifes optimality condition. Solution is polished if
                     /// `QPSolverParams::polish = true`.
  PolishFailed,      /// @brief Solution satisfies optimality condition but is not polished
  PrimalInfeasible,  /// @brief A certificate of primal infeasibility was found, no solution
                     /// returned
  DualInfeasible,    /// @brief A certificate of dual infeasibility was found, no solution returned
  MaxIterations,  /// @brief Max number of iterations was reached, returned solution is not optimal
  MaxTime,        /// @brief Max time was reached, returned solution is not optimal
  Unknown         /// @brief Solution is useless because of other reasons, no solution returned
};

/// Solver solution
template<Eigen::Index M, Eigen::Index N, typename Scalar = double>
struct QPSolution
{
  /// Exit code
  QPSolutionStatus code = QPSolutionStatus::Unknown;
  /// Number of iterations
  uint32_t iter;
  /// Primal vector
  Eigen::Matrix<Scalar, N, 1> primal;
  /// Dual vector
  Eigen::Matrix<Scalar, M, 1> dual;
  /// Solution objective value
  Scalar objective{0.};
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__QP_HPP_
