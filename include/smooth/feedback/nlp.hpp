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

#ifndef SMOOTH__FEEDBACK__NLP_HPP_
#define SMOOTH__FEEDBACK__NLP_HPP_

/**
 * @file
 * @brief Nonlinear program definition.
 */

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <smooth/diff.hpp>

#include <limits>

#include "collocation.hpp"

namespace smooth::feedback {

/**
 * @brief Nonlinear Programming Problem
 * \f[
 *  \begin{cases}
 *   \min_{x}    & f(x)                    \\
 *   \text{s.t.} & x_l \leq x \leq x_u     \\
 *               & g_l \leq g(x) \leq g_u
 *  \end{cases}
 * \f]
 * for \f$ f : \mathbb{R}^n \rightarrow \mathbb{R} \f$ and
 * \f$ g : \mathbb{R}^n \rightarrow \mathbb{R}^m \f$.
 */
struct NLP
{
  /// @brief Number of variables
  std::size_t n;

  /// @brief Number of constraints
  std::size_t m;

  /// @brief Objective function (R^n -> R)
  std::function<double(Eigen::VectorXd)> f;

  /// @brief Variable bounds (R^n)
  Eigen::VectorXd xl, xu;

  /// @brief Constraint function (R^n -> R^m)
  std::function<Eigen::VectorXd(Eigen::VectorXd)> g;

  /// @brief Constaint bounds (R^m)
  Eigen::VectorXd gl, gu;

  /// @brief Jacobian of objective function (R^n -> R^{n x n})
  std::function<Eigen::SparseMatrix<double>(Eigen::VectorXd)> df_dx;

  /// @brief Jacobian of constraint function (R^n -> R^{m x n})
  std::function<Eigen::SparseMatrix<double>(Eigen::VectorXd)> dg_dx;

  /// @brief Hessian of objective function (R^n -> R^{n x n}) [optional]
  std::optional<std::function<Eigen::SparseMatrix<double>(Eigen::VectorXd, Eigen::VectorXd)>>
    d2f_dx2 = std::nullopt;

  /**
   * @brief Projected Hessian of constraint function (R^m, R^n -> R^{n x n}) [optional]
   *
   * Should return the derivative
   * \f[
   *  H_g(\lambda, x) = \nabla^2_x \lambda^T g(x), \quad \lambda \in \mathbb{R}^m, x \in
   * \mathbb{R}^n \f]
   */
  std::optional<std::function<Eigen::SparseMatrix<double>(Eigen::VectorXd, Eigen::VectorXd)>>
    d2g_dx2 = std::nullopt;
};

struct NLPSolution
{
  /// @brief Solver status
  enum class Status {
    Optimal,
    PrimalInfeasible,
    DualInfeasible,
    MaxIterations,
    MaxTime,
    Unknown,
  } status;

  /// @brief Number of iterations
  std::size_t iter{0};

  /// @brief Variable values
  Eigen::VectorXd x;

  /// @brief Inequality multipliers
  Eigen::VectorXd zl, zu;

  /// @brief Constraint multipliers
  Eigen::VectorXd lambda;

  /// @brief Objective
  double objective{0};
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__NLP_HPP_
