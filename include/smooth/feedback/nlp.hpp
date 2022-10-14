// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Nonlinear program definition.
 */

#include <limits>
#include <optional>

#include <Eigen/Core>
#include <Eigen/Sparse>

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
template<typename T>
concept NLP = requires(
  std::decay_t<T> & nlp, const Eigen::Ref<const Eigen::VectorXd> x, const Eigen::Ref<const Eigen::VectorXd> lambda)
{
  // clang-format off
  {nlp.n()} -> std::convertible_to<std::size_t>;
  {nlp.m()} -> std::convertible_to<std::size_t>;

  // variable bounds
  {nlp.xl()} -> std::convertible_to<Eigen::VectorXd>;
  {nlp.xu()} -> std::convertible_to<Eigen::VectorXd>;

  // objective and its derivatives
  {nlp.f(x)}     -> std::convertible_to<double>;
  {nlp.df_dx(x)} -> std::convertible_to<Eigen::SparseMatrix<double>>;

  // constraint, its bounds, and its derivatives
  {nlp.g(x)}     -> std::convertible_to<Eigen::VectorXd>;
  {nlp.gl()}     -> std::convertible_to<Eigen::VectorXd>;
  {nlp.gu()}     -> std::convertible_to<Eigen::VectorXd>;
  {nlp.dg_dx(x)} -> std::convertible_to<Eigen::SparseMatrix<double>>;
  // clang-format on
};

/**
 * @brief Nonlinear Programming Problem with Hessian information
 */
template<typename T>
concept HessianNLP = NLP<T> && requires(std::decay_t<T> & nlp, Eigen::VectorXd x, Eigen::VectorXd lambda)
{
  // clang-format off
  {nlp.d2f_dx2(x)}         -> std::convertible_to<Eigen::SparseMatrix<double>>;
  {nlp.d2g_dx2(x, lambda)} -> std::convertible_to<Eigen::SparseMatrix<double>>;
  // clang-format on
};

/**
 * @brief Solution to a Nonlinear Programming Problem
 */
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
  };

  /// @brief Solver status
  Status status;

  /// @brief Number of iterations
  std::size_t iter{0};

  /// @brief Variable values
  Eigen::VectorXd x;

  ///@{
  /// @brief Inequality multipliers
  Eigen::VectorXd zl, zu;
  ///@}

  /// @brief Constraint multipliers
  Eigen::VectorXd lambda;

  /// @brief Objective
  double objective{0};
};

}  // namespace smooth::feedback
