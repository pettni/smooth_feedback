// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

#include <smooth/concepts/manifold.hpp>

namespace smooth::feedback {

/**
 * @brief Manifold constraint set.
 *
 * The set consists of all \f$ m \f$ s.t.
 * \f[
 *   m \in M = \{ l \leq  A ( m \ominus c ) \leq u \}.
 * \f]
 */
template<Manifold M>
struct ManifoldBounds
{
  static_assert(Dof<M> > 0, "Dynamic size not supported");

  /// Transformation matrix
  Eigen::Matrix<double, -1, Dof<M>> A{Eigen::Matrix<double, -1, Dof<M>>::Zero(0, Dof<M>)};
  /// Nominal point in constraint set.
  M c{Default<M>()};
  /// Lower bound
  Eigen::VectorXd l{Eigen::VectorXd::Constant(0, -std::numeric_limits<double>::infinity())};
  /// Upper bound
  Eigen::VectorXd u{Eigen::VectorXd::Constant(0, std::numeric_limits<double>::infinity())};
};

}  // namespace smooth::feedback
