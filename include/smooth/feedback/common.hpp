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

#ifndef SMOOTH__FEEDBACK__COMMON_HPP_
#define SMOOTH__FEEDBACK__COMMON_HPP_

#include "smooth/manifold.hpp"

namespace smooth::feedback {

/**
 * @brief Manifold constraint set.
 *
 * The set consists of all \f$ m \f$ s.t.
 * \f[
 *   l \leq  A ( m \ominus c ) \leq u.
 * \f]
 *
 * @note The number of constraints is hard-coded so that \f$ A \f$ is a square matrix.
 */
template<Manifold M>
struct ManifoldBounds
{
  /// Dimensionality
  static constexpr int N = Dof<M>;

  /// Number of constraints
  static constexpr int NumCon = N;

  /// Transformation matrix
  Eigen::Matrix<double, NumCon, N> A = Eigen::Matrix<double, NumCon, N>::Identity(NumCon, N);
  /// Nominal point in constraint set.
  M c;
  /// Lower bound
  Eigen::Matrix<double, NumCon, 1> l =
    Eigen::Matrix<double, NumCon, 1>::Constant(-std::numeric_limits<double>::infinity());
  /// Upper bound
  Eigen::Matrix<double, NumCon, 1> u =
    Eigen::Matrix<double, NumCon, 1>::Constant(std::numeric_limits<double>::infinity());
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COMMON_HPP_
