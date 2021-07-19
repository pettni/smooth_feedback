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

namespace smooth::feedback {

/**
 * @brief Quadratic program definition.
 *
 * The quadratic program is on the form
 * \f[
 * \begin{cases}
 *  \min_{x} & \frac{1}{2} x^T P x + q^T x, \\
 *  \text{s.t.} & l \leq A x \leq u.
 * \end{cases}
 * \f]
 */
template<Eigen::Index nvar, Eigen::Index ncon>
struct QuadraticProgram
{
  /// Quadratic cost
  Eigen::Matrix<double, nvar, nvar> P = Eigen::Matrix<double, nvar, nvar>::Zero();
  /// Linear cost
  Eigen::Matrix<double, nvar, 1> q = Eigen::Matrix<double, nvar, 1>::Zero();

  /// Inequality matrix
  Eigen::Matrix<double, ncon, nvar> A = Eigen::Matrix<double, ncon, nvar>::Zero();
  /// Inequality lower bound
  Eigen::Matrix<double, ncon, 1> l = Eigen::Matrix<double, ncon, 1>::Zero();
  /// Inequality upper bound
  Eigen::Matrix<double, ncon, 1> u = Eigen::Matrix<double, ncon, 1>::Zero();
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__QP_HPP_
