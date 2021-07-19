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

#ifndef SMOOTH__FEEDBACK__ASIF_HPP_
#define SMOOTH__FEEDBACK__ASIF_HPP_

/**
 * @file
 * @brief Active Set Invariance Filtering (ASIF) on Lie groups.
 */

#include <Eigen/Core>

#include <smooth/concepts.hpp>
#include <smooth/diff.hpp>

#include "qp.hpp"

namespace smooth::feedback {

/**
 * @brief Get row-wise sparsity pattern for the ASIF qp matrix.
 */
template<std::size_t K, LieGroup G, LieGroup U, typename SafeSet>
Eigen::VectorXi asif_to_qp_sparsity(SafeSet && h)
{
  static constexpr int nx = G::SizeAtCompileTime;
  static constexpr int nu = U::SizeAtCompileTime;
  static constexpr int nh = std::invoke_result_t<SafeSet, G>::SizeAtCompileTime;

  Eigen::Matrix<std::size_t, -1, 1> qp_spA_sparsity(nu + K * nh, 1);
  qp_spA_sparsity.head(nu + K * nh).setConstant(nu + 1);
  qp_spA_sparsity.tail(nu).setOnes();

  return qp_spA_sparsity;
}

/**
 * @brief Pose active set invariance problem as a QuadraticProgram.
 *
 * @tparam K number of constraints.
 * @tparam G state Lie group type \f$\mathbb{G}\f$
 * @tparam U input Lie group type \f$\mathbb{G}\f$
 *
 * @param f system model \f$f : \mathbb{G} \times \mathbb{U} \rightarrow \mathbb{R}^{\dim \mathfrak
 * g}\f$ s.t. \f$ \mathrm{d}^r x_t = f(x, u) \f$
 * @param h safe set \f$h : \mathbb{G} \rightarrow \mathbb{R}^{n_h}\f$ s.t. \f$ h(x) \geq 0 \f$
 * denote safe set
 * @param ub backup controller \f$ub : \mathbb{G} \rightarrow \mathbb{U} \f$
 */
template<std::size_t K, LieGroup G, LieGroup U, typename Dyn, typename SafeSet, typename BackupU>
auto asif_to_qp(Dyn && f, SafeSet && h, BackupU && bu)
{
  using std::placeholders::_1;

  static constexpr int nx = G::SizeAtCompileTime;
  static constexpr int nu = U::SizeAtCompileTime;
  static constexpr int nh = std::invoke_result_t<SafeSet, G>::SizeAtCompileTime;

  static constexpr int ncon = K;
  static constexpr int nvar = ncon * nh + nu;

  QuadraticProgram<nvar, ncon> ret;
  return ret;
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__ASIF_HPP_
