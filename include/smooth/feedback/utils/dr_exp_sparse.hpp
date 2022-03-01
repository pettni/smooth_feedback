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

#ifndef SMOOTH__FEEDBACK__UTILS__DR_EXP_SPARSE_HPP_
#define SMOOTH__FEEDBACK__UTILS__DR_EXP_SPARSE_HPP_

/**
 * @file
 * @brief Sparse derivatives of the exponential maps.
 *
 * @todo Implement for all standard Lie groups
 */

#include <Eigen/Sparse>
#include <smooth/algo/hessian.hpp>
#include <smooth/bundle.hpp>
#include <smooth/lie_group.hpp>
#include <smooth/se2.hpp>

#include <numeric>
#include <optional>

#include "sparse.hpp"

namespace smooth::feedback {

// \cond
namespace detail {

template<typename Scalar>
void dr_exp_sparse_se2(
  Eigen::SparseMatrix<Scalar> & out,
  Eigen::Index r0,
  Eigen::Index c0,
  const Eigen::Vector3<Scalar> & a)
{
  if (!out.isCompressed()) {
    Eigen::VectorXi pattern = Eigen::VectorXi::Zero(out.cols());
    pattern(c0)             = 2;
    pattern(c0 + 1)         = 2;
    pattern(c0 + 2)         = 3;
    out.reserve(pattern);
  }

  const auto ret_dense = dr_exp<SE2<Scalar>>(a);
  for (auto i = 0u; i < 2; ++i) {
    for (auto j = 0u; j < 3; ++j) { out.coeffRef(r0 + i, c0 + j) = ret_dense(i, j); }
  }
  out.coeffRef(r0 + 2, c0 + 2) = 1;
}

template<typename Scalar>
void dr_expinv_sparse_se2(
  Eigen::SparseMatrix<Scalar> & out,
  Eigen::Index r0,
  Eigen::Index c0,
  const Eigen::Vector3<Scalar> & a)
{
  if (!out.isCompressed()) {
    Eigen::VectorXi pattern = Eigen::VectorXi::Zero(out.cols());
    pattern(c0)             = 2;
    pattern(c0 + 1)         = 2;
    pattern(c0 + 2)         = 3;
    out.reserve(pattern);
  }

  const auto ret_dense = dr_expinv<SE2<Scalar>>(a);
  for (auto i = 0u; i < 2; ++i) {
    for (auto j = 0u; j < 3; ++j) { out.coeffRef(r0 + i, c0 + j) = ret_dense(i, j); }
  }
  out.coeffRef(r0 + 2, c0 + 2) = 1;
}

template<typename G, bool Inv>
  requires(std::is_base_of_v<BundleBase<G>, G>)
void dr_exp_sparse_bundle(
  Eigen::SparseMatrix<Scalar<G>> & out, Eigen::Index r0, Eigen::Index c0, const Tangent<G> & a)
{
  static constexpr auto BundleSize = G::BundleSize;

  std::size_t cntr = 0;
  utils::static_for<BundleSize>([&](auto I) {
    using GI = typename G::template PartType<I>;
    dr_exp_impl<GI, Inv>(out, r0 + cntr, c0 + cntr, a.segment(cntr, Dof<GI>));
    cntr += Dof<GI>;
  });
}

template<LieGroup G, bool Inv>
void dr_exp_impl(
  Eigen::SparseMatrix<Scalar<G>> & out, Eigen::Index r0, Eigen::Index c0, const Tangent<G> & a)
{
  if constexpr (smooth::traits::RnType<G> || smooth::traits::ScalarType<G>) {
    for (auto i = 0u; i < a.size(); ++i) { out.coeffRef(r0 + i, c0 + i) = 1; }
  } else if constexpr (std::is_base_of_v<smooth::SE2Base<G>, G>) {
    if constexpr (Inv == false) {
      dr_exp_sparse_se2<Scalar<G>>(out, r0, c0, a);
    } else {
      dr_expinv_sparse_se2<Scalar<G>>(out, r0, c0, a);
    }
  } else if constexpr (std::is_base_of_v<smooth::BundleBase<G>, G>) {
    dr_exp_sparse_bundle<G, Inv>(out, r0, c0, a);
  } else {
    if (!out.isCompressed()) {
      Eigen::VectorXi pattern = Eigen::VectorXi::Zero(out.cols());
      pattern.segment(c0, a.size()).setConstant(a.size());
      out.reserve(pattern);
    }
    if constexpr (Inv) {
      const auto A = dr_expinv<G>(a);
      for (auto i = 0u; i < a.size(); ++i) {
        for (auto j = 0u; j < a.size(); ++j) { out.coeffRef(r0 + i, c0 + j) = A(i, j); }
      }
    } else {
      const auto A = dr_exp<G>(a);
      for (auto i = 0u; i < a.size(); ++i) {
        for (auto j = 0u; j < a.size(); ++j) { out.coeffRef(r0 + i, c0 + j) = A(i, j); }
      }
    }
  }

  out.makeCompressed();
}

}  // namespace detail
// \endcond

/**
 * @brief Derivative of the exponential map.
 *
 * @param[out] out output matrix
 * @param[in] a tangent element
 * @param[in] r0 row to insert result
 * @param[in] c0 column to insert result
 *
 * On return out.block(r0, c0, Dof, Dof) contains the derivative
 * \f[
 *   \mathrm{d}^{r} \exp_{a}.
 * \f]
 */
template<LieGroup G>
void dr_exp_sparse(
  Eigen::SparseMatrix<Scalar<G>> & out,
  const Tangent<G> & a,
  Eigen::Index r0 = 0,
  Eigen::Index c0 = 0)
{
  detail::dr_exp_impl<G, false>(out, r0, c0, a);
}

/**
 * @brief Derivative of the inverse of the exponential map.
 *
 * @param[out] out output matrix
 * @param[in] a tangent element
 * @param[in] r0 row to insert result
 * @param[in] c0 column to insert result
 *
 * On return out.block(r0, c0, Dof, Dof) contains the derivative
 * \f[
 *   \mathrm{d}^r \exp_a^{-1}.
 * \f]
 */
template<LieGroup G>
void dr_expinv_sparse(
  Eigen::SparseMatrix<Scalar<G>> & out,
  const Tangent<G> & a,
  Eigen::Index r0 = 0,
  Eigen::Index c0 = 0)
{
  detail::dr_exp_impl<G, true>(out, r0, c0, a);
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__UTILS__DR_EXP_SPARSE_HPP_
