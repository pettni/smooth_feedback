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

#ifndef SMOOTH__FEEDBACK__UTILS__D2R_EXP_SPARSE_HPP_
#define SMOOTH__FEEDBACK__UTILS__D2R_EXP_SPARSE_HPP_

/**
 * @file
 * @brief Sparse second derivatives of the exponential maps.
 *
 * \todo Prevent allocation in Bundle version
 * \todo Implement for all standard Lie groups
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

namespace detail {

template<typename Scalar>
void d2r_exp_sparse_se2(Eigen::SparseMatrix<Scalar> & out, const Eigen::Vector3<Scalar> & a)
{
  const auto & vx = a.x();
  const auto & vy = a.y();
  const auto & th = a.z();

  const Scalar th2 = th * th;
  const Scalar th3 = th2 * th;
  const Scalar th4 = th2 * th2;

  const auto [A, B, dA, dB] = [&]() -> std::array<Scalar, 4> {
    if (th2 < Scalar(eps2)) {
      return {
        Scalar(1) / Scalar(2) - th2 / Scalar(24),
        Scalar(1) / Scalar(6) - th2 / Scalar(120),
        -th / Scalar(12),
        -th / Scalar(60),
      };
    } else {
      return {
        (Scalar(1) - cos(th)) / th2,
        (th - sin(th)) / (th2 * th),
        sin(th) / th2 - Scalar(2) * (Scalar(1) - cos(th)) / th3,
        (Scalar(1) - cos(th)) / th3 - Scalar(3) * (th - sin(th)) / th4,
      };
    }
  }();

  out.resize(3, 9);

  if (out.isCompressed()) {
    out.coeffs().setZero();
  } else {
    out.setZero();
    out.reserve(Eigen::VectorXi{{1, 1, 3, 1, 1, 3, 0, 0, 0}});
  }

  // first row
  out.coeffRef(2, 3 * 0 + 0) = B * th;
  out.coeffRef(2, 3 * 0 + 1) = -A;
  out.coeffRef(0, 3 * 0 + 2) = -Scalar(2) * B * th - dB * th2;
  out.coeffRef(1, 3 * 0 + 2) = A + dA * th;
  out.coeffRef(2, 3 * 0 + 2) = -dA * vy + B * vx + dB * th * vx;

  // second row
  out.coeffRef(2, 3 * 1 + 0) = A;
  out.coeffRef(2, 3 * 1 + 1) = out.coeff(2, 0);
  out.coeffRef(0, 3 * 1 + 2) = -out.coeff(1, 2);
  out.coeffRef(1, 3 * 1 + 2) = out.coeff(0, 2);
  out.coeffRef(2, 3 * 1 + 2) = dA * vx + B * vy + dB * th * vy;

  out.makeCompressed();
}

template<typename Scalar>
void d2r_expinv_sparse_se2(Eigen::SparseMatrix<Scalar> & out, const Eigen::Vector3<Scalar> & a)
{
  const auto & vx = a.x();
  const auto & vy = a.y();
  const auto & th = a.z();

  const Scalar th2 = th * th;
  const Scalar th3 = th2 * th;

  const auto [A, dA] = [&]() -> std::array<Scalar, 2> {
    if (th2 < Scalar(eps2)) {
      return {
        Scalar(1) / Scalar(12) + th2 / Scalar(720),
        th / Scalar(360),
      };
    } else {
      const Scalar q  = Scalar(2) * th * sin(th);
      const Scalar dq = Scalar(2) * (sin(th) + th * cos(th));
      return {
        (Scalar(1) / th2) - (Scalar(1) + cos(th)) / q,
        (-Scalar(2) / th3) + sin(th) / q + (Scalar(1) + cos(th)) / (q * q) * dq,
      };
    }
  }();

  out.resize(3, 9);

  if (out.isCompressed()) {
    out.coeffs().setZero();
  } else {
    out.setZero();
    out.reserve(Eigen::VectorXi{{1, 1, 3, 1, 1, 3, 0, 0, 0}});
  }

  // first row
  out.coeffRef(2, 3 * 0 + 0) = A * th;
  out.coeffRef(2, 3 * 0 + 1) = Scalar(0.5);
  out.coeffRef(0, 3 * 0 + 2) = -dA * th2 - A * Scalar(2) * th;
  out.coeffRef(1, 3 * 0 + 2) = -Scalar(0.5);
  out.coeffRef(2, 3 * 0 + 2) = dA * th * vx + A * vx;

  // second row
  out.coeffRef(2, 3 * 1 + 0) = -Scalar(0.5);
  out.coeffRef(2, 3 * 1 + 1) = out.coeff(2, 0);
  out.coeffRef(0, 3 * 1 + 2) = Scalar(0.5);
  out.coeffRef(1, 3 * 1 + 2) = out.coeff(0, 2);
  out.coeffRef(2, 3 * 1 + 2) = dA * th * vy + A * vy;

  out.makeCompressed();
}

template<typename G, bool Inv>
  requires(std::is_base_of_v<BundleBase<G>, G>)
void d2r_exp_sparse_bundle(Eigen::SparseMatrix<Scalar<G>> & out, const Tangent<G> & a)
{
  static constexpr auto BundleSize = G::BundleSize;
  std::size_t cntr                 = 0;
  utils::static_for<BundleSize>([&](auto I) {
    using GI = typename G::template PartType<I>;
    Eigen::SparseMatrix<Scalar<G>> tmp;
    d2r_exp_impl<GI, Inv>(tmp, a.segment(cntr, Dof<GI>));
    for (auto j = 0u; j < Dof<GI>; ++j) {
      block_add(out, cntr, Dof<G> * (cntr + j) + cntr, tmp.middleCols(Dof<GI> * j, Dof<GI>));
    }
    cntr += Dof<GI>;
  });
  out.makeCompressed();
}

template<LieGroup G, bool Inv>
void d2r_exp_impl(Eigen::SparseMatrix<Scalar<G>> & out, const Tangent<G> & a)
{
  out.resize(a.size(), a.size() * a.size());

  if constexpr (smooth::traits::RnType<G> || smooth::traits::ScalarType<G>) {
    out.setZero();
  } else if constexpr (std::is_base_of_v<smooth::SE2Base<G>, G>) {
    if constexpr (Inv == false) {
      d2r_exp_sparse_se2<Scalar<G>>(out, a);
    } else {
      d2r_expinv_sparse_se2<Scalar<G>>(out, a);
    }
  } else if constexpr (std::is_base_of_v<smooth::BundleBase<G>, G>) {
    d2r_exp_sparse_bundle<G, Inv>(out, a);
  } else {
    if (out.isCompressed()) {
      out.coeffs().setZero();
    } else {
      out.setZero();
      out.reserve(Eigen::VectorXi::Constant(a.size() * a.size(), a.size()));
    }
    if constexpr (Inv == false) {
      block_add(out, 0, 0, smooth::d2r_exp<G>(a));
    } else {
      block_add(out, 0, 0, smooth::d2r_expinv<G>(a));
    }
    out.makeCompressed();
  }
}

}  // namespace detail

/**
 * @brief Second derivative of the exponential map.
 *
 * @param[out] out result
 * @param[in] a tangent element
 *
 * On return out contains the derivative
 * \f[
 *   \mathrm{d}^{2r} \left( \exp(a) \right)_{aa}
 *   = \mathrm{d}^{r} \left( \mathrm{d}^r \exp_a^{T} \right)_{a},
 * \f]
 * on horizontal-block Hessian form.
 */
template<LieGroup G>
void d2r_exp_sparse(Eigen::SparseMatrix<Scalar<G>> & out, const Tangent<G> & a)
{
  detail::d2r_exp_impl<G, false>(out, a);
}

/**
 * @brief Second derivative of the inverse of the exponential map.
 *
 * On return out contains the derivative
 * \f[
 *   \mathrm{d}^{r} \left( \mathrm{d}^r \exp_a^{-T} \right)_{a},
 * \f]
 * on horizontal-block Hessian form.
 */
template<LieGroup G>
void d2r_expinv_sparse(Eigen::SparseMatrix<Scalar<G>> & out, const Tangent<G> & a)
{
  detail::d2r_exp_impl<G, true>(out, a);
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__UTILS__D2R_EXP_SPARSE_HPP_
