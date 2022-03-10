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

#ifndef SMOOTH__FEEDBACK__UTILS__SPARSE_HPP_
#define SMOOTH__FEEDBACK__UTILS__SPARSE_HPP_

/**
 * @file
 * @brief Sparse matrix utilities.
 */

#include <Eigen/Sparse>

#include <numeric>
#include <optional>

namespace smooth::feedback {

/**
 * @brief Write block into a sparse matrix.
 *
 * After this function the output variable dest is s.t.
 *
 * dest[row0 + r, col0 + c] += scale * source[r, c]
 *
 * @param dest destination
 * @param row0 starting row for block
 * @param col0 starting column for block
 * @param source block values
 * @param scale scaling parameter
 * @param upper_only only add into upper triangular part
 *
 * @note Values are accessed with coeffRef().
 */
template<typename Source, int Options>
  requires(std::is_base_of_v<Eigen::EigenBase<Source>, Source>)
inline void block_add(
  Eigen::SparseMatrix<double, Options> & dest,
  Eigen::Index row0,
  Eigen::Index col0,
  const Source & source,
  double scale      = 1,
  double upper_only = false)
{
  for (auto c = 0; c < source.outerSize(); ++c) {
    for (Eigen::InnerIterator it(source, c); it; ++it) {
      if (!upper_only || row0 + it.row() <= col0 + it.col()) {
        dest.coeffRef(row0 + it.row(), col0 + it.col()) += scale * it.value();
      }
    }
  }
}

/**
 * @brief Write identity matrix block into sparse matrix.
 *
 * After this function the output variable dest is s.t.
 *
 * dest[row0 + k, col0 + k] += scale, k = 0...n-1
 *
 * @param dest destination
 * @param row0 starting row for block
 * @param col0 starting column for block
 * @param n size of identity matrix
 * @param scale scaling parameter
 *
 * @note Values are accessed with coeffRef().
 */
template<int Options>
inline void block_add_identity(
  Eigen::SparseMatrix<double, Options> & dest,
  Eigen::Index row0,
  Eigen::Index col0,
  Eigen::Index n,
  double scale = 1)
{
  for (auto k = 0u; k < n; ++k) { dest.coeffRef(row0 + k, col0 + k) += scale; }
}

/**
 * @brief Zero a sparse matrix without changing allocation.
 *
 * If mat is compressed all coefficients are set to explicit zeros.
 */
template<typename Scalar, int Options>
inline void set_zero(Eigen::SparseMatrix<Scalar, Options> & mat)
{
  if (mat.isCompressed()) {
    mat.coeffs().setZero();
  } else {
    for (auto i = 0; i < mat.outerSize(); ++i) {
      for (Eigen::InnerIterator it(mat, i); it; ++it) { mat.coeffRef(it.row(), it.col()) = 0; }
    }
  }
}

/**
 * @brief Count number of explicit zeros in sparse matrix.
 *
 * @param mat sparse matrix
 */
template<typename Mat>
  requires(std::is_base_of_v<Eigen::EigenBase<Mat>, Mat>)
uint64_t count_explicit_zeros(const Mat & mat)
{
  uint64_t ret = 0;
  for (auto c = 0; c < mat.outerSize(); ++c) {
    for (Eigen::InnerIterator it(mat, c); it; ++it) {
      if (it.value() == 0) { ++ret; }
    }
  }
  return ret;
}

/**
 * @brief Mark explicit zeros in sparse matrix.
 *
 * @param mat sparse matrix
 *
 * Returns a matrix that has values as follows:
 *  - 0 for implicit zeros
 *  - 1 for non-zeros
 *  - 9 for explicit zeros
 */
template<typename Mat>
  requires(std::is_base_of_v<Eigen::EigenBase<Mat>, Mat>)
Eigen::MatrixX<typename Mat::Scalar> mark_explicit_zeros(const Mat & mat)
{
  Eigen::MatrixX<typename Mat::Scalar> ret;
  ret.setConstant(mat.rows(), mat.cols(), 0);
  for (auto c = 0; c < mat.outerSize(); ++c) {
    for (Eigen::InnerIterator it(mat, c); it; ++it) {
      if (it.value() == 0) {
        ret(it.row(), it.col()) = 9;
      } else {
        ret(it.row(), it.col()) = 1;
      }
    }
  }
  return ret;
}

/**
 * @brief (Right) Hessian of composed function \f$ (f \circ g)(x) \f$.
 *
 * @param[out] out result                           [No x No*Nx]
 * @param[in] Jf (Right) Jacobian of f at y = g(x)  [No x Ny   ]
 * @param[in] Hf (Right) Hessian of f at y = g(x)   [Ny x No*Ny]
 * @param[in] Jg (Right) Jacobian of g at x         [Ny x Nx   ]
 * @param[in] Hg (Right) Hessian of g at x          [Nx x Ny*Nx]
 * @param[in] r0 row to insert result
 * @param[in] r0 col to insert result
 *
 * @note out must have appropriate size
 */
template<typename S1, typename S2, typename S3, typename S4>
  requires(
    std::is_base_of_v<Eigen::EigenBase<S1>, S1> && std::is_base_of_v<Eigen::EigenBase<S2>, S2> &&
      std::is_base_of_v<Eigen::EigenBase<S3>, S3> && std::is_base_of_v<Eigen::EigenBase<S4>, S4>)
inline void d2r_fog(
  Eigen::SparseMatrix<double> & out,
  const S1 & Jf,
  const S2 & Hf,
  const S3 & Jg,
  const S4 & Hg,
  Eigen::Index r0 = 0,
  Eigen::Index c0 = 0)
{
  const auto No = Jf.rows();
  const auto Ny = Jf.cols();

  [[maybe_unused]] const auto Ni = Jg.rows();
  const auto Nx                  = Jg.cols();

  // check some dimensions
  assert(Ny == Ni);
  assert(Hf.rows() == Ny);
  assert(Hf.cols() == No * Ny);
  assert(Hg.rows() == Nx);
  assert(Hg.cols() == Ni * Nx);

  for (auto no = 0u; no < No; ++no) {
    // TODO sparse-sparse-sparse product is expensive and allocates temporary
    block_add(out, r0, c0 + no * Nx, Jg.transpose() * Hf.middleCols(no * Ny, Ny) * Jg);
  }

  for (auto i = 0u; i < Jf.outerSize(); ++i) {
    for (Eigen::InnerIterator it(Jf, i); it; ++it) {
      block_add(out, r0, c0 + it.row() * Nx, Hg.middleCols(it.col() * Nx, Nx), it.value());
    }
  }
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__UTILS__SPARSE_HPP_
