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

#include <gtest/gtest.h>
#include <smooth/se3.hpp>

#include "smooth/feedback/utils/dr_exp_sparse.hpp"

template<typename G>
smooth::Tangent<G> random_tangent(int i)
{
  if (i == 0) {
    return smooth::Tangent<G>::Zero();
  } else {
    return smooth::Tangent<G>::Random();
  }
};

TEST(DEXP, Rn)
{
  using G = Eigen::Vector4d;

  for (auto i = 0u; i < 5; ++i) {
    const smooth::Tangent<G> a = random_tangent<G>(i);

    Eigen::SparseMatrix<double> calc1(4, 4);
    smooth::feedback::dr_exp_sparse<G>(calc1, a);
    ASSERT_EQ(calc1.nonZeros(), 4);

    const auto calc2 = smooth::dr_exp<G>(a);

    ASSERT_TRUE(calc1.isApprox(calc2));
  }
}

TEST(DEXP, SE2)
{
  using G = smooth::SE2d;

  for (auto i = 0u; i < 5; ++i) {
    const smooth::Tangent<G> a = random_tangent<G>(i);

    Eigen::SparseMatrix<double> calc1(3, 3);
    smooth::feedback::dr_exp_sparse<G>(calc1, a);
    ASSERT_EQ(calc1.nonZeros(), 7);

    const auto calc2 = smooth::dr_exp<G>(a);

    ASSERT_TRUE(calc1.isApprox(calc2));
  }
}

TEST(DEXP, SE3)
{
  using G = smooth::SE3d;

  for (auto i = 0u; i < 5; ++i) {
    const smooth::Tangent<G> a = random_tangent<G>(i);

    Eigen::SparseMatrix<double> calc1(6, 6);
    smooth::feedback::dr_exp_sparse<G>(calc1, a);

    const auto calc2 = smooth::dr_exp<G>(a);

    ASSERT_TRUE(calc1.isApprox(calc2));
  }
}

TEST(DEXP, Bundle)
{
  using G = smooth::Bundle<smooth::SE2d, Eigen::Vector2d, smooth::SE2d>;

  for (auto i = 0u; i < 5; ++i) {
    const smooth::Tangent<G> a = random_tangent<G>(i);

    Eigen::SparseMatrix<double> calc1(8, 8);
    smooth::feedback::dr_exp_sparse<G>(calc1, a);
    ASSERT_EQ(calc1.nonZeros(), 16);

    const auto calc2 = smooth::dr_exp<G>(a);

    ASSERT_TRUE(calc1.isApprox(calc2));
  }
}

TEST(DEXPINV, Rn)
{
  using G = Eigen::Vector4d;

  for (auto i = 0u; i < 5; ++i) {
    const smooth::Tangent<G> a = random_tangent<G>(i);

    Eigen::SparseMatrix<double> calc1(4, 4);
    smooth::feedback::dr_expinv_sparse<G>(calc1, a);
    ASSERT_EQ(calc1.nonZeros(), 4);

    const auto calc2 = smooth::dr_expinv<G>(a);

    ASSERT_TRUE(calc1.isApprox(calc2));
  }
}

TEST(DEXPINV, SE2)
{
  using G = smooth::SE2d;

  for (auto i = 0u; i < 5; ++i) {
    const smooth::Tangent<G> a = random_tangent<G>(i);

    Eigen::SparseMatrix<double> calc1(3, 3);
    smooth::feedback::dr_expinv_sparse<G>(calc1, a);
    ASSERT_EQ(calc1.nonZeros(), 7);

    const auto calc2 = smooth::dr_expinv<G>(a);

    ASSERT_TRUE(calc1.isApprox(calc2));
  }
}

TEST(DEXPINV, SE3)
{
  using G = smooth::SE3d;

  for (auto i = 0u; i < 5; ++i) {
    const smooth::Tangent<G> a = random_tangent<G>(i);

    Eigen::SparseMatrix<double> calc1(6, 6);
    smooth::feedback::dr_expinv_sparse<G>(calc1, a);

    const auto calc2 = smooth::dr_expinv<G>(a);

    ASSERT_TRUE(calc1.isApprox(calc2));
  }
}

TEST(DEXPINV, Bundle)
{
  using G = smooth::Bundle<smooth::SE2d, Eigen::Vector2d, smooth::SE2d>;

  for (auto i = 0u; i < 5; ++i) {
    const smooth::Tangent<G> a = random_tangent<G>(i);

    Eigen::SparseMatrix<double> calc1(8, 8);
    smooth::feedback::dr_expinv_sparse<G>(calc1, a);
    ASSERT_EQ(calc1.nonZeros(), 16);

    const auto calc2 = smooth::dr_expinv<G>(a);

    ASSERT_TRUE(calc1.isApprox(calc2));
  }
}
