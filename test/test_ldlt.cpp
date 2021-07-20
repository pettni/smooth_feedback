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

#include <smooth/feedback/internal/ldlt_lapack.hpp>

template<typename Scalar, std::size_t N>
void basic_test()
{
  for (auto i = 0; i != 10; ++i) {
    Eigen::Matrix<Scalar, N, N> A = Eigen::Matrix<Scalar, N, N>::Random();
    Eigen::Matrix<Scalar, N, 1> b = Eigen::Matrix<Scalar, N, 1>::Random();

    smooth::feedback::detail::LDLTLapack<Scalar, N> ldlt(A);
    auto x = ldlt.solve(b);

    ASSERT_LE(
      (A.template selfadjointView<Eigen::Upper>() * x - b).template lpNorm<Eigen::Infinity>(),
      std::sqrt(std::numeric_limits<Scalar>::epsilon()));
  }
}

TEST(Ldlt, Basic)
{
  basic_test<float, 3>();
  basic_test<double, 3>();

  basic_test<float, 10>();
  basic_test<double, 10>();

  basic_test<float, 100>();
  basic_test<double, 100>();
}
