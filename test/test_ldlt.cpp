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
#include <smooth/feedback/internal/ldlt_sparse.hpp>

template<typename Scalar, std::size_t N>
void dense_test()
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
  srand(42);

  dense_test<float, 3>();
  dense_test<double, 3>();

  dense_test<float, 10>();
  dense_test<double, 10>();

  dense_test<float, 100>();
  dense_test<double, 100>();
}

void sparse_test(int size)
{
  for (auto i = 0; i != 10; ++i) {
    Eigen::MatrixXd A_dense = Eigen::MatrixXd::Random(size, size);
    A_dense.topRightCorner(size/2, size/2).setZero();
    Eigen::SparseMatrix<double, Eigen::ColMajor, long> A(size, size);
    A = A_dense.sparseView();

    Eigen::VectorXd b = Eigen::VectorXd::Random(size);

    smooth::feedback::detail::LDLTSparse ldlt(A);
    auto x = ldlt.solve(b);

    ASSERT_LE(
      (A.template selfadjointView<Eigen::Upper>() * x - b).template lpNorm<Eigen::Infinity>(),
      std::sqrt(std::numeric_limits<double>::epsilon()));
  }
}

TEST(LdltSparse, Basic)
{
  srand(42);
  sparse_test(3);
  sparse_test(10);
  sparse_test(100);
}
