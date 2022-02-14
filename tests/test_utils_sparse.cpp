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

#include "smooth/feedback/utils/sparse.hpp"

TEST(Utils, sbmat)
{
  Eigen::SparseMatrix<double> s11(3, 4);

  Eigen::SparseMatrix<double> s21(1, 4);
  Eigen::SparseMatrix<double> s22(1, 2);

  s11.insert(0, 1) = 1;
  s11.insert(0, 2) = 2;
  s21.insert(0, 3) = 3;
  s22.insert(0, 0) = 4;

  auto ret = smooth::feedback::sparse_block_matrix({{s11, {}}, {s21, s22}});

  ASSERT_EQ(ret.nonZeros(), 4);
  ASSERT_EQ(ret.coeff(0, 1), 1);
  ASSERT_EQ(ret.coeff(0, 2), 2);
  ASSERT_EQ(ret.coeff(3, 3), 3);
  ASSERT_EQ(ret.coeff(3, 4), 4);
}

TEST(Utils, KronDense)
{
  Eigen::MatrixXd s(2, 3);
  s << 1, 2, 3, 4, 5, 6;

  auto res = smooth::feedback::kron_identity(s, 3);

  ASSERT_EQ(res.rows(), 6);
  ASSERT_EQ(res.cols(), 9);
  ASSERT_EQ(res.nonZeros(), 3 * 6);

  ASSERT_EQ(res.coeff(0, 0), 1);
  ASSERT_EQ(res.coeff(1, 1), 1);
  ASSERT_EQ(res.coeff(2, 2), 1);

  ASSERT_EQ(res.coeff(0, 6), 3);
  ASSERT_EQ(res.coeff(1, 7), 3);
  ASSERT_EQ(res.coeff(2, 8), 3);

  ASSERT_EQ(res.coeff(3, 3), 5);
  ASSERT_EQ(res.coeff(4, 4), 5);
  ASSERT_EQ(res.coeff(5, 5), 5);
}

TEST(Utils, KronColmajor)
{
  Eigen::SparseMatrix<double> s(2, 3);

  s.insert(0, 0) = 1;
  s.insert(0, 2) = 3;
  s.insert(1, 0) = 5;

  s.makeCompressed();

  auto res1 = smooth::feedback::kron_identity(s, 3);

  ASSERT_EQ(res1.rows(), 6);
  ASSERT_EQ(res1.cols(), 9);
  ASSERT_EQ(res1.nonZeros(), 9);

  ASSERT_EQ(res1.coeff(0, 0), 1);
  ASSERT_EQ(res1.coeff(1, 1), 1);
  ASSERT_EQ(res1.coeff(2, 2), 1);

  ASSERT_EQ(res1.coeff(0, 6), 3);
  ASSERT_EQ(res1.coeff(1, 7), 3);
  ASSERT_EQ(res1.coeff(2, 8), 3);

  ASSERT_EQ(res1.coeff(3, 0), 5);
  ASSERT_EQ(res1.coeff(4, 1), 5);
  ASSERT_EQ(res1.coeff(5, 2), 5);

  auto res2 = smooth::feedback::kron_identity(s.transpose(), 3);

  ASSERT_TRUE(res2.IsRowMajor);

  ASSERT_TRUE(res2.isApprox(res1.transpose()));
}

TEST(Utils, KronRowmajor)
{
  Eigen::SparseMatrix<double, Eigen::RowMajor> s(2, 3);

  s.insert(0, 0) = 1;
  s.insert(0, 2) = 3;
  s.insert(1, 0) = 5;

  s.makeCompressed();

  auto res1 = smooth::feedback::kron_identity(s, 3);

  ASSERT_TRUE(res1.IsRowMajor);

  ASSERT_EQ(res1.rows(), 6);
  ASSERT_EQ(res1.cols(), 9);

  ASSERT_EQ(res1.nonZeros(), 9);

  ASSERT_EQ(res1.coeff(0, 0), 1);
  ASSERT_EQ(res1.coeff(1, 1), 1);
  ASSERT_EQ(res1.coeff(2, 2), 1);

  ASSERT_EQ(res1.coeff(0, 6), 3);
  ASSERT_EQ(res1.coeff(1, 7), 3);
  ASSERT_EQ(res1.coeff(2, 8), 3);

  ASSERT_EQ(res1.coeff(3, 0), 5);
  ASSERT_EQ(res1.coeff(4, 1), 5);
  ASSERT_EQ(res1.coeff(5, 2), 5);

  auto res2 = smooth::feedback::kron_identity(s.transpose(), 3);
  ASSERT_TRUE(res2.isApprox(res1.transpose()));
}

TEST(Utils, BlockCopy)
{
  const Eigen::MatrixXd source1 = Eigen::MatrixXd::Random(5, 10);

  Eigen::MatrixXd source2(5, 10);
  source2.setRandom();
  const Eigen::SparseMatrix<double> source2_sp = source2.sparseView();

  Eigen::SparseMatrix<double> dest(10, 10);

  smooth::feedback::block_add(dest, 0, 0, source1.leftCols(5));
  smooth::feedback::block_add(dest, 5, 5, source2_sp.rightCols(5));
  smooth::feedback::block_add(dest, 0, 5, source2_sp.rightCols(5).transpose());

  Eigen::MatrixXd dest_d(dest);

  ASSERT_TRUE(dest_d.topLeftCorner(5, 5).isApprox(source1.leftCols(5)));
  ASSERT_TRUE(dest_d.topRightCorner(5, 5).isApprox(source2.rightCols(5).transpose()));
  ASSERT_TRUE(dest_d.bottomRightCorner(5, 5).isApprox(source2.rightCols(5)));
  ASSERT_TRUE(dest_d.bottomLeftCorner(5, 5).isApprox(Eigen::MatrixXd::Zero(5, 5)));
}
