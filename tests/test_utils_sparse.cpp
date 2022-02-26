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

TEST(Utils, BlockAdd)
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
