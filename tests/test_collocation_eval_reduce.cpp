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

#include <Eigen/Core>

#include "smooth/feedback/collocation/eval_reduce.hpp"

template<typename T>
using Vec = Eigen::VectorX<T>;

using Vecd = Vec<double>;

TEST(Collocation, FunctionEval)
{
  smooth::feedback::Mesh<5, 5> m;

  m.refine_ph(0, 10);

  const auto N = m.N_colloc();

  const auto nf = 3u;
  const auto nx = 2u;
  const auto nu = 1u;

  Eigen::MatrixXd xs(nx, N + 1);
  xs.setRandom();

  Eigen::MatrixXd us(nu, N);
  us.setRandom();

  Eigen::VectorXd ls(N);
  ls.setOnes();

  const auto f = []<typename T>(T, Vec<T> x, Vec<T> u) -> Vec<T> {
    Vec<T> ret(nx + nu);
    ret.head(nx) = x;
    ret.tail(nu) = u;
    return ret;
  };


  smooth::feedback::CollocEvalReduceResult res(nf, nx, nu, N);
  smooth::feedback::colloc_eval_reduce<1>(res, ls, f, m, 0, 1, xs.colwise(), us.colwise());

  ASSERT_TRUE(res.F.head(nx).isApprox(xs.leftCols(N).rowwise().sum()));
  ASSERT_TRUE(res.F.tail(nu).isApprox(us.leftCols(N).rowwise().sum()));

  ASSERT_TRUE(res.dF_dt0.isApprox(Eigen::MatrixXd::Zero(3, 1)));
  ASSERT_TRUE(res.dF_dtf.isApprox(Eigen::MatrixXd::Zero(3, 1)));
}
