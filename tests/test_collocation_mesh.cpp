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

#include "smooth/feedback/collocation/mesh.hpp"

template<typename T>
using Vec = Eigen::VectorX<T>;

using Vecd = Vec<double>;

TEST(CollocMesh, Basic)
{
  smooth::feedback::Mesh<5, 10> m;
  m.refine_ph(0, 5 * 10);
  ASSERT_EQ(m.N_ivals(), 10);

  for (auto i = 0u; i < 10; ++i) {
    auto [n, w] = m.interval_nodes_and_weights(i);
    ASSERT_DOUBLE_EQ(n(0), i * 0.1);
  }

  // will only increase degree
  m.refine_ph(1, 10);
  ASSERT_EQ(m.N_ivals(), 10);
  {
    auto [n, w] = m.interval_nodes_and_weights(1);
    ASSERT_DOUBLE_EQ(n(0), 0.1);
  }

  // actually split it
  m.refine_ph(1, 13);
  ASSERT_EQ(m.N_ivals(), 12);
  {
    auto [n1, w1] = m.interval_nodes_and_weights(1);
    auto [n2, w2] = m.interval_nodes_and_weights(2);
    auto [n3, w3] = m.interval_nodes_and_weights(3);
    ASSERT_DOUBLE_EQ(n1(0), 0.1);
    ASSERT_DOUBLE_EQ(n2(0), 0.1 + 0.1 / 3);
    ASSERT_DOUBLE_EQ(n3(0), 0.1 + 2 * 0.1 / 3);
  }

  m.refine_ph(2, 27);

  m.refine_ph(7, 33);
  m.refine_ph(9, 22);

  const auto [alln, allw] = m.all_nodes_and_weights();

  for (auto i = 0u; i + 1 < alln.size(); ++i) { ASSERT_LE(alln[i], alln[i + 1]); }
}

TEST(Collocation, DifferentiationIntegration)
{
  smooth::feedback::Mesh<8, 8> m;
  m.refine_ph(0, 40);

  // define a function and its derivative
  const auto x  = [](double t) -> double { return 1 + 2 * t + 3 * t * t + 4 * t * t * t; };
  const auto dx = [](double t) -> double { return 2 + 3 * 2 * t + 4 * 3 * t * t; };

  for (auto ival = 0u; ival < m.N_ivals(); ++ival) {
    const auto N               = m.N_colloc_ival(ival);
    const auto [taus, weights] = m.interval_nodes_and_weights(ival);

    // specify function values on mesh
    Eigen::RowVectorXd xvals(N + 1);
    for (auto i = 0u; i < N + 1; ++i) { xvals(i) = x(taus(i)); }
    Eigen::RowVectorXd dxvals(N);
    for (auto i = 0u; i < N; ++i) { dxvals(i) = dx(taus(i)); }

    // derivative and integral matrices
    const auto D = m.interval_diffmat(ival);
    const auto I = m.interval_intmat(ival);

    // expect [dx0 ... dxN-1] = [x0 ... xN] * D
    ASSERT_TRUE(dxvals.isApprox(xvals * D));

    // expect [x1 ... xN] = [x0 ... x0] + [dx0 ... dxN-1] * I
    ASSERT_TRUE(xvals.rightCols(N).isApprox(xvals.leftCols(1).replicate(1, N) + dxvals * I));
  }
}

TEST(Collocation, FunctionEval)
{
  smooth::feedback::Mesh<5, 5> m;

  {
    Eigen::MatrixXd vals = Eigen::MatrixXd::Ones(3, m.N_colloc() + 1);

    const auto x1 = m.eval<Eigen::VectorXd>(0, vals.colwise());
    ASSERT_TRUE(x1.isApprox(Eigen::VectorXd::Ones(3)));

    const auto x2 = m.eval<Eigen::VectorXd>(0.5, vals.colwise());
    ASSERT_TRUE(x2.isApprox(Eigen::VectorXd::Ones(3)));

    const auto x3 = m.eval<Eigen::VectorXd>(1, vals.colwise());
    ASSERT_TRUE(x3.isApprox(Eigen::VectorXd::Ones(3)));
  }

  {
    Eigen::MatrixXd vals = Eigen::MatrixXd::Ones(3, m.N_colloc());

    const auto x1 = m.eval<Eigen::VectorXd>(0, vals.colwise(), 0, false);
    ASSERT_TRUE(x1.isApprox(Eigen::VectorXd::Ones(3)));

    const auto x2 = m.eval<Eigen::VectorXd>(0.5, vals.colwise(), 0, false);
    ASSERT_TRUE(x2.isApprox(Eigen::VectorXd::Ones(3)));

    const auto x3 = m.eval<Eigen::VectorXd>(1, vals.colwise(), 0, false);
    ASSERT_TRUE(x3.isApprox(Eigen::VectorXd::Ones(3)));
  }

  m.refine_ph(0, 40);

  {
    Eigen::MatrixXd vals_refined = Eigen::MatrixXd::Ones(3, m.N_colloc() + 1);

    const auto x1 = m.eval<Eigen::VectorXd>(0, vals_refined.colwise());
    ASSERT_TRUE(x1.isApprox(Eigen::VectorXd::Ones(3)));

    const auto x2 = m.eval<Eigen::VectorXd>(0.5, vals_refined.colwise());
    ASSERT_TRUE(x2.isApprox(Eigen::VectorXd::Ones(3)));

    const auto x3 = m.eval<Eigen::VectorXd>(1, vals_refined.colwise());
    ASSERT_TRUE(x3.isApprox(Eigen::VectorXd::Ones(3)));
  }
}