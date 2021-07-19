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

#include <iostream>

// #include <smooth/compat/autodiff.hpp>
#include <smooth/feedback/qp.hpp>

#include <gtest/gtest.h>

TEST(QP, Basic)
{
  smooth::feedback::QuadraticProgram<2, 2> problem;
  problem.A.setIdentity();
  problem.q << -4, 0.25;

  problem.P.setIdentity();
  problem.l << -1, -1;
  problem.u << 1, 1;

  auto sol = smooth::feedback::solveQP(problem, smooth::feedback::SolverParams{});

  std::cout << "Exited w/ code " << static_cast<int>(sol.code) << std::endl;
  std::cout << "Found solution " << sol.primal.transpose() << std::endl;

  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(Eigen::Vector2d(1, -0.25), 1e-1));
}
