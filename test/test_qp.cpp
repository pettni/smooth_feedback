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

#include <smooth/feedback/qp.hpp>

#include <gtest/gtest.h>

TEST(QP, BasicStatic)
{
  smooth::feedback::QuadraticProgram<2, 2> problem;
  problem.P.setIdentity();
  problem.q << -4, 0.25;

  problem.A.setIdentity();
  problem.l << -1, -1;
  problem.u << 1, 1;

  auto sol = smooth::feedback::solveQP(problem, smooth::feedback::SolverParams{});
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(Eigen::Vector2d(1, -0.25), 1e-1));
}

TEST(QP, BasicDynamic)
{
  smooth::feedback::QuadraticProgram<-1, -1> problem;
  problem.P.setIdentity(2, 2);
  problem.q.resize(2);
  problem.q << -4, 0.25;

  problem.A.setIdentity(2, 2);
  problem.l.resize(2);
  problem.l << -1, -1;
  problem.u.resize(2);
  problem.u << 1, 1;

  auto sol = smooth::feedback::solveQP(problem, smooth::feedback::SolverParams{});
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(Eigen::Vector2d(1, -0.25), 1e-1));

  static_assert(decltype(sol.primal)::SizeAtCompileTime == -1);
  static_assert(decltype(sol.dual)::SizeAtCompileTime == -1);
}

TEST(QP, BasicPartialDynamic)
{
  smooth::feedback::QuadraticProgram<-1, 2> problem;
  problem.P.setIdentity();
  problem.q.resize(2);
  problem.q << -4, 0.25;

  problem.A.setIdentity(2, 2);
  problem.l.resize(2);
  problem.l << -1, -1;
  problem.u.resize(2);
  problem.u << 1, 1;

  auto sol = smooth::feedback::solveQP(problem, smooth::feedback::SolverParams{});
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(Eigen::Vector2d(1, -0.25), 1e-1));

  static_assert(decltype(sol.primal)::SizeAtCompileTime == 2);
  static_assert(decltype(sol.dual)::SizeAtCompileTime == -1);
}

TEST(QP, Unconstrained)
{
  smooth::feedback::QuadraticProgram<1, 3> problem;
  problem.P << 4, 2, 2, 2, 4, 2, 2, 2, 4;
  problem.q << -8, -6, -10;

  problem.A.setZero();
  problem.l.setConstant(-std::numeric_limits<double>::infinity());
  problem.u.setConstant(std::numeric_limits<double>::infinity());

  auto sol = smooth::feedback::solveQP(problem, smooth::feedback::SolverParams{});
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(Eigen::Matrix<double, 3, 1>(1, 0, 2), 1e-1));
}

TEST(QP, PrimalInfeasibleEasy)
{
  smooth::feedback::QuadraticProgram<2, 2> problem;
  problem.P.setIdentity();
  problem.q << 0.1, 0.1;

  problem.A.setIdentity();
  problem.l << -1, 1;
  problem.u << 1, -1;

  auto sol = smooth::feedback::solveQP(problem, smooth::feedback::SolverParams{});
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::PrimalInfeasible);
}

TEST(QP, PrimalInfeasibleHard)
{
  smooth::feedback::QuadraticProgram<2, 2> problem;
  problem.P.setIdentity();
  problem.q << 0.1, 0.1;

  problem.A << 1, 1, -1, -1;
  problem.l << 0.5, 0.5;
  problem.u << 1, 1;

  auto sol = smooth::feedback::solveQP(problem, smooth::feedback::SolverParams{});
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::PrimalInfeasible);
}

TEST(QP, PrimalInfeasibleInfinity)
{
  smooth::feedback::QuadraticProgram<4, 2> problem;
  problem.P.setIdentity();
  problem.q << 0.1, 0.1;

  problem.A << 1, 1, -1, -1, 1, 0, 0, 1;
  problem.l << 0.5, 0.5, -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity();
  problem.u << 1, 1, std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity();

  auto sol = smooth::feedback::solveQP(problem, smooth::feedback::SolverParams{});
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::PrimalInfeasible);
}

TEST(QP, DualInfeasible)
{
  smooth::feedback::QuadraticProgram<2, 2> problem;
  problem.P.setZero();
  problem.P(0, 0) = 1;
  problem.q << 1, -1;

  problem.A.setIdentity();
  problem.l << -1, -std::numeric_limits<double>::infinity();
  problem.u << 1, std::numeric_limits<double>::infinity();

  auto sol = smooth::feedback::solveQP(problem, smooth::feedback::SolverParams{});
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::DualInfeasible);
}
