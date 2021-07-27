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

#include <smooth/feedback/qp.hpp>

static constexpr auto inf   = std::numeric_limits<double>::infinity();
static constexpr double tol = 1e-4;  // TODO decrease this as solver improves
static constexpr smooth::feedback::SolverParams prm{};

TEST(QP, BasicStatic)
{
  smooth::feedback::QuadraticProgram<2, 2> problem;
  problem.P.setIdentity();
  problem.q << -4, 0.25;

  problem.A.setIdentity();
  problem.l << -1, -1;
  problem.u << 1, 1;

  auto sol = smooth::feedback::solve_qp(problem, prm);
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(Eigen::Vector2d(1, -0.25), tol));

  auto sol_hs = smooth::feedback::solve_qp(problem, prm, sol);
  ASSERT_EQ(sol_hs.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol_hs.primal.isApprox(Eigen::Vector2d(1, -0.25), tol));
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

  auto sol = smooth::feedback::solve_qp(problem, prm);
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(Eigen::Vector2d(1, -0.25), tol));

  static_assert(decltype(sol.primal)::SizeAtCompileTime == -1);
  static_assert(decltype(sol.dual)::SizeAtCompileTime == -1);

  auto sol_hs = smooth::feedback::solve_qp(problem, prm, sol);
  ASSERT_EQ(sol_hs.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol_hs.primal.isApprox(Eigen::Vector2d(1, -0.25), tol));
}

TEST(QP, BasicSparse)
{
  smooth::feedback::QuadraticProgramSparse problem;
  problem.P.resize(2, 2);
  problem.P.setIdentity();
  problem.q.resize(2);
  problem.q << -4, 0.25;

  problem.A.resize(2, 2);
  problem.A.setIdentity();
  problem.l.resize(2);
  problem.l << -1, -1;
  problem.u.resize(2);
  problem.u << 1, 1;

  auto sol = smooth::feedback::solve_qp(problem, prm);
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(Eigen::Vector2d(1, -0.25), tol));

  auto sol_hs = smooth::feedback::solve_qp(problem, prm, sol);
  ASSERT_EQ(sol_hs.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol_hs.primal.isApprox(Eigen::Vector2d(1, -0.25), tol));
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

  auto sol = smooth::feedback::solve_qp(problem, prm);
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(Eigen::Vector2d(1, -0.25), tol));

  static_assert(decltype(sol.primal)::SizeAtCompileTime == 2);
  static_assert(decltype(sol.dual)::SizeAtCompileTime == -1);

  auto sol_hs = smooth::feedback::solve_qp(problem, prm, sol);
  ASSERT_EQ(sol_hs.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol_hs.primal.isApprox(Eigen::Vector2d(1, -0.25), tol));
}

TEST(QP, Unconstrained)
{
  smooth::feedback::QuadraticProgram<1, 3> problem;
  problem.P << 4, 2, 2, 2, 4, 2, 2, 2, 4;
  problem.q << -8, -6, -10;

  problem.A.setZero();
  problem.l.setConstant(-inf);
  problem.u.setConstant(inf);

  auto sol = smooth::feedback::solve_qp(problem, prm);
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(Eigen::Matrix<double, 3, 1>(1, 0, 2), tol));

  auto sol_hs = smooth::feedback::solve_qp(problem, prm, sol);
  ASSERT_EQ(sol_hs.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(Eigen::Matrix<double, 3, 1>(1, 0, 2), tol));
}

TEST(QP, PrimalInfeasibleEasy)
{
  smooth::feedback::QuadraticProgram<2, 2> problem;
  problem.P.setIdentity();
  problem.q << 0.1, 0.1;

  problem.A.setIdentity();
  problem.l << -1, 1;
  problem.u << 1, -1;

  auto sol = smooth::feedback::solve_qp(problem, prm);
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

  auto sol = smooth::feedback::solve_qp(problem, prm);
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::PrimalInfeasible);
}

TEST(QP, PrimalInfeasibleInfinity)
{
  smooth::feedback::QuadraticProgram<4, 2> problem;
  problem.P.setIdentity();
  problem.q << 0.1, 0.1;

  problem.A << 1, 1, -1, -1, 1, 0, 0, 1;
  problem.l << 0.5, 0.5, -inf, -inf;
  problem.u << 1, 1, inf, inf;

  auto sol = smooth::feedback::solve_qp(problem, prm);
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::PrimalInfeasible);
}

TEST(QP, DualInfeasible)
{
  smooth::feedback::QuadraticProgram<2, 2> problem;
  problem.P.setZero();
  problem.P(0, 0) = 1;
  problem.q << 1, -1;

  problem.A.setIdentity();
  problem.l << -1, -inf;
  problem.u << 1, inf;

  auto sol = smooth::feedback::solve_qp(problem, prm);
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::DualInfeasible);
}

TEST(QP, PortfolioOptimization)
{
  smooth::feedback::QuadraticProgram<5, 3> problem;

  problem.P << 0.018641, 0.00359853, 0.00130976, 0.00359853, 0.00643694, 0.00488727, 0.00130976,
    0.00488727, 0.0686828;
  problem.q.setZero();

  problem.A.row(0) << 1, 1, 1;
  problem.A.row(1) << 0.0260022, 0.00810132, 0.0737159;
  problem.A.bottomRows(3).setIdentity();

  problem.l << -inf, 50, 0, 0, 0;
  problem.u << 1000, inf, inf, inf, inf;

  Eigen::Vector3d answer(497.04552984986384, 0.0, 502.9544801594811);

  auto sol = smooth::feedback::solve_qp(problem, prm);
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(answer, tol));

  auto sol_hs = smooth::feedback::solve_qp(problem, prm, sol);
  ASSERT_EQ(sol_hs.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol_hs.primal.isApprox(answer, tol));
}

TEST(QP, PortfolioOptimizationSparse)
{
  smooth::feedback::QuadraticProgramSparse problem;

  Eigen::Matrix<double, 3, 3> P_d;
  P_d << 0.018641, 0.00359853, 0.00130976, 0.00359853, 0.00643694, 0.00488727, 0.00130976,
    0.00488727, 0.0686828;

  problem.P.resize(3, 3);
  problem.P = P_d.sparseView();
  problem.q.setZero(3);

  problem.A.resize(5, 3);
  problem.A.insert(0, 0) = 1;
  problem.A.insert(0, 1) = 1;
  problem.A.insert(0, 2) = 1;
  problem.A.insert(1, 0) = 0.0260022;
  problem.A.insert(1, 1) = 0.00810132;
  problem.A.insert(1, 2) = 0.0737159;
  problem.A.insert(2, 0) = 1;
  problem.A.insert(3, 1) = 1;
  problem.A.insert(4, 2) = 1;

  problem.l.resize(5);
  problem.l << -inf, 50, 0, 0, 0;
  problem.u.resize(5);
  problem.u << 1000, inf, inf, inf, inf;

  Eigen::Vector3d answer(497.04552984986384, 0.0, 502.9544801594811);

  auto sol = smooth::feedback::solve_qp(problem, prm);
  ASSERT_EQ(sol.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol.primal.isApprox(answer, tol));

  auto sol_hs = smooth::feedback::solve_qp(problem, prm, sol);
  ASSERT_EQ(sol_hs.code, smooth::feedback::ExitCode::Optimal);
  ASSERT_TRUE(sol_hs.primal.isApprox(answer, tol));
}

TEST(QP, TwoDimensional)
{
  smooth::feedback::QuadraticProgram<2, 2> problem;
  problem.P << 0.0100131, 0, 0, 0.01;
  problem.q << -0.329554, 0.536459;
  problem.A << -0.0639209, -0.168, -0.467, 0;
  problem.l << -inf, -inf;
  problem.u << -0.034974, 0.46571;
  auto sol = smooth::feedback::solve_qp(problem, prm);

  smooth::feedback::QuadraticProgramSparse sp_problem;
  sp_problem.A = problem.A.sparseView();
  sp_problem.P = problem.P.sparseView();
  sp_problem.q = problem.q;
  sp_problem.l = problem.l;
  sp_problem.u = problem.u;
  auto sp_sol  = smooth::feedback::solve_qp(sp_problem, prm);

  ASSERT_TRUE(sol.primal.isApprox(sp_sol.primal));
  ASSERT_TRUE(sol.dual.isApprox(sp_sol.dual));

  ASSERT_TRUE(sol.primal.isApprox(Eigen::Vector2d(46.6338, -17.5351), 1e-4));
}

TEST(QP, Scale)
{
  static constexpr int M = 8;
  static constexpr int N = 4;
  smooth::feedback::QuadraticProgram<M, N> problem;
  problem.P.setRandom();
  problem.P *= 100;
  problem.q.setRandom();

  problem.A.setRandom();
  problem.A *= 100;
  problem.l.setRandom();
  problem.u.setRandom();

  auto [scaled_problem, scale, c] = smooth::feedback::detail::scale_qp(problem);

  Eigen::Matrix<double, N, N> D_x = scale.head(N).asDiagonal();
  Eigen::Matrix<double, M, M> D_e = scale.tail(M).asDiagonal();

  ASSERT_TRUE((c * D_x * problem.P * D_x).isApprox(scaled_problem.P));
  ASSERT_TRUE((c * D_x * problem.q).isApprox(scaled_problem.q));
  ASSERT_TRUE((D_e * problem.A * D_x).isApprox(scaled_problem.A));
  ASSERT_TRUE((D_e * problem.l).isApprox(scaled_problem.l));
  ASSERT_TRUE((D_e * problem.u).isApprox(scaled_problem.u));

  smooth::feedback::QuadraticProgramSparse sp_problem;
  sp_problem.A                             = problem.A.sparseView();
  sp_problem.P                             = problem.P.sparseView();
  sp_problem.q                             = problem.q;
  sp_problem.l                             = problem.l;
  sp_problem.u                             = problem.u;
  auto [sp_scaled_problem, sp_scale, sp_c] = smooth::feedback::detail::scale_qp(sp_problem);

  ASSERT_NEAR(c, sp_c, 1e-5);
  ASSERT_TRUE(sp_scale.isApprox(scale));
}