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

TEST(CollocationEvalReduce, FunctionEval)
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

TEST(CollocationEvalReduce, TimeIntegral)
{
  const std::size_t nx = 1;
  const std::size_t nu = 0;
  const std::size_t nq = 1;

  const double t0 = 3;
  const double tf = 5;

  const auto x = [](double t) -> Vec<double> { return Vec<double>{{0.1 * t * t - 0.4 * t + 0.2}}; };
  const auto g = []<typename T>(const T &, const Vec<T> & x, const Vec<T> &) -> Vec<T> {
    return Vec<T>{{0.1 + x.squaredNorm()}};
  };

  smooth::feedback::Mesh<5, 5> m;
  m.refine_ph(0, 40);
  ASSERT_EQ(m.N_ivals(), 8);

  const auto N = m.N_colloc();

  Eigen::MatrixXd X(nx, m.N_colloc() + 1);
  Eigen::MatrixXd U(nu, m.N_colloc());

  // fill X with curve values at the two intervals
  std::size_t M = 0;
  for (auto p = 0u; p < m.N_ivals(); ++p) {
    for (const auto & [i, tau] :
         smooth::utils::zip(std::views::iota(0u, m.N_colloc_ival(p)), m.interval_nodes(p))) {
      X.col(M + i) = x(t0 + (tf - t0) * tau);
    }
    M += m.N_colloc_ival(p);
  }
  X.col(m.N_colloc()) = x(tf);

  Eigen::VectorXd Q{{0}};

  smooth::feedback::CollocEvalReduceResult res(nq, nx, nu, N);
  smooth::feedback::colloc_integrate<2>(res, g, m, t0, tf, X.colwise(), U.colwise());

  ASSERT_NEAR(res.F.x(), 0.217333 + 0.1 * (tf - t0), 1e-4);
}

TEST(CollocationEvalReduce, StateIntegral)
{
  // given trajectory
  const std::size_t nx = 1;
  const std::size_t nu = 0;
  const std::size_t nq = 1;

  const double t0 = 3;
  const double tf = 5;

  const auto x = [](double t) { return Vec<double>{{1.5 * exp(-t)}}; };
  const auto g = []<typename T>(const T &, const Vec<T> & x, const Vec<T> &) -> Vec<T> {
    return Vec<T>{{x.squaredNorm()}};
  };

  smooth::feedback::Mesh<5, 5> m;

  // trajectory is not a polynomial, so we need a couple of intervals for a good approximation
  m.refine_ph(0, 16 * 5);
  ASSERT_EQ(m.N_ivals(), 16);

  const auto N = m.N_colloc();

  Eigen::MatrixXd X(1, m.N_colloc() + 1);

  // fill X with curve values at the two intervals
  std::size_t M = 0;
  for (auto p = 0u; p < m.N_ivals(); ++p) {
    for (const auto & [i, tau] :
         smooth::utils::zip(std::views::iota(0u, m.N_colloc_ival(p)), m.interval_nodes(p))) {
      X.col(M + i) = x(t0 + (tf - t0) * tau);
    }
    M += m.N_colloc_ival(p);
  }
  X.col(m.N_colloc()) = x(tf);

  Eigen::MatrixXd U(nu, m.N_colloc());
  Eigen::VectorXd Q{{0}};

  smooth::feedback::CollocEvalReduceResult res(nq, nx, nu, N);
  smooth::feedback::colloc_integrate<2>(res, g, m, t0, tf, X.colwise(), U.colwise());
  ASSERT_NEAR(res.F.x(), 0.00273752, 1e-4);
}

TEST(CollocationEvalReduce, Hessian)
{
  std::srand(5);

  // given trajectory
  const std::size_t nx = 2;
  const std::size_t nu = 1;
  const std::size_t nq = 1;

  const double t0 = 3;
  const double tf = 5;

  const auto g = []<typename T>(const T & t, const Vec<T> & x, const Vec<T> & u) -> Vec<T> {
    return Vec<T>{{t * x.norm() * u.norm()}};
  };

  smooth::feedback::Mesh<5, 5> m;
  m.refine_ph(0, 10);

  const auto N = m.N_colloc();

  const Eigen::MatrixXd X = Eigen::MatrixXd::Random(nx, N + 1);
  const Eigen::MatrixXd U = Eigen::MatrixXd::Random(nu, N);

  smooth::feedback::CollocEvalReduceResult res(nq, nx, nu, N);
  smooth::feedback::colloc_integrate<2>(res, g, m, t0, tf, X.colwise(), U.colwise());

  const auto f_int =
    [&]<typename T>(T t0, T tf, Eigen::VectorX<T> xvar, Eigen::VectorX<T> uvar) -> T {
    const Eigen::MatrixX<T> Xvar = xvar.reshaped(nx, N + 1);
    const Eigen::MatrixX<T> Uvar = uvar.reshaped(nu, N);

    smooth::feedback::CollocEvalReduceResult res(nq, nx, nu, N);
    smooth::feedback::colloc_integrate<0>(res, g, m, t0, tf, Xvar.colwise(), Uvar.colwise());
    return res.F.x();
  };

  Eigen::VectorXd x_flat = X.reshaped();
  Eigen::VectorXd u_flat = U.reshaped();

  const auto [tmp1, dF, d2F] = smooth::diff::dr<2>(f_int, smooth::wrt(t0, tf, x_flat, u_flat));

  const auto d2F_analytic = smooth::feedback::sparse_block_matrix({
    {res.d2F_dt0t0, res.d2F_dt0tf, res.d2F_dt0X, res.d2F_dt0U},
    {res.d2F_dt0tf, res.d2F_dtftf, res.d2F_dtfX, res.d2F_dtfU},
    {res.d2F_dt0X.transpose(), res.d2F_dtfX.transpose(), res.d2F_dXX, res.d2F_dXU},
    {res.d2F_dt0U.transpose(), res.d2F_dtfU.transpose(), res.d2F_dXU.transpose(), res.d2F_dUU},
  });

  ASSERT_LE((Eigen::MatrixXd(d2F_analytic) - d2F).cwiseAbs().maxCoeff(), 1e-2);
}
