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

#include "smooth/feedback/collocation.hpp"

template<typename T>
using Vec = Eigen::VectorX<T>;

using Vecd = Vec<double>;

TEST(Collocation, Mesh)
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

TEST(Collocation, TimeTrajectory)
{
  // given trajectory
  std::size_t nx = 1;
  const auto x = [](double t) -> Vec<double> { return Vec<double>{{0.1 * t * t - 0.4 * t + 0.2}}; };

  // running constraints
  std::size_t ncr = 2;
  const auto cr   = []<typename T>(const T &, const Vec<T> & x, const Vec<T> &) -> Vec<T> {
    return Vec<T>{{x.x(), 0}};
  };

  // system dynamics
  std::size_t nu = 0;
  const auto f   = []<typename T>(const T & t, const Vec<T> &, const Vec<T> &) -> Vec<T> {
    return Vec<T>{{0.2 * t - 0.4}};
  };

  // integrand
  std::size_t nq = 1;
  const auto g   = []<typename T>(const T &, const Vec<T> & x, const Vec<T> &) -> Vec<T> {
    return Vec<T>{{0.1 + x.squaredNorm()}};
  };

  double t0 = 3;
  double tf = 5;

  smooth::feedback::Mesh<5, 5> m;
  m.refine_ph(0, 40);
  ASSERT_EQ(m.N_ivals(), 8);

  Eigen::MatrixXd X(nx, m.N_colloc() + 1);
  Eigen::MatrixXd U(nu, m.N_colloc());
  Eigen::MatrixXd C(ncr, m.N_colloc());

  // fill X with curve values at the two intervals
  std::size_t M = 0;
  for (auto p = 0u; p < m.N_ivals(); ++p) {
    const auto [tau_s, w_s] = m.interval_nodes_and_weights(p);
    for (auto i = 0u; i + 1 < tau_s.size(); ++i) {
      X.col(M + i) = x(t0 + (tf - t0) * tau_s[i]);
      C.col(M + i) = cr.operator()<double>(0, X.col(M + i), U.col(M + i));
    }
    M += m.N_colloc_ival(p);
  }
  X.col(m.N_colloc()) = x(tf);

  Eigen::VectorXd Q{{0}};

  const Eigen::VectorXd dyn_vals =
    smooth::feedback::dynamics_constraint<false>(nx, f, m, t0, tf, X, U);
  ASSERT_EQ(dyn_vals.rows(), m.N_colloc());
  ASSERT_EQ(dyn_vals.cols(), 1);
  ASSERT_LE(dyn_vals.cwiseAbs().maxCoeff(), 1e-8);

  const Eigen::VectorXd cr_vals =
    smooth::feedback::colloc_eval<false>(ncr, cr, m, t0, tf, X, U).reshaped();
  ASSERT_EQ(cr_vals.rows(), 2 * m.N_colloc());
  ASSERT_EQ(cr_vals.cols(), 1);
  ASSERT_TRUE(C.reshaped().isApprox(cr_vals));

  const Eigen::VectorXd q_vals =
    smooth::feedback::integral_constraint<false>(nq, g, m, t0, tf, Q, X, U);
  ASSERT_NEAR(q_vals.x(), 0.217333 + 0.1 * (tf - t0), 1e-4);
}

TEST(Collocation, StateTrajectory)
{
  // given trajectory and system dynamics
  std::size_t nx = 1;
  std::size_t nu = 0;
  const auto x   = [](double t) { return Vec<double>{{1.5 * exp(-t)}}; };
  const auto f   = []<typename T>(const T &, const Vec<T> & x, const Vec<T> &) -> Vec<T> {
    return Vec<T>{{-x.x()}};
  };

  // integrals
  std::size_t nq = 1;
  const auto g   = []<typename T>(const T &, const Vec<T> & x, const Vec<T> &) -> Vec<T> {
    return Vec<T>{{x.squaredNorm()}};
  };

  double t0 = 3;
  double tf = 5;

  smooth::feedback::Mesh<5, 5> m;

  // trajectory is not a polynomial, so we need a couple of intervals for a good approximation
  m.refine_ph(0, 16 * 5);
  ASSERT_EQ(m.N_ivals(), 16);

  Eigen::MatrixXd X(1, m.N_colloc() + 1);

  // fill X with curve values at the two intervals
  std::size_t M = 0;
  for (auto p = 0u; p < m.N_ivals(); ++p) {
    const auto [tau_s, w_s] = m.interval_nodes_and_weights(p);
    for (auto i = 0u; i + 1 < tau_s.size(); ++i) { X.col(M + i) = x(t0 + (tf - t0) * tau_s[i]); }
    M += m.N_colloc_ival(p);
  }
  X.col(m.N_colloc()) = x(tf);

  Eigen::MatrixXd U(nu, m.N_colloc());
  Eigen::VectorXd Q{{0}};

  const auto dyn_vals = smooth::feedback::dynamics_constraint<false>(nx, f, m, t0, tf, X, U);
  ASSERT_LE(dyn_vals.cwiseAbs().maxCoeff(), 1e-8);

  const auto q_vals = smooth::feedback::integral_constraint<false>(nq, g, m, t0, tf, Q, X, U);
  ASSERT_NEAR(q_vals.x(), 0.00273752, 1e-4);
}
