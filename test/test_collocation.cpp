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
using state_t = Eigen::Matrix<T, 1, 1>;

template<typename T>
using input_t = Eigen::Matrix<T, 1, 1>;

TEST(Collocation, Mesh)
{
  smooth::feedback::Mesh m;
  m.split(0, 10);

  ASSERT_EQ(m.N_ivals(), 10);

  for (auto i = 0u; i < 10; ++i) {
    auto [n, w] = m.interval_nodes_and_weights(i);
    ASSERT_DOUBLE_EQ(n(0), i * 0.1);
  }

  m.split(1, 2);
  ASSERT_EQ(m.N_ivals(), 11);
  {
    auto [n, w] = m.interval_nodes_and_weights(1);
    ASSERT_DOUBLE_EQ(n(0), 0.1);
  }
  {
    auto [n, w] = m.interval_nodes_and_weights(2);
    ASSERT_DOUBLE_EQ(n(0), 0.15);
  }
}

TEST(Collocation, TimeTrajectory)
{
  // given trajectory
  const auto x = [](double t) { return state_t<double>{0.1 * t * t - 0.4 * t + 0.2}; };

  // path function
  const auto c = []<typename T>(
                   const T &, const state_t<T> & x, const input_t<T> & u) -> Eigen::Vector2<T> {
    return Eigen::Vector2<T>{x.x(), u.x()};
  };

  // system dynamics
  const auto f = []<typename T>(const T & t, const state_t<T> &, const input_t<T> &) -> state_t<T> {
    return state_t<T>{0.2 * t - 0.4};
  };

  // integrand
  const auto g = []<typename T>(
                   const T &, const state_t<T> & x, const input_t<T> & u) -> Eigen::VectorX<T> {
    return Eigen::VectorX<T>::Constant(1, 0.1 + x.squaredNorm() + 2 * u.squaredNorm());
  };

  double t0   = 3;
  double tf   = 5;
  double uval = 0.3;

  smooth::feedback::Mesh m;
  m.split(m.N_ivals() - 1);
  m.split(m.N_ivals() - 1);
  m.split(m.N_ivals() - 1);

  Eigen::MatrixXd X(1, m.N_colloc() + 1);
  Eigen::MatrixXd U = Eigen::MatrixXd::Constant(1, m.N_colloc(), uval);
  Eigen::MatrixXd C(2, m.N_colloc());

  // fill X with curve values at the two intervals
  std::size_t M = 0;
  for (auto p = 0u; p < m.N_ivals(); ++p) {
    const auto [tau_s, w_s] = m.interval_nodes_and_weights(p);
    for (auto i = 0u; i + 1 < tau_s.size(); ++i) {
      X.col(M + i) = x(t0 + (tf - t0) * tau_s[i]);
      C.col(M + i) = c.operator()<double>(0, X.col(M + i), U.col(M + i));
    }
    M += m.N_colloc_ival(p);
  }
  X.col(m.N_colloc()) = x(tf);

  Eigen::VectorXd I = Eigen::VectorXd::Constant(1, 0);

  const Eigen::VectorXd residual =
    std::get<0>(smooth::feedback::dynamics_constraint(f, m, t0, tf, X, U));
  const Eigen::VectorXd path =
    std::get<0>(smooth::feedback::colloc_eval(2, c, m, t0, tf, X, U)).reshaped();
  const Eigen::VectorXd integral =
    std::get<0>(smooth::feedback::integral_constraint(g, m, t0, tf, I, X, U));

  ASSERT_EQ(residual.rows(), m.N_colloc());
  ASSERT_EQ(residual.cols(), 1);

  ASSERT_EQ(path.rows(), 2 * m.N_colloc());
  ASSERT_EQ(path.cols(), 1);

  ASSERT_LE(residual.cwiseAbs().maxCoeff(), 1e-8);
  ASSERT_TRUE(C.reshaped().isApprox(path));
  ASSERT_NEAR(integral.x(), 0.217333 + 0.1 * (tf - t0) + 2 * uval * uval * (tf - t0), 1e-4);
}

TEST(Collocation, StateTrajectory)
{
  // system dynamics
  const auto f = []<typename T>(const T &, const state_t<T> & x, const input_t<T> &) -> state_t<T> {
    return state_t<T>{-x.x()};
  };

  // given trajectory
  const auto x = [](double t) { return state_t<double>{1.5 * exp(-t)}; };

  // integrand
  const auto g = []<typename T>(
                   const T &, const state_t<T> & x, const input_t<T> &) -> Eigen::VectorX<T> {
    return Eigen::VectorX<T>::Constant(1, x.squaredNorm());
  };

  double t0   = 3;
  double tf   = 5;
  double uval = 0;

  smooth::feedback::Mesh m;

  // trajectory is not a polynomial, so we need a couple of intervals for a good approximation
  for (int i = 0; i < 4; ++i) { m.split_all(); }

  Eigen::MatrixXd X(1, m.N_colloc() + 1);

  // fill X with curve values at the two intervals
  std::size_t M = 0;
  for (auto p = 0u; p < m.N_ivals(); ++p) {
    const auto [tau_s, w_s] = m.interval_nodes_and_weights(p);
    for (auto i = 0u; i + 1 < tau_s.size(); ++i) { X.col(M + i) = x(t0 + (tf - t0) * tau_s[i]); }
    M += m.N_colloc_ival(p);
  }
  X.col(m.N_colloc()) = x(tf);

  Eigen::MatrixXd U = Eigen::MatrixXd::Constant(1, m.N_colloc(), uval);

  Eigen::VectorXd I = Eigen::VectorXd::Constant(1, 0);

  const auto residual = std::get<0>(smooth::feedback::dynamics_constraint(f, m, t0, tf, X, U));
  const auto integral = std::get<0>(smooth::feedback::integral_constraint(g, m, t0, tf, I, X, U));

  ASSERT_LE(residual.cwiseAbs().maxCoeff(), 1e-8);
  ASSERT_NEAR(integral.x(), 0.00273752, 1e-4);
}
