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

#include "smooth/feedback/collocation/dyn.hpp"

template<typename T>
using Vec = Eigen::VectorX<T>;

using Vecd = Vec<double>;

TEST(CollocationDyn, TimeTrajectory)
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
    for (const auto & [i, tau] :
         smooth::utils::zip(std::views::iota(0u, m.N_colloc_ival(p)), m.interval_nodes(p))) {
      X.col(M + i) = x(t0 + (tf - t0) * tau);
      C.col(M + i) = cr.operator()<double>(0, X.col(M + i), U.col(M + i));
    }
    M += m.N_colloc_ival(p);
  }
  X.col(m.N_colloc()) = x(tf);

  Eigen::VectorXd Q{{0}};

  const Eigen::VectorXd dyn_vals = smooth::feedback::colloc_dyn<false>(nx, f, m, t0, tf, X, U);
  ASSERT_EQ(dyn_vals.rows(), m.N_colloc());
  ASSERT_EQ(dyn_vals.cols(), 1);
  ASSERT_LE(dyn_vals.cwiseAbs().maxCoeff(), 1e-8);

  smooth::feedback::CollocEvalResult CReval(ncr, nx, nu, m.N_colloc());
  smooth::feedback::colloc_eval<0>(CReval, cr, m, t0, tf, X.colwise(), U.colwise());

  ASSERT_EQ(CReval.F.rows(), 2);
  ASSERT_EQ(CReval.F.cols(), m.N_colloc());
  ASSERT_TRUE(C.reshaped().isApprox(CReval.F.reshaped()));
}

TEST(CollocationDyn, DynError)
{
  // given trajectory
  const auto x = [](double t) -> Eigen::Vector<double, 1> {
    return Eigen::Vector<double, 1>{{0.1 * t * t - 0.4 * t + 0.2}};
  };

  // system dynamics
  const auto f =
    []<typename T>(const T & t, const Eigen::Vector<T, 1> &, const Eigen::Vector<T, 0> &)
    -> Eigen::Vector<T, 1> { return Eigen::Vector<T, 1>{{0.2 * t - 0.4}}; };

  double t0 = 3;
  double tf = 5;

  smooth::feedback::Mesh<5, 5> m;

  // trajectory is not a polynomial, so we need a couple of intervals for a good approximation
  m.refine_ph(0, 16 * 5);
  ASSERT_EQ(m.N_ivals(), 16);

  // fill X with curve values at the two intervals
  std::size_t M = 0;
  Eigen::MatrixXd X(1, m.N_colloc() + 1);
  for (auto p = 0u; p < m.N_ivals(); ++p) {
    for (const auto & [i, tau] :
         smooth::utils::zip(std::views::iota(0u, m.N_colloc_ival(p)), m.interval_nodes(p))) {
      X.col(M + i) = x(t0 + (tf - t0) * tau);
    }
    M += m.N_colloc_ival(p);
  }
  X.col(m.N_colloc()) = x(tf);

  auto xfun = [X = X, t0 = t0, tf = tf, m = m](const double t) -> Eigen::Vector<double, 1> {
    return m.eval<Eigen::Vector<double, 1>>((t - t0) / (tf - t0), X.colwise(), 0, true);
  };
  auto ufun = [](const double) -> Eigen::Vector<double, 0> {
    return Eigen::Vector<double, 0>::Zero();
  };

  m.increase_degrees();
  auto rel_errs = smooth::feedback::mesh_dyn_error(f, m, t0, tf, xfun, ufun);
  m.decrease_degrees();

  ASSERT_LE(rel_errs.cwiseAbs().maxCoeff(), 1e-8);

  const auto Npre = m.N_ivals();
  m.refine_errors(rel_errs, 1e-8);

  ASSERT_EQ(m.N_ivals(), Npre);
}

TEST(CollocationDyn, StateTrajectory)
{
  // given trajectory and system dynamics
  std::size_t nx = 1;
  std::size_t nu = 0;
  const auto x   = [](double t) { return Vec<double>{{1.5 * exp(-t)}}; };
  const auto f   = []<typename T>(const T &, const Vec<T> & x, const Vec<T> &) -> Vec<T> {
    return Vec<T>{{-x.x()}};
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
    for (const auto & [i, tau] :
         smooth::utils::zip(std::views::iota(0u, m.N_colloc_ival(p)), m.interval_nodes(p))) {
      X.col(M + i) = x(t0 + (tf - t0) * tau);
    }
    M += m.N_colloc_ival(p);
  }
  X.col(m.N_colloc()) = x(tf);

  Eigen::MatrixXd U(nu, m.N_colloc());
  Eigen::VectorXd Q{{0}};

  const auto dyn_vals = smooth::feedback::colloc_dyn<false>(nx, f, m, t0, tf, X, U);
  ASSERT_LE(dyn_vals.cwiseAbs().maxCoeff(), 1e-8);
}
