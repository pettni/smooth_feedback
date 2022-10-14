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

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "smooth/feedback/collocation/dyn_error.hpp"

TEST(CollocationDyn, DynError)
{
  // given trajectory
  const auto x = [](double t) -> Eigen::Vector<double, 1> {
    return Eigen::Vector<double, 1>{{0.1 * t * t - 0.4 * t + 0.2}};
  };

  // system dynamics
  const auto f =
    []<typename T>(const T & t, const Eigen::Vector<T, 1> &, const Eigen::Vector<T, 0> &) -> Eigen::Vector<T, 1> {
    return Eigen::Vector<T, 1>{{0.2 * t - 0.4}};
  };

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
    for (const auto & [i, tau] : smooth::utils::zip(std::views::iota(0u, m.N_colloc_ival(p)), m.interval_nodes(p))) {
      X.col(M + i) = x(t0 + (tf - t0) * tau);
    }
    M += m.N_colloc_ival(p);
  }
  X.col(m.N_colloc()) = x(tf);

  auto xfun = [X = X, t0 = t0, tf = tf, m = m](const double t) -> Eigen::Vector<double, 1> {
    return m.eval<Eigen::Vector<double, 1>>((t - t0) / (tf - t0), X.colwise(), 0, true);
  };
  auto ufun = [](const double) -> Eigen::Vector<double, 0> { return Eigen::Vector<double, 0>::Zero(); };

  m.increase_degrees();
  auto rel_errs = smooth::feedback::mesh_dyn_error(f, m, t0, tf, xfun, ufun);
  m.decrease_degrees();

  ASSERT_LE(rel_errs.cwiseAbs().maxCoeff(), 1e-8);

  const auto Npre = m.N_ivals();
  m.refine_errors(rel_errs, 1e-8);

  ASSERT_EQ(m.N_ivals(), Npre);
}
