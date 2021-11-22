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
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EVecPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>

#include <Eigen/Core>

// #include "smooth/compat/autodiff.hpp"
#include "smooth/feedback/ocp.hpp"

template<typename T>
using Vec = Eigen::VectorX<T>;

TEST(Ocp, Jacobians)
{
  // objective
  auto theta = []<typename T>(T, T tf, Vec<T> x0, Vec<T> xf, Vec<T> q) -> T {
    return (tf - 2) * (tf - 2) + x0.squaredNorm() + xf.squaredNorm() + q.sum();
  };

  // dynamics
  auto f = []<typename T>(T t, Vec<T> x, Vec<T> u) -> Vec<T> { return Vec<T>{{x.y() + t, u.x()}}; };

  // integrals
  auto g = []<typename T>(T t, Vec<T> x, Vec<T> u) -> Vec<T> {
    return Vec<T>{{t + x.squaredNorm() + u.squaredNorm()}};
  };

  // running constraint
  auto cr = []<typename T>(T t, Vec<T> x, Vec<T> u) -> Vec<T> {
    Vec<T> ret(4);
    ret << t, x, u;
    return ret;
  };

  // end constraint
  auto ce = []<typename T>(T, T tf, Vec<T> x0, Vec<T> xf, Vec<T> q) -> Vec<T> {
    Vec<T> ret(6);
    ret << tf, x0, xf, q;
    return ret;
  };

  smooth::feedback::OCP<decltype(theta), decltype(f), decltype(g), decltype(cr), decltype(ce)> ocp{
    .nx    = 2,
    .nu    = 1,
    .nq    = 1,
    .ncr   = 4,
    .nce   = 6,
    .theta = theta,
    .f     = f,
    .g     = g,
    .cr    = cr,
    .crl   = Vec<double>::Constant(4, -1),
    .cru   = Vec<double>::Constant(4, 1),
    .ce    = ce,
    .cel   = Vec<double>::Constant(6, -1),
    .ceu   = Vec<double>::Constant(6, 1),
  };

  smooth::feedback::Mesh mesh(5, 5);
  mesh.refine_ph(0, 8 * 5);

  auto nlp = smooth::feedback::ocp_to_nlp(ocp, mesh);

  Eigen::VectorXd x = Eigen::VectorXd::Constant(nlp.n, 1);

  const auto df_dx = nlp.df_dx(x);
  const auto dg_dx = nlp.dg_dx(x);

  const auto [fval, df_dx_num] =
    smooth::diff::dr<smooth::diff::Type::NUMERICAL>(nlp.f, smooth::wrt(x));
  const auto [gval, dg_dx_num] =
    smooth::diff::dr<smooth::diff::Type::NUMERICAL>(nlp.g, smooth::wrt(x));

  ASSERT_TRUE(Eigen::MatrixXd(df_dx).isApprox(df_dx_num, 1e-8));
  ASSERT_TRUE(Eigen::MatrixXd(dg_dx).isApprox(dg_dx_num, 1e-8));
}
