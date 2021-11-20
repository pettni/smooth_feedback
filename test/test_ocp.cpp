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

#include "smooth/feedback/nlp.hpp"

template<typename T>
using Vec = Eigen::VectorX<T>;

TEST(Ocp, Jacobians)
{
  // objective integrand
  auto phi = []<typename T>(T, Vec<T>, Vec<T> u) -> T { return u.squaredNorm(); };

  // objective end
  auto theta = []<typename T>(T, Vec<T>, Vec<T>) -> T { return 0; };

  // dynamics
  auto f = []<typename T>(T, Vec<T> x, Vec<T> u) -> Vec<T> {
    Vec<T> ret(2);
    ret << x.y(), u.x();
    return ret;
  };

  // integrals
  auto g = []<typename T>(T, Vec<T>, Vec<T>) -> Vec<T> { return Vec<T>::Ones(1); };

  // running constraint
  auto cr = []<typename T>(T, Vec<T>, Vec<T> u) -> Vec<T> { return u; };

  // end constraint
  auto ce = []<typename T>(T, T tf, Vec<T> x0, Vec<T> xf, Vec<T>) -> Vec<T> {
    Vec<T> ret(5);
    ret << tf, x0, xf;
    return ret;
  };

  smooth::feedback::
    OCP<decltype(phi), decltype(theta), decltype(f), decltype(g), decltype(cr), decltype(ce)>
      ocp{
        .nx    = 2,
        .nu    = 1,
        .nq    = 1,
        .ncr   = 1,
        .nce   = 5,
        .phi   = phi,
        .theta = theta,
        .f     = f,
        .g     = g,
        .cr    = cr,
        .crl   = Vec<double>::Constant(1, -1),
        .cru   = Vec<double>::Constant(1, 1),
        .ce    = ce,
        .cel   = Vec<double>::Zero(5),
        .ceu   = Vec<double>::Zero(5),
      };

  smooth::feedback::Mesh<5, 10> mesh;
  auto nlp = smooth::feedback::ocp_to_nlp(ocp, mesh);

  Eigen::VectorXd x = Eigen::VectorXd::Constant(nlp.n, 1);

  const auto df_dx = nlp.df_dx(x);
  const auto dg_dx = nlp.dg_dx(x);

  const auto [fval, df_dx_num] = smooth::diff::dr(nlp.f, smooth::wrt(x));
  const auto [gval, dg_dx_num] = smooth::diff::dr(nlp.g, smooth::wrt(x));

  ASSERT_TRUE(Eigen::MatrixXd(df_dx).isApprox(df_dx_num, 1e-6));
  ASSERT_TRUE(Eigen::MatrixXd(dg_dx).isApprox(dg_dx_num, 1e-6));
}
