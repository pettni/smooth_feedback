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

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <smooth/so3.hpp>

#include "smooth/feedback/ocp_to_nlp.hpp"

template<typename T, std::size_t N>
using Vec = Eigen::Vector<T, N>;

TEST(OcpToNlp, Derivatives2)
{
  // objective
  auto theta = []<typename T>(T tf, Vec<T, 2> x0, Vec<T, 2> xf, Vec<T, 1> q) -> T {
    return (tf - 2) * (tf - 2) + x0.cwiseProduct(xf).squaredNorm() + xf.squaredNorm() + q.sum();
  };

  // dynamics
  auto f = []<typename T>(T t, Vec<T, 2> x, Vec<T, 1> u) -> Vec<T, 2> {
    return Vec<T, 2>{{x.y() + t, x.x() * u.x() * u.x()}};
  };

  // integrals
  auto g = []<typename T>(T t, Vec<T, 2> x, Vec<T, 1> u) -> Vec<T, 1> {
    return Vec<T, 1>{{t + t * x.squaredNorm() + u.squaredNorm()}};
  };

  // running constraint
  auto cr = []<typename T>(T t, Vec<T, 2> x, Vec<T, 1> u) -> Vec<T, 4> {
    Vec<T, 4> ret(4);
    ret << t, t * x * u.x(), u.cwiseAbs2();
    return ret;
  };

  // end constraint
  auto ce = []<typename T>(T tf, Vec<T, 2> x0, Vec<T, 2> xf, Vec<T, 1> q) -> Vec<T, 6> {
    Vec<T, 6> ret(6);
    ret << tf, x0.cwiseProduct(xf), xf, q.cwiseAbs2();
    return ret;
  };

  const smooth::feedback::
    OCP<Vec<double, 2>, Vec<double, 1>, decltype(theta), decltype(f), decltype(g), decltype(cr), decltype(ce)>
      ocp{
        .theta = theta,
        .f     = f,
        .g     = g,
        .cr    = cr,
        .crl   = Vec<double, 4>::Constant(4, -1),
        .cru   = Vec<double, 4>::Constant(4, 1),
        .ce    = ce,
        .cel   = Vec<double, 6>::Constant(6, -1),
        .ceu   = Vec<double, 6>::Constant(6, 1),
      };

  smooth::feedback::Mesh<3, 3> mesh;
  mesh.refine_ph(0, 4);
  mesh.refine_ph(0, 4);

  auto nlp = ocp_to_nlp(ocp, mesh);

  using nlp_t = std::decay_t<decltype(nlp)>;
  static_assert(smooth::feedback::HessianNLP<nlp_t>);

  srand(5);
  const Eigen::VectorXd x      = Eigen::VectorXd::Random(nlp.n());
  const Eigen::VectorXd lambda = Eigen::VectorXd::Random(nlp.m());

  // Analytic derivatives
  const auto & df_dx   = nlp.df_dx(x);
  const auto & d2f_dx2 = nlp.d2f_dx2(x);
  const auto & dg_dx   = nlp.dg_dx(x);
  const auto & d2g_dx2 = nlp.d2g_dx2(x, lambda);

  // Numerical derivatives (of base function)
  const auto [fval, df_dx_num, d2f_dx2_num] =
    smooth::diff::dr<2>([&](const auto & xvar) { return nlp.f(xvar); }, smooth::wrt(x));
  const auto [gval, dg_dx_num] = smooth::diff::dr<1>([&](const auto & xvar) { return nlp.g(xvar); }, smooth::wrt(x));
  const auto g_l_fun           = [&](Eigen::VectorXd xvar) -> double { return lambda.dot(nlp.g(xvar)); };
  const auto [u1_, u2_, d2g_dx2_num] = smooth::diff::dr<2>(g_l_fun, smooth::wrt(x));

  ASSERT_TRUE(Eigen::MatrixXd(df_dx).isApprox(df_dx_num, 1e-4));
  ASSERT_TRUE(Eigen::MatrixXd(dg_dx).isApprox(dg_dx_num, 1e-4));
  ASSERT_TRUE(Eigen::MatrixXd(Eigen::MatrixXd(d2f_dx2).selfadjointView<Eigen::Upper>()).isApprox(d2f_dx2_num, 1e-3));
  ASSERT_TRUE(Eigen::MatrixXd(Eigen::MatrixXd(d2g_dx2).selfadjointView<Eigen::Upper>()).isApprox(d2g_dx2_num, 1e-3));
}
