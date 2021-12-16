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

#include <smooth/compat/autodiff.hpp>

#include "smooth/feedback/ocp.hpp"
#include "smooth/feedback/ocp_to_qp.hpp"

template<typename T>
using X = Eigen::VectorX<T>;

template<typename T>
using U = Eigen::VectorX<T>;

template<typename T>
using Vec = Eigen::VectorX<T>;

TEST(OcpToQp, Basic)
{
  const auto theta = []<typename T>(T, X<T>, X<T> xf, Vec<T> q) -> T {
    return xf.squaredNorm() + 2 * q.sum();
  };

  const auto f = []<typename T>(T, X<T> x, U<T> u) -> Vec<T> {
    return smooth::Tangent<X<T>>{{x.y(), u.x()}};
  };

  const auto g = []<typename T>(T, X<T>, U<T> u) -> Vec<T> {
    Vec<T> ret(1);
    ret << u.x() * u.x();
    return ret;
  };

  const auto cr = []<typename T>(T, X<T>, U<T> u) -> Vec<T> {
    Vec<T> ret(1);
    ret << u.x();
    return ret;
  };

  const auto ce = []<typename T>(T, X<T>, X<T> xf, Vec<T>) -> Vec<T> {
    Vec<T> ret(2);
    ret << xf;
    return ret;
  };

  smooth::feedback::
    OCP<X<double>, U<double>, decltype(theta), decltype(f), decltype(g), decltype(cr), decltype(ce)>
      ocp{
        .nx    = 2,
        .nu    = 1,
        .nq    = 1,
        .ncr   = 1,
        .nce   = 2,
        .theta = theta,
        .f     = f,
        .g     = g,
        .cr    = cr,
        .crl   = Eigen::VectorXd{{-1}},
        .cru   = Eigen::VectorXd{{1}},
        .ce    = ce,
        .cel   = Eigen::Vector2d{-5, -5},
        .ceu   = Eigen::Vector2d{5, 5},
      };

  smooth::feedback::Mesh<5> mesh;

  constexpr auto tf = 2.;

  const auto xl_fun = []<typename T>(T t) -> X<T> { return X<T>{{0.05 * t * t, 0.1 * t}}; };

  const auto ul_fun = []<typename T>(T) -> U<T> { return U<T>{{0.1}}; };

  const auto qp = smooth::feedback::ocp_to_qp(ocp, mesh, tf, xl_fun, ul_fun);

  ASSERT_EQ(qp.P.cols(), qp.q.size());
  ASSERT_EQ(qp.P.rows(), qp.q.size());
  ASSERT_EQ(qp.P.cols(), qp.A.cols());

  ASSERT_EQ(qp.A.rows(), qp.l.size());
  ASSERT_EQ(qp.A.rows(), qp.u.size());

  // check that simple trajectory satisfies constraints

  static constexpr double x0 = 3;
  static constexpr double v0 = -0.3;
  static constexpr double u0 = 0.1;

  const auto xtraj = [](double t) {
    return X<double>{{x0 + v0 * t + u0 * t * t / 2, v0 + u0 * t}};
  };

  Eigen::MatrixXd Xvar(ocp.nx, mesh.N_colloc() + 1);
  Eigen::MatrixXd Uvar(ocp.nu, mesh.N_colloc());

  const auto [nodes, weights] = mesh.all_nodes_and_weights();

  for (const auto [i, t] : smooth::utils::zip(std::views::iota(0u), nodes)) {
    Xvar.col(i) = xtraj(tf * t);
    if (i < mesh.N_colloc()) { Uvar.col(i).setConstant(u0); }
  }

  Eigen::VectorXd var(Xvar.size() + Uvar.size());
  var.head(Xvar.size()) = Xvar.reshaped();
  var.tail(Uvar.size()) = Uvar.reshaped();

  // check constraint satisfaction
  ASSERT_GE((qp.A * var - qp.l).minCoeff(), -1e-8);
  ASSERT_GE((qp.u - qp.A * var).minCoeff(), -1e-8);

  std::cout << "lower " << qp.l.transpose() << std::endl;
  std::cout << "value " << (qp.A * var).transpose() << std::endl;
  std::cout << "upper " << qp.u.transpose() << std::endl;

  std::cout << "P\n" << Eigen::MatrixXd(qp.P) << '\n';
  std::cout << "q" << qp.q.transpose() << '\n';
}
