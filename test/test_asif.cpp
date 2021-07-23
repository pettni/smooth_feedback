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

#include <smooth/se2.hpp>
#include <smooth/tn.hpp>
#include <smooth/feedback/asif.hpp>

TEST(Asif, Basic)
{
  const auto f = [](const auto &, const auto & u) {
    using T = typename std::decay_t<decltype(u)>::Scalar;
    return Eigen::Matrix<T, 3, 1>(u.rn()(0), T(0), u.rn()(1));
  };

  const auto h = [](const auto & g) {
    return g.r2();
  };

  const auto bu = [](const auto & g) {
    using T = typename std::decay_t<decltype(g)>::Scalar;
    return smooth::T2<T>(Eigen::Matrix<T, 2, 1>(-0.1, 1));
  };

  smooth::SE2d x0 = smooth::SE2d::Random();

  smooth::feedback::AsifParams prm;

  smooth::T2<double> u(Eigen::Matrix<double, 2, 1>::Zero());
  smooth::T2<double> u_lin(Eigen::Matrix<double, 2, 1>::Zero());

  auto qp = smooth::feedback::asif_to_qp<3, smooth::SE2d, smooth::T2d>(x0, u, u_lin, f, h, bu, prm);

  std::cout << "A" << std::endl;
  std::cout << qp.A << std::endl;
  std::cout << "lu" << std::endl;
  std::cout << qp.l.transpose() << std::endl;
  std::cout << qp.u.transpose() << std::endl;

  std::cout << "P" << std::endl;
  std::cout << qp.P << std::endl;
  std::cout << "q" << std::endl;
  std::cout << qp.q.transpose() << std::endl;
}
