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

#include <smooth/feedback/asif.hpp>
#include <smooth/se2.hpp>

template<typename T>
using G = smooth::SE2<T>;

template<typename T>
using U = Eigen::Matrix<T, 2, 1>;

TEST(Asif, Basic)
{
  const auto f = []<typename T>(T, const G<T> &, const U<T> & u) -> Eigen::Matrix<T, 3, 1> {
    return Eigen::Matrix<T, 3, 1>(u(0), T(0), u(1));
  };

  const auto h = []<typename T>(T, const G<T> & g) -> Eigen::Matrix<T, 2, 1> { return g.r2(); };

  const auto bu = []<typename T>(T, const G<T> &) -> Eigen::Matrix<T, 2, 1> {
    return Eigen::Matrix<T, 2, 1>(-0.1, 1);
  };

  smooth::feedback::ASIFProblem<smooth::SE2d, Eigen::Vector2d> pbm{
    .x0    = smooth::SE2d::Random(),
    .u_des = Eigen::Vector2d::Zero(),
  };
  smooth::feedback::ASIFtoQPParams prm{};

  auto qp = smooth::feedback::asif_to_qp<3, smooth::SE2d, Eigen::Vector2d>(pbm, f, h, bu, prm);

  // TODO check qp...
}
