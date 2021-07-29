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

#include <boost/numeric/odeint.hpp>
#include <smooth/compat/autodiff.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/mpc.hpp>
#include <smooth/se2.hpp>

using std::chrono::nanoseconds;

TEST(Mpc, BasicLieInput)
{
  auto f = []<typename T>(const smooth::SE2<T> &, const smooth::T2<T> & u) {
    return Eigen::Matrix<T, 3, 1>(u.rn()(0), T(0), u.rn()(1));
  };

  smooth::feedback::MPC<3, nanoseconds, smooth::SE2d, smooth::T2d, decltype(f)> mpc(
    std::move(f), nanoseconds(100));
  mpc.set_xudes(
    [](nanoseconds t) -> smooth::SE2d {
      double t_dbl = std::chrono::duration_cast<std::chrono::duration<double>>(t).count();
      return smooth::SE2<double>::exp(t_dbl * Eigen::Vector3d(0.2, 0.1, -0.1));
    },
    [](nanoseconds) -> smooth::T2d { return smooth::T2d::Identity(); });

  ASSERT_NO_THROW(mpc(std::chrono::milliseconds(100), smooth::SE2d::Random()));
}

TEST(Mpc, BasicEigenInput)
{
  auto f = []<typename T>(const smooth::SE2<T> &, const Eigen::Matrix<T, 2, 1> & u) {
    return Eigen::Matrix<T, 3, 1>(u(0), T(0), u(1));
  };

  smooth::feedback::MPC<3, nanoseconds, smooth::SE2d, Eigen::Vector2d, decltype(f)> mpc(
    std::move(f), nanoseconds(100));
  mpc.set_xudes(
    [](nanoseconds t) -> smooth::SE2d {
      double t_dbl = std::chrono::duration_cast<std::chrono::duration<double>>(t).count();
      return smooth::SE2<double>::exp(t_dbl * Eigen::Vector3d(0.2, 0.1, -0.1));
    },
    [](nanoseconds) -> Eigen::Vector2d { return Eigen::Vector2d::Zero(); });

  ASSERT_NO_THROW(mpc(std::chrono::milliseconds(100), smooth::SE2d::Random()));
}
