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

#include <smooth/feedback/pid.hpp>
#include <smooth/se2.hpp>
#include <smooth/spline/spline.hpp>

#include <chrono>

using namespace std::chrono_literals;

TEST(PID, Basic)
{
  smooth::feedback::PID<std::chrono::duration<double>, smooth::SE2d> pid;

  pid.set_kp(1);
  pid.set_kd(1);
  pid.set_ki(1);

  pid.set_kp(Eigen::Vector3d::Constant(1));
  pid.set_kd(Eigen::Vector3d::Constant(1));
  pid.set_ki(Eigen::Vector3d::Constant(1));

  auto u = pid(5s, smooth::SE2d::Identity(), Eigen::Vector3d::Zero());

  ASSERT_LE(u.squaredNorm(), 1e-10);

  // integral state
  u = pid(6s, smooth::SE2d::Random(), Eigen::Vector3d::Zero());
  u = pid(7s, smooth::SE2d::Random(), Eigen::Vector3d::Zero());

  u = pid(8s, smooth::SE2d::Identity(), Eigen::Vector3d::Zero());
  ASSERT_GE(u.squaredNorm(), 1e-10);

  pid.reset_integral();

  u = pid(9s, smooth::SE2d::Identity(), Eigen::Vector3d::Zero());
  ASSERT_LE(u.squaredNorm(), 1e-10);
}

TEST(PID, SetDesiredCurve)
{
  for (auto i = 0u; i != 5; ++i) {
    smooth::feedback::PID<std::chrono::duration<double>, smooth::SE2d> pid;

    pid.set_kp(2);
    pid.set_kd(3);

    std::vector<double> tt{0, 1, 2, 3};

    std::vector<smooth::SE2d> gg{smooth::SE2d::Random(),
      smooth::SE2d::Random(),
      smooth::SE2d::Random(),
      smooth::SE2d::Random()};

    smooth::Spline<smooth::SE2d> c(smooth::fit_cubic_bezier(tt, gg));

    pid.set_xdes(0.5s, c);

    smooth::SE2d g    = smooth::SE2d::Random();
    Eigen::Vector3d v = Eigen::Vector3d::Random();

    typename smooth::SE2d::Tangent u = pid(1s, g, v);

    Eigen::Vector3d v_des, a_des;
    auto g_des = c(0.5, v_des, a_des);

    Eigen::Vector3d u_expected = a_des + 3 * (v_des - v) + 2 * (g_des - g);

    ASSERT_TRUE(u.isApprox(u_expected));
  }
}
