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

TEST(Mpc, Basic)
{
  smooth::feedback::OptimalControlProblem<smooth::SE2d, smooth::T2d> ocp{};
  ocp.gdes = []<typename T>(T t) -> smooth::SE2<T> {
    return smooth::SE2<T>::exp(t * Eigen::Matrix<T, 3, 1>(0.2, 0.1, -0.1));
  };
  ocp.udes = []<typename T>(T) -> smooth::T2<T> { return smooth::T2<T>::Identity(); };

  ocp.x0 = smooth::SE2d::Random();
  ocp.R.setIdentity();
  ocp.Q.diagonal().setConstant(2);
  ocp.QT.setIdentity();

  const auto f = [](const auto &, const auto & u) {
    using T = typename std::decay_t<decltype(u)>::Scalar;
    return Eigen::Matrix<T, 3, 1>(u.rn()(0), T(0), u.rn()(1));
  };
  const auto glin = [](double t) -> smooth::SE2d {
    return smooth::SE2d::exp(t * Eigen::Vector3d(0.2, 0.1, -0.1));
  };
  const auto dglin = [](double) -> Eigen::Vector3d { return Eigen::Vector3d::Zero(); };
  const auto ulin  = [](double) -> smooth::T2d { return smooth::T2d::Identity(); };

  ASSERT_NO_THROW(smooth::feedback::ocp_to_qp<3>(ocp, f, glin, dglin, ulin););
}
