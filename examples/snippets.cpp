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

#include <smooth/feedback/ekf.hpp>
#include <smooth/se2.hpp>

#include <iostream>

void ekf_snippet()
{
  smooth::feedback::EKF<smooth::SE2d> ekf;

  // PREDICT STEP: propagate filter over time
  ekf.predict(
    [](double, const auto &) { return smooth::SE2d::Tangent(0.4, 0.01, 0.1); },  // motion model
    Eigen::Matrix3d::Identity(),  // motion covariance
    1.                            // time step length
  );

  // UPDATE STEP: register a measurement of a landmark at [1, 1]
  Eigen::Vector2d landmark(1, 1);
  ekf.update([&landmark](const auto & x) { return x.inverse() * landmark; },  // measurement model
    Eigen::Vector2d(0.3, 0.6),                                                // measurement result
    Eigen::Matrix2d::Identity()  // measurement covariance
  );

  // access estimate
  std::cout << ekf.estimate() << std::endl;
}

int main()
{
  std::cout << "RUNNING EKF" << std::endl;
  ekf_snippet();
}
