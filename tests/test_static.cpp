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

#include "smooth/feedback/time.hpp"

#ifdef ENABLE_ROS_TESTS
#include "smooth/feedback/compat/ros.hpp"
#endif  // ENABLE_ROS_TESTS

TEST(Static, TimeConcept)
{
  static_assert(smooth::feedback::Time<double>);
  static_assert(smooth::feedback::Time<std::chrono::nanoseconds>);
  static_assert(smooth::feedback::Time<std::chrono::milliseconds>);
  static_assert(smooth::feedback::Time<std::chrono::steady_clock::duration>);
  static_assert(smooth::feedback::Time<std::chrono::duration<double>>);
}

#ifdef ENABLE_ROS_TESTS
TEST(Static, TimeConceptRos) { static_assert(smooth::feedback::Time<rclcpp::Time>); }
#endif  // ENABLE_ROS_TESTS
