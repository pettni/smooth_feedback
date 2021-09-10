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

#ifndef SMOOTH__FEEDBACK__COMPAT__ROS_HPP_
#define SMOOTH__FEEDBACK__COMPAT__ROS_HPP_

#include <rclcpp/time.hpp>

#include "smooth/feedback/time.hpp"

namespace smooth::feedback {

template<>
struct time_trait<rclcpp::Time>
{
  static rclcpp::Time plus(rclcpp::Time t, double t_dbl)
  {
    return t + rclcpp::Duration::from_seconds(t_dbl);
  }

  static double minus(rclcpp::Time t2, rclcpp::Time t1)
  {
    return (t2 - t1).to_chrono<std::chrono::duration<double>>().count();
  }
};

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COMPAT__ROS_HPP_
