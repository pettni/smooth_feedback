// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Specialization of traits for ROS types.
 */

#include <rclcpp/time.hpp>

#include "smooth/feedback/time.hpp"

namespace smooth::feedback {

/// @brief Specialization of time_trait for ROS time type.
template<>
struct time_trait<rclcpp::Time>
{
  /// @brief Time plus double
  static rclcpp::Time plus(rclcpp::Time t, double t_dbl) { return t + rclcpp::Duration::from_seconds(t_dbl); }

  /// @brief Time minus Time
  static double minus(rclcpp::Time t2, rclcpp::Time t1)
  {
    return (t2 - t1).to_chrono<std::chrono::duration<double>>().count();
  }
};

}  // namespace smooth::feedback
