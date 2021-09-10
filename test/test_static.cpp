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
