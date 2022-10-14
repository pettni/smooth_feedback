// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

#include <chrono>
#include <concepts>

namespace smooth::feedback {

/**
 * @brief Trait class to specify Time operations.
 */
template<typename T>
struct time_trait;

// clang-format off

/**
 * @brief A Time type supports right-addition with a double, and subtraction of two time types
 * should be converible to a double.
 *
 * @note This is analogous to a one-dimensional Manifold with tangent type double.
 */
template<typename T>
concept Time = requires(T t1, T t2, double t_dbl)
{
  /// @brief Add double to time
  {time_trait<T>::plus(t1, t_dbl)}->std::convertible_to<T>;

  /// @brief Subtract times and get double
  {time_trait<T>::minus(t2, t1)}->std::convertible_to<double>;
};

// clang-format on

/**
 * @brief Specilization of time trait for floating point types.
 */
template<std::floating_point T>
struct time_trait<T>
{
  // \cond
  static constexpr T plus(T t, double t_dbl) { return t + static_cast<T>(t_dbl); }

  static constexpr double minus(T t2, T t1) { return static_cast<double>(t2 - t1); }
  // \endcond
};

/**
 * @brief Specilization of time trait for std::chrono::time_point types.
 */
template<typename Clock, typename Duration>
struct time_trait<std::chrono::time_point<Clock, Duration>>
{
  // \cond
  using T = std::chrono::time_point<Clock, Duration>;

  static constexpr T plus(T t, double t_dbl)
  {
    return t + std::chrono::duration_cast<Duration>(std::chrono::duration<double>(t_dbl));
  }

  static constexpr double minus(T t2, T t1)
  {
    return std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
  }
  // \endcond
};

/**
 * @brief Specilization of time trait for std::chrono::duration types.
 */
template<typename Rep, typename Ratio>
struct time_trait<std::chrono::duration<Rep, Ratio>>
{
  // \cond
  using T = std::chrono::duration<Rep, Ratio>;

  static constexpr T plus(T t, double t_dbl)
  {
    return t + std::chrono::duration_cast<T>(std::chrono::duration<double>(t_dbl));
  }

  static constexpr double minus(T t2, T t1)
  {
    return std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
  }
  // \endcond
};

}  // namespace smooth::feedback
