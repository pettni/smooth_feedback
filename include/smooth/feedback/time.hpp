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

#ifndef SMOOTH__FEEDBACK__TIME_HPP_
#define SMOOTH__FEEDBACK__TIME_HPP_

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

#endif  // SMOOTH__FEEDBACK__TIME_HPP_
