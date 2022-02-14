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
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EVecPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <smooth/compat/autodiff.hpp>

#include "smooth/feedback/ocp_flatten.hpp"

constexpr auto DT = smooth::diff::Type::Autodiff;
// constexpr auto DT = smooth::diff::Type::Numerical;

#include "ocp.hpp"

TEST(OcpFlatten, Basic)
{
  // test derivatives
  std::srand(10);

  const auto t1 = smooth::feedback::test_ocp_derivatives<DT>(ocp_test, 5);
  ASSERT_TRUE(t1);

  const auto xl = []<typename T>(const T & t) -> smooth::CastT<T, OcpTest::X> {
    const Eigen::Vector<T, Nx> vel{1, 2, 3};
    return smooth::exp<smooth::CastT<T, OcpTest::X>>(t * vel);
  };
  const auto ul = []<typename T>(const T & t) -> smooth::CastT<T, OcpTest::U> {
    const Eigen::Vector<T, Nu> vel{1, 2};
    return smooth::exp<smooth::CastT<T, OcpTest::U>>(t * vel);
  };

  auto ocp_flat  = smooth::feedback::flatten_ocp(ocp_test, xl, ul);
  const auto t2a = smooth::feedback::test_ocp_derivatives<DT>(ocp_flat, 5);
  ASSERT_TRUE(t2a);

  const auto t2b = smooth::feedback::test_ocp_derivatives<DT>(ocp_flat, 5);
  ASSERT_TRUE(t2b);
}
