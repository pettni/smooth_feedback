// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
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

/**
 * @file Optimal control problem on SE2
 */

#ifndef OCP_SE2_HPP_
#define OCP_SE2_HPP_

#include <Eigen/Core>
#include <smooth/bundle.hpp>
#include <smooth/feedback/ocp.hpp>
#include <smooth/se2.hpp>

template<typename T>
using X = smooth::Bundle<smooth::SE2<T>, Eigen::Vector3<T>>;

template<typename T>
using U = Eigen::Vector2<T>;

template<typename T, std::size_t N>
using Vec = Eigen::Vector<T, N>;

/// @brief Objective function
const auto theta = []<typename T>(T tf, const X<T> &, const X<T> &, const Vec<T, 1> & q) -> T {
  return tf + q.x();
};

/// @brief Dynamics
const auto f = []<typename T>(T, const X<T> & x, const U<T> & u) -> smooth::Tangent<X<T>> {
  smooth::Tangent<X<T>> ret;
  ret.segment(0, 3) = x.template part<1>();
  ret(3)            = u.x();
  ret(4)            = T(0);
  ret(5)            = u.y();
  return ret;
};

/// @brief Integrals
const auto g = []<typename T>(T, const X<T> &, const U<T> & u) -> Vec<T, 1> {
  return Vec<T, 1>{{u.squaredNorm()}};
};

/// @brief Running constraints
const auto cr = []<typename T>(T, const X<T> &, const U<T> & u) -> Vec<T, 2> { return u; };

/// @brief End constraints
const auto ce =
  []<typename T>(T tf, const X<T> & x0, const X<T> & xf, const Vec<T, 1> &) -> Vec<T, 10> {
  const smooth::SE2<T> target(smooth::SO2<T>(-0.5), Eigen::Vector2<T>{2, 0.5});
  Vec<T, 10> ret;
  ret << tf, x0.template part<0>().log(), x0.template part<1>(), xf.template part<0>() - target;
  return ret;
};

using OcpSE2 = smooth::feedback::
  OCP<X<double>, U<double>, decltype(theta), decltype(f), decltype(g), decltype(cr), decltype(ce)>;

inline const OcpSE2 ocp_se2{
  .theta = theta,
  .f     = f,
  .g     = g,
  .cr    = cr,
  .crl   = Vec<double, 2>{{-1, -1}},
  .cru   = Vec<double, 2>{{1, 1}},
  .ce    = ce,
  .cel   = Vec<double, 10>{{3, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
  .ceu   = Vec<double, 10>{{15, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
};

#endif  // OCP_SE2_HPP_
