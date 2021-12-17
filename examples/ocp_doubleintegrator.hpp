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

#include <Eigen/Core>
#include <smooth/feedback/ocp.hpp>

template<typename T>
using X = Eigen::Vector<T, 2>;

template<typename T>
using U = Eigen::Vector<T, 1>;

template<typename T, std::size_t N>
using Vec = Eigen::Vector<T, N>;

/// @brief Objective function
const auto theta = []<typename T>(T, const X<T> &, const X<T> &, const Vec<T, 1> & q) -> T {
  return q.x();
};

/// @brief Dynamics
const auto f = []<typename T>(T, const X<T> & x, const U<T> & u) -> smooth::Tangent<X<T>> {
  return smooth::Tangent<X<T>>{{x.y(), u.x()}};
};

/// @brief Integrals
const auto g = []<typename T>(T, const X<T> & x, const U<T> & u) -> Vec<T, 1> {
  return Vec<T, 1>{{x.squaredNorm() + u.squaredNorm()}};
};

/// @brief Running constraints
const auto cr = []<typename T>(T, const X<T> & x, const U<T> & u) -> Vec<T, 2> {
  return Vec<T, 2>{{x.y(), u.x()}};
};

/// @brief End constraints
const auto ce =
  []<typename T>(T tf, const X<T> & x0, const X<T> & xf, const Vec<T, 1> &) -> Vec<T, 5> {
  Vec<T, 5> ret(5);
  ret << tf, x0, xf;
  return ret;
};

using OcpDI = smooth::feedback::
  OCP<X<double>, U<double>, decltype(theta), decltype(f), decltype(g), decltype(cr), decltype(ce)>;

inline const OcpDI ocp_di{
  .nx    = 2,
  .nu    = 1,
  .nq    = 1,
  .ncr   = 2,
  .nce   = 5,
  .theta = theta,
  .f     = f,
  .g     = g,
  .cr    = cr,
  .crl   = Vec<double, 2>{{-0.5, -1}},
  .cru   = Vec<double, 2>{{1.5, 1}},
  .ce    = ce,
  .cel   = Vec<double, 5>{{5, 1, 1, 0.1, 0}},
  .ceu   = Vec<double, 5>{{5, 1, 1, 0.1, 0}},
};
