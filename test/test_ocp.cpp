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

#include "smooth/feedback/nlp.hpp"

template<typename T>
using Vec = Eigen::VectorX<T>;

TEST(Ocp, Basic)
{
  // Objective integrand function
  const auto phi = []<typename T>(
                     T, const Vec<T> &, const Vec<T> & u) -> T { return u.squaredNorm(); };

  // Objective end function
  const auto theta = []<typename T>(T, const Vec<T> &, const Vec<T> &) -> T { return 0; };

  /// @brief System dynamics
  const auto f = []<typename T>(T, const Vec<T> & x, const Vec<T> & u) -> Vec<T> {
    Vec<T> ret(2);
    ret << x.y(), u.x();
    return ret;
  };

  /// @brief System integrals
  const auto g = []<typename T>(
                   T, const Vec<T> &, const Vec<T> &) -> Vec<T> { return Vec<T>::Zero(0); };

  /// @brief Running constraint function
  const auto cr = []<typename T>(
                    T, const Vec<T> &, const Vec<T> &) -> Vec<T> { return Vec<T>::Zero(0); };

  /// @brief End constraint function
  const auto ce = []<typename T>(
                    T tf, const Vec<T> & x0, const Vec<T> & xf, const Vec<T> &) -> Vec<T> {
    Vec<T> ret(5);
    ret << tf, x0, xf;
    return ret;
  };

  OCP<decltype(phi), decltype(theta), decltype(f), decltype(g), decltype(cr), decltype(ce)> ocp{
    .nx    = 2,
    .nu    = 1,
    .nq    = 0,
    .ncr   = 0,
    .nce   = 5,
    .phi   = phi,
    .theta = theta,
    .f     = f,
    .g     = g,
    .cr    = cr,
    .crl   = Vec<double>::Zero(0),
    .cru   = Vec<double>::Zero(0),
    .ce    = ce,
    .cel   = Vec<double>::Zero(5),
    .ceu   = Vec<double>::Zero(5),
  };

  std::cout << "Creating mesh" << std::endl;

  smooth::feedback::Mesh<5, 10> mesh;

  std::cout << "Calling ocp_to_nlp" << std::endl;

  auto nlp = ocp_to_nlp(ocp, mesh);

  std::cout << "nlp has " << nlp.n << " vars and " << nlp.m << " constraints" << std::endl;

  Eigen::VectorXd x = Eigen::VectorXd::Constant(nlp.n, 1);

  std::cout << "f(x) = " << nlp.f(x) << std::endl;
  std::cout << "g(x) = " << nlp.g(x).transpose() << std::endl;
}
