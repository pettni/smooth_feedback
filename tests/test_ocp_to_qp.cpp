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

#include "smooth/feedback/ocp_to_qp.hpp"

template<typename T>
using X = Eigen::Vector2<T>;

template<typename T>
using U = T;

template<typename T>
using Vec = Eigen::VectorX<T>;

TEST(OcpToQp, Basic)
{
  const auto theta = []<typename T>(T, X<T>, X<T> xf, Vec<T> q) -> T {
    return xf.squaredNorm() + 2 * q.sum();
  };

  const auto f = []<typename T>(T, X<T> x, U<T> u) -> smooth::Tangent<X<T>> {
    return smooth::Tangent<X<T>>{{x.y(), u}};
  };

  const auto g = []<typename T>(T, X<T>, U<T> u) -> Vec<T> {
    Vec<T> ret(1);
    ret << u * u;
    return ret;
  };

  const auto cr = []<typename T>(T, X<T>, U<T> u) -> Vec<T> {
    Vec<T> ret(1);
    ret << u;
    return ret;
  };

  const auto ce = []<typename T>(T, X<T>, X<T> xf, Vec<T>) -> Vec<T> {
    Vec<T> ret(2);
    ret << xf;
    return ret;
  };

  smooth::feedback::
    OCP<X<double>, U<double>, decltype(theta), decltype(f), decltype(g), decltype(cr), decltype(ce)>
      ocp{
        .nx  = 2,
        .nu  = 1,
        .nq  = 1,
        .ncr = 1,
        .nce = 2,
        .f   = f,
        .g   = g,
        .cr  = cr,
        .crl = Eigen::VectorXd(0),
        .cru = Eigen::VectorXd(0),
        .ce  = ce,
        .cel = Eigen::Vector2d{1, 1},
        .ceu = Eigen::Vector2d{1, 1},
      };

  smooth::feedback::Mesh<5> mesh;

  const auto xl_fun = []<typename T>(T t) -> X<T> { return X<T>{t, -t}; };

  const auto ul_fun = []<typename T>(T) -> U<T> { return T(0); };

  const auto qp = smooth::feedback::ocp_to_qp(ocp, mesh, 1., xl_fun, ul_fun);
}
