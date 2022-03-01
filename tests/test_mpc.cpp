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

#include <gtest/gtest.h>

#include <smooth/feedback/mpc.hpp>
#include <smooth/se2.hpp>

TEST(Mpc, Api)
{
  using T = double;
  using X = smooth::SE2d;
  using U = Eigen::Vector2d;

  auto f = []<typename S>(const smooth::CastT<S, X> &, const smooth::CastT<S, U> & u) {
    return smooth::Tangent<smooth::CastT<S, X>>(u(0), S(0), u(1));
  };

  auto cr = []<typename S>(const smooth::CastT<S, X> &, const smooth::CastT<S, U> & u) {
    return u;
  };
  Eigen::Vector2d crl = Eigen::Vector2d::Ones();

  smooth::feedback::MPC<T, X, U, decltype(f), decltype(cr)> mpc{f, cr, -crl, crl};

  const T t = 1;
  const X x = X::Random();

  // nothing set
  auto [u0, code0] = mpc(t, x);
  ASSERT_EQ(code0, smooth::feedback::QPSolutionStatus::Optimal);

  mpc.reset_warmstart();

  mpc.set_weights({
    .Q  = Eigen::Matrix3d::Identity(),
    .QT = Eigen::Matrix3d::Identity(),
    .R  = Eigen::Matrix2d::Identity(),
  });

  mpc.set_udes([](T) -> U { return U::Ones(); });
  mpc.set_xdes_rel(
    []<typename S>(S) -> smooth::CastT<S, X> { return smooth::CastT<S, X>::Identity(); });

  // no warmstart
  auto [u1, code1] = mpc(t, x);
  ASSERT_EQ(code1, smooth::feedback::QPSolutionStatus::Optimal);

  // with warmstart
  auto [u2, code2] = mpc(t, x);
  ASSERT_EQ(code2, smooth::feedback::QPSolutionStatus::Optimal);

  ASSERT_TRUE(u1.isApprox(u2));

  // output stuff
  std::vector<X> xs;
  std::vector<U> us;
  auto [u3, code3] = mpc(t, x, us, xs);

  ASSERT_TRUE(u3.isApprox(u1));

  ASSERT_TRUE(us.size() + 1 == xs.size());
}
