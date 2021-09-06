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

#include <smooth/feedback/asif.hpp>
#include <smooth/se2.hpp>
#include <smooth/so3.hpp>

template<typename T>
using G = smooth::SE2<T>;

template<typename T>
using U1 = Eigen::Matrix<T, 2, 1>;

TEST(Asif, Basic)
{
  static constexpr int K  = 3;
  static constexpr int Nu = 2;
  static constexpr int Nh = 2;

  const auto f = []<typename T>(T, const G<T> &, const U1<T> & u) -> Eigen::Matrix<T, 3, 1> {
    return Eigen::Matrix<T, 3, 1>(u(0), T(0), u(1));
  };

  const auto h = []<typename T>(T, const G<T> & g) -> Eigen::Matrix<T, Nh, 1> { return g.r2(); };

  const auto bu = []<typename T>(T, const G<T> &) -> Eigen::Matrix<T, 2, 1> {
    return Eigen::Matrix<T, 2, 1>(-0.1, 1);
  };

  smooth::feedback::ASIFProblem<smooth::SE2d, Eigen::Vector2d> pbm{
    .x0    = smooth::SE2d::Random(),
    .u_des = Eigen::Vector2d{0.5, 0.5},
    .ulim =
      {
        .A = Eigen::Matrix<double, 2, 2>{{1, 0}, {0, 1}},
        .c = U1<double>::Zero(),
        .l = Eigen::Vector2d{-1, -1},
        .u = Eigen::Vector2d{1, 1},
      },
  };
  smooth::feedback::ASIFtoQPParams prm{};

  int niq = pbm.ulim.A.rows();

  auto qp = smooth::feedback::asif_to_qp<K, G<double>, U1<double>>(pbm, prm, f, h, bu);

  ASSERT_EQ(qp.P.rows(), Nu + 1);
  ASSERT_EQ(qp.P.cols(), Nu + 1);
  ASSERT_EQ(qp.q.size(), Nu + 1);

  ASSERT_EQ(qp.A.rows(), Nh * K + niq + 1);
  ASSERT_EQ(qp.A.cols(), Nu + 1);
  ASSERT_EQ(qp.l.size(), qp.A.rows());
  ASSERT_EQ(qp.u.size(), qp.A.rows());

  // Expect
  // A = [ BAR   1 ;   Nh * k rows
  //       A_u   0 ;   niq    rows
  //        0    1 ]   1      row

  ASSERT_TRUE(qp.A.block(0, Nu, Nh * K, 1).isApprox(Eigen::Matrix<double, Nh * K, 1>::Ones()));
  ASSERT_TRUE(qp.A.block(Nh * K, 0, niq, Nu).isApprox(pbm.ulim.A));
  ASSERT_TRUE(qp.A.row(Nh * K + niq).isApprox(Eigen::Matrix<double, 1, Nu + 1>::Unit(Nu)));

  ASSERT_EQ(qp.u.head(Nh * K).minCoeff(), std::numeric_limits<double>::infinity());

  ASSERT_TRUE(qp.l.segment(Nh * K, niq).isApprox(pbm.ulim.l - pbm.ulim.A * pbm.u_des));
  ASSERT_TRUE(qp.u.segment(Nh * K, niq).isApprox(pbm.ulim.u - pbm.ulim.A * pbm.u_des));

  ASSERT_EQ(qp.l(Nh * K + niq), 0);
  ASSERT_EQ(qp.u(Nh * K + niq), std::numeric_limits<double>::infinity());
}

template<typename T>
using X = smooth::SO3<T>;

template<typename T>
using U = Eigen::Vector3<T>;

TEST(Asif, Filter)
{
  // dynamics
  auto f = []<typename T>(T, const X<T> &, const U<T> & u) -> smooth::Tangent<X<T>> { return u; };

  // safety set
  auto h = []<typename T>(T, const X<T> & g) -> Eigen::Vector3<T> { return g.log(); };

  // backup controller
  auto bu = []<typename T>(T, const X<T> &) -> U<T> { return U<T>(1, 1, 1); };

  using ASIF =
    smooth::feedback::ASIFilter<100, X<double>, U<double>, decltype(f), decltype(h), decltype(bu)>;

  ASIF asif(f, h, bu);

  smooth::SO3d g           = smooth::SO3d::Random();
  Eigen::Vector3<double> u = Eigen::Vector3d::Zero();

  auto code = asif(u, 0, g);

  ASSERT_EQ(code, smooth::feedback::QPSolutionStatus::Optimal);
}
