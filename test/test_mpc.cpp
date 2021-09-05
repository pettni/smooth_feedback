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

#include <boost/numeric/odeint.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/mpc.hpp>
#include <smooth/se2.hpp>
#include <unsupported/Eigen/MatrixFunctions>

using std::chrono::nanoseconds;

TEST(Mpc, OcpToQP)
{
  using G = Eigen::Vector2d;
  using U = Eigen::Vector2d;

  static constexpr int K = 3;

  smooth::feedback::OptimalControlProblem<G, U> ocp;

  Eigen::Matrix2d A = Eigen::Matrix2d::Random();
  Eigen::Matrix2d B = Eigen::Matrix2d::Random();

  Eigen::Vector2d x_des = Eigen::Vector2d::Random();
  Eigen::Vector2d u_des = Eigen::Vector2d::Random();

  ocp.Q << 1, 2, 2, 4;
  ocp.QT << 5, 6, 5, 8;
  ocp.R << 9, 10, 9, 12;
  ocp.T = 0.1;

  ocp.glim = smooth::feedback::OptimalControlBounds<G>{
    .l = Eigen::Vector2d(-2, -3),
    .u = Eigen::Vector2d(2, 3),
  };
  ocp.ulim = smooth::feedback::OptimalControlBounds<U>{
    .l = Eigen::Vector2d(-4, -5),
    .u = Eigen::Vector2d(4, 5),
  };

  ocp.gdes = [&](double) -> G { return x_des; };
  ocp.udes = [&](double) -> U { return u_des; };

  auto dyn = [&]<typename T>(double, const Eigen::Vector2<T> & x, const Eigen::Vector2<T> & u) {
    return A * x + B * u;
  };

  auto qp = smooth::feedback::ocp_to_qp<K, G, U>(ocp, dyn);

  double dt = ocp.T / K;

  Eigen::Matrix2d expA = (A * dt).exp();

  ASSERT_EQ(qp.A.cols(), 4 * K);
  ASSERT_GE(qp.A.rows(), 2 * K + 4 * K);

  // dense versions
  Eigen::MatrixXd Ad(qp.A);
  Eigen::MatrixXd Pd(qp.P);

  // CHECK A

  // check B matrices
  for (auto k = 1u; k != K; ++k) {
    bool test = Ad.block<2, 2>(0, 0).isApprox(Ad.block<2, 2>(2 * k, 2 * k));
    ASSERT_TRUE(test);
  }

  // check identity matrices
  for (auto k = 0; k != K; ++k) {
    bool test = Ad.block<2, 2>(2 * k, K * 2 + 2 * k).isApprox(Eigen::Matrix2d::Identity());
    ASSERT_TRUE(test);
  }

  // check A matrices
  for (auto k = 1; k != K; ++k) {
    bool test = Ad.block<2, 2>(2 * k, K * 2 + 2 * (k - 1)).isApprox(-expA, 1e-3);
    ASSERT_TRUE(test);
  }

  // check bounds part of A
  bool test =
    Ad.block<4 * K, 4 * K>(K * 2, 0).isApprox(Eigen::Matrix<double, 4 * K, 4 * K>::Identity());
  ASSERT_TRUE(test);

  for (auto k = 0; k != K; ++k) {
    ASSERT_TRUE(qp.l.segment(2 * K + 2 * k, 2).isApprox(ocp.ulim.value().l));
    ASSERT_TRUE(qp.u.segment(2 * K + 2 * k, 2).isApprox(ocp.ulim.value().u));

    ASSERT_TRUE(qp.l.segment(4 * K + 2 * k, 2).isApprox(ocp.glim.value().l));
    ASSERT_TRUE(qp.u.segment(4 * K + 2 * k, 2).isApprox(ocp.glim.value().u));
  }

  // CHECK P AND q

  for (auto k = 0; k != K; ++k) {
    bool test = Pd.block<2, 2>(2 * k, 2 * k).isApprox(ocp.R * dt);
    ASSERT_TRUE(test);
    test = qp.q.segment<2>(2 * k).isApprox(-ocp.R * u_des * dt);
    ASSERT_TRUE(test);
  }

  for (auto k = 0; k != K - 1; ++k) {
    bool test = Pd.block<2, 2>(2 * K + 2 * k, 2 * K + 2 * k).isApprox(ocp.Q * dt);
    ASSERT_TRUE(test);

    test = qp.q.segment<2>(2 * K + 2 * k).isApprox(-ocp.Q * x_des * dt);
    ASSERT_TRUE(test);
  }

  test = Pd.block<2, 2>(2 * K + 2 * (K - 1), 2 * K + 2 * (K - 1)).isApprox(ocp.QT);
  ASSERT_TRUE(test);

  test = qp.q.segment<2>(2 * K + 2 * (K - 1)).isApprox(-ocp.QT * x_des);
  ASSERT_TRUE(test);
}

TEST(Mpc, BasicEigenInput)
{
  using Time = std::chrono::duration<double>;

  auto f = []<typename T>(Time, const smooth::SE2<T> &, const Eigen::Vector2<T> & u) {
    return Eigen::Vector3<T>(u(0), T(0), u(1));
  };

  smooth::feedback::MPC<3, Time, smooth::SE2d, Eigen::Vector2d, decltype(f)> mpc(
    std::move(f), smooth::feedback::MPCParams{});
  mpc.set_xudes(
    [](Time t) -> smooth::SE2d {
      double t_dbl = std::chrono::duration_cast<Time>(t).count();
      return smooth::SE2<double>::exp(t_dbl * Eigen::Vector3d(0.2, 0.1, -0.1));
    },
    [](Time) -> Eigen::Vector2d { return Eigen::Vector2d::Zero(); });

  Eigen::Vector2d u;
  ASSERT_NO_THROW(mpc(u, std::chrono::milliseconds(100), smooth::SE2d::Random()));
}

TEST(Mpc, BasicEigenInputClock)
{
  using Time = std::chrono::steady_clock::time_point;

  auto f = []<typename T>(Time, const smooth::SE2<T> &, const Eigen::Vector2<T> & u) {
    return Eigen::Vector3<T>(u(0), T(0), u(1));
  };

  smooth::feedback::MPC<3, Time, smooth::SE2d, Eigen::Vector2d, decltype(f)> mpc(
    std::move(f), smooth::feedback::MPCParams{});
  mpc.set_xudes(
    [](Time) -> smooth::SE2d { return smooth::SE2<double>::exp(Eigen::Vector3d(0.2, 0.1, -0.1)); },
    [](Time) -> Eigen::Vector2d { return Eigen::Vector2d::Zero(); });

  std::chrono::steady_clock clock;

  Eigen::Vector2d u;
  ASSERT_NO_THROW(mpc(u, clock.now(), smooth::SE2d::Random()));
}
