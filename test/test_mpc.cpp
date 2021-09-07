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
#include <unsupported/Eigen/MatrixFunctions>

using std::chrono::nanoseconds;

TEST(Mpc, OcpToQP)
{
  using G = Eigen::Vector2d;
  using U = Eigen::Vector2d;

  static constexpr int K = 3;

  Eigen::Vector2d x_des = Eigen::Vector2d::Random();
  Eigen::Vector2d u_des = Eigen::Vector2d::Random();

  smooth::feedback::OptimalControlProblem<G, U> ocp{
    .T    = 0.1,
    .x0   = G::Zero(),
    .udes = [&u_des](double) -> U { return u_des; },
    .gdes = [&x_des](double) -> G { return x_des; },
    .ulim =
      smooth::feedback::ManifoldBounds<U>{
        .A = Eigen::Matrix<double, 3, smooth::Dof<U>>{{1, 0}, {0, 1}, {1, 1}},
        .c = U::Zero(),
        .l = Eigen::Vector3d(-4, -5, -6),
        .u = Eigen::Vector3d(4, 5, 6),
      },
    .glim =
      smooth::feedback::ManifoldBounds<G>{
        .A = Eigen::Matrix<double, 3, smooth::Dof<G>>{{1, 0}, {0, 1}, {1, 1}},
        .c = G::Zero(),
        .l = Eigen::Vector3d(-2, -3, -4),
        .u = Eigen::Vector3d(2, 3, 4),
      },
    .weights =
      {
        .Q  = Eigen::Matrix2d{{1, 2}, {2, 4}},
        .QT = Eigen::Matrix2d{{5, 6}, {5, 8}},
        .R  = Eigen::Matrix2d{{9, 10}, {9, 12}},
      },
  };

  smooth::feedback::LinearizationInfo<G, U> lin{
    .g = [](double) -> std::pair<G, smooth::Tangent<G>> {
      return {
        G::Zero(),
        smooth::Tangent<G>::Zero(),
      };
    },
    .u        = [](double) -> U { return U::Zero(); },
    .g_domain = Eigen::Matrix<double, smooth::Dof<G>, 1>(1.5, 2.5),
  };

  Eigen::Matrix2d A = Eigen::Matrix2d::Random();
  Eigen::Matrix2d B = Eigen::Matrix2d::Random();
  auto dyn = [&]<typename T>(double, const Eigen::Vector2<T> & x, const Eigen::Vector2<T> & u) {
    return A * x + B * u;
  };

  auto qp = smooth::feedback::ocp_to_qp<K, G, U>(ocp, dyn, lin);

  double dt = ocp.T / K;

  Eigen::Matrix2d expA = (A * dt).exp();

  static constexpr int Nx = smooth::Dof<G>;
  static constexpr int Nu = smooth::Dof<U>;
  int nu_ineq             = ocp.ulim.value().A.rows();
  int nx_ineq             = ocp.glim.value().A.rows();

  ASSERT_EQ(qp.A.cols(), K * (Nx + Nu));
  ASSERT_GE(qp.A.rows(), Nx * K + (nu_ineq * K) + (nx_ineq * K) + Nx * K);

  // dense versions
  Eigen::MatrixXd Ad(qp.A);
  Eigen::MatrixXd Pd(qp.P);

  // CHECK A

  // check B matrices
  for (auto k = 1u; k != K; ++k) {
    bool test = Ad.block(0, 0, Nx, Nx).isApprox(Ad.block(Nx * k, Nx * k, Nx, Nx));
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

  // check input bounds
  int row0 = K * Nx;
  int col0 = 0;
  for (auto k = 0; k != K; ++k) {
    bool test =
      Ad.block(row0 + k * nu_ineq, col0 + k * Nu, nu_ineq, Nu).isApprox(ocp.ulim.value().A);
    ASSERT_TRUE(test);
    ASSERT_TRUE(qp.l.segment(row0 + k * nu_ineq, nu_ineq).isApprox(ocp.ulim.value().l));
    ASSERT_TRUE(qp.u.segment(row0 + k * nu_ineq, nu_ineq).isApprox(ocp.ulim.value().u));
  }

  // check state bounds
  row0 = K * Nx + K * nu_ineq;
  col0 = K * Nu;
  for (auto k = 0; k != K; ++k) {
    bool test =
      Ad.block(row0 + k * nx_ineq, col0 + k * Nx, nx_ineq, Nx).isApprox(ocp.glim.value().A);
    ASSERT_TRUE(test);
    ASSERT_TRUE(qp.l.segment(row0 + k * nx_ineq, nx_ineq).isApprox(ocp.glim.value().l));
    ASSERT_TRUE(qp.u.segment(row0 + k * nx_ineq, nx_ineq).isApprox(ocp.glim.value().u));
  }

  // check state linearization bounds
  row0 = K * Nx + K * nu_ineq + K * nx_ineq;
  col0 = K * Nu;
  for (auto k = 0; k != K; ++k) {
    bool test = Ad.block(row0 + k * Nx, col0 + k * Nx, Nx, Nx)
                  .isApprox(Eigen::Matrix<double, Nx, Nx>::Identity());
    ASSERT_TRUE(test);

    ASSERT_TRUE(qp.l.segment(row0 + Nx * k, Nx).isApprox(-lin.g_domain));
    ASSERT_TRUE(qp.u.segment(row0 + Nx * k, Nx).isApprox(lin.g_domain));
  }

  // CHECK P AND q

  for (auto k = 0; k != K; ++k) {
    bool test = Pd.block<2, 2>(2 * k, 2 * k).isApprox(ocp.weights.R * dt);
    ASSERT_TRUE(test);
    test = qp.q.segment<2>(2 * k).isApprox(-ocp.weights.R * u_des * dt);
    ASSERT_TRUE(test);
  }

  for (auto k = 0; k != K - 1; ++k) {
    bool test = Pd.block<2, 2>(2 * K + 2 * k, 2 * K + 2 * k).isApprox(ocp.weights.Q * dt);
    ASSERT_TRUE(test);

    test = qp.q.segment<2>(2 * K + 2 * k).isApprox(-ocp.weights.Q * x_des * dt);
    ASSERT_TRUE(test);
  }

  bool test = Pd.block<2, 2>(2 * K + 2 * (K - 1), 2 * K + 2 * (K - 1)).isApprox(ocp.weights.QT);
  ASSERT_TRUE(test);

  test = qp.q.segment<2>(2 * K + 2 * (K - 1)).isApprox(-ocp.weights.QT * x_des);
  ASSERT_TRUE(test);
}

TEST(Mpc, BasicEigenInput)
{
  using Time = std::chrono::duration<double>;

  auto f = []<typename T>(Time, const smooth::SE2<T> &, const Eigen::Vector2<T> & u) {
    return Eigen::Vector3<T>(u(0), T(0), u(1));
  };

  smooth::feedback::MPC<3, Time, smooth::SE2d, Eigen::Vector2d, decltype(f)> mpc(std::move(f),
    smooth::feedback::MPCParams<smooth::SE2d, Eigen::Vector2d>{
      .relinearize_around_solution = true,
      .iterative_relinearization   = 5,
    });

  mpc.set_xdes([](Time t) -> std::pair<smooth::SE2d, Eigen::Vector3d> {
    double t_dbl = std::chrono::duration_cast<Time>(t).count();
    return {
      smooth::SE2<double>::exp(t_dbl * Eigen::Vector3d(0.2, 0.1, -0.1)),
      Eigen::Vector3d(0.2, 0.1, -0.1),
    };
  });
  mpc.set_udes([](Time) -> Eigen::Vector2d { return Eigen::Vector2d::Zero(); });

  Eigen::Vector2d u;

  std::vector<Eigen::Vector2d> uvec;
  std::vector<smooth::SE2d> xvec;
  ASSERT_NO_THROW(mpc(u, std::chrono::milliseconds(100), smooth::SE2d::Random(), uvec, xvec));
}
