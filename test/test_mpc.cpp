#include <iostream>

#include <smooth/feedback/mpc.hpp>
#include <smooth/se2.hpp>

#include <gtest/gtest.h>

TEST(Mpc, Basic)
{
  smooth::feedback::OptimalControlProblem<smooth::SE2d, smooth::T2d> ocp{};
  ocp.gdes = []<typename T>(T t) -> smooth::SE2<T> {
    return smooth::SE2<T>::exp(t * Eigen::Matrix<T, 3, 1>(0.2, 0.1, -0.1));
  };
  ocp.udes = []<typename T>(T) -> smooth::T2<T> { return smooth::T2<T>::Identity(); };

  ocp.x0 = smooth::SE2d::Random();
  ocp.R.setIdentity();
  ocp.Q.diagonal().setConstant(2);
  ocp.QT.setIdentity();

  const auto f = [](const auto &, const auto & u) {
    using T = typename std::decay_t<decltype(u)>::Scalar;
    return Eigen::Matrix<T, 3, 1>(u.rn()(0), T(0), u.rn()(1));
  };
  const auto glin = [](auto t) {
    using T = std::decay_t<decltype(t)>;
    return smooth::SE2<T>::exp(t * Eigen::Matrix<T, 3, 1>(T(0.2), T(0.1), T(-0.1)));
  };
  const auto ulin = [](double) -> smooth::T2d { return smooth::T2d::Identity(); };

  auto qp = smooth::feedback::ocp_to_qp<3>(ocp, f, glin, ulin);

  std::cout << "qp.P" << std::endl;
  std::cout << qp.P << std::endl;

  std::cout << "qp.q" << std::endl;
  std::cout << qp.q << std::endl;

  std::cout << "qp.A" << std::endl;
  std::cout << qp.A << std::endl;

  std::cout << "qp.l" << std::endl;
  std::cout << qp.l << std::endl;

  std::cout << "qp.u" << std::endl;
  std::cout << qp.u << std::endl;
}
