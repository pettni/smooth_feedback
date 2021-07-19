#include <iostream>

#include <smooth/feedback/mpc.hpp>
#include <smooth/se2.hpp>

#include <gtest/gtest.h>

TEST(Mpc, Basic)
{
  auto dyn = [](double, const auto &, const auto & u) -> Eigen::Vector3d {
    return Eigen::Vector3d(u(0), 0, u(1));
  };

  auto x_nom = [](double) -> smooth::SE2d { return smooth::SE2d::Identity(); };

  auto u_nom = [](double) -> Eigen::Vector2d { return Eigen::Vector2d::Zero(); };

  smooth::SE2d x0 = smooth::SE2d::Random();

  smooth::feedback::MpcWeights<3, 2> w;
  w.R.setConstant(0.5);
  w.Q.setConstant(3);
  w.QT.setConstant(4);


  auto qp = smooth::feedback::mpc<5, smooth::SE2d, Eigen::Vector2d>(
    dyn, x0, 5, x_nom, u_nom, x_nom, u_nom, w);

  std::cout << "qp.P" << std::endl;
  std::cout << qp.P << std::endl;

  std::cout << "qp.q" << std::endl;
  std::cout << qp.q << std::endl;

  std::cout << "qp.A" << std::endl;
  std::cout << qp.A<< std::endl;

  std::cout << "qp.l" << std::endl;
  std::cout << qp.l << std::endl;

  std::cout << "qp.u" << std::endl;
  std::cout << qp.u << std::endl;
}
