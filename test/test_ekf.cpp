#include <gtest/gtest.h>

#include <smooth/feedback/ekf.hpp>
#include <smooth/so3.hpp>

TEST(Ekf, NoCrash)
{
  smooth::feedback::EKF<smooth::SO3d> ekf;

  ekf.reset(smooth::SO3d::Identity(), Eigen::Matrix3d::Identity());

  const auto dyn    = [](double, const smooth::SO3d &) { return Eigen::Vector3d::UnitX(); };
  Eigen::Matrix3d Q = Eigen::Matrix3d::Identity();
  const auto meas   = [](const smooth::SO3d & g) -> Eigen::Vector3d {
    return g * Eigen::Vector3d::UnitZ();
  };

  ASSERT_NO_THROW(ekf.predict(dyn, Q, 1, 0.6););
  ASSERT_NO_THROW(ekf.update(meas, Eigen::Vector3d::UnitY(), Q););
  ASSERT_NO_THROW(ekf.predict(dyn, Q, 1, 0.1););
}

TEST(Ekf, Liner)
{
  // TODO: create filter on e.g. R3 and test linear equations
}
