#include <gtest/gtest.h>

#include <unsupported/Eigen/MatrixFunctions>

#include <smooth/compat/ceres.hpp>
#include <smooth/feedback/ekf.hpp>
#include <smooth/so3.hpp>
#include <smooth/tn.hpp>

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

template<int Nx, int Ny>
void test_update_linear()
{
  for (auto it = 0; it != 10; ++it) {
    // true state
    Eigen::Matrix<double, Nx, 1> x = Eigen::Matrix<double, Nx, 1>::Random();

    // initial estimate
    Eigen::Matrix<double, Nx, 1> xhat = Eigen::Matrix<double, Nx, 1>::Random();

    // linear system
    smooth::feedback::EKF<smooth::Tn<Nx, double>> ekf;

    Eigen::Matrix<double, Nx, Nx> P = Eigen::Matrix<double, Nx, 1>::Random().asDiagonal();
    P.diagonal() += Eigen::Matrix<double, Nx, 1>::Constant(1.1);

    ekf.reset(xhat, P);

    // measurement model
    Eigen::Matrix<double, Ny, Nx> H = Eigen::Matrix<double, Ny, Nx>::Random();
    Eigen::Matrix<double, Ny, 1> h  = Eigen::Matrix<double, Ny, 1>::Random();
    Eigen::Matrix<double, Ny, Ny> R = Eigen::Matrix<double, Ny, 1>::Random().asDiagonal();
    R.diagonal() += Eigen::Matrix<double, Ny, 1>::Constant(1.1);

    // add measurement (without noise)
    ekf.update([&H, &h](const auto & xvar) { return H * xvar.rn() + h; }, H * x + h, R);

    // kalman update for linear system
    Eigen::Matrix<double, Ny, Ny> S = H * P * H.transpose() + R;
    Eigen::Matrix<double, Nx, Ny> K = P * H.transpose() * S.inverse();

    // expected results
    Eigen::Matrix<double, Nx, 1> x_new  = xhat + K * (H * x + h - (H * xhat + h));
    Eigen::Matrix<double, Nx, Nx> P_new = (Eigen::Matrix<double, Nx, Nx>::Identity() - K * H) * P;

    ASSERT_TRUE(x_new.isApprox(ekf.state().rn(), 1e-6));
    ASSERT_TRUE(P_new.isApprox(ekf.cov(), 1e-6));
  }
}

TEST(Ekf, UpdateLinear)
{
  test_update_linear<3, 3>();
  test_update_linear<3, 3>();

  test_update_linear<10, 3>();
  test_update_linear<10, 3>();

  test_update_linear<3, 10>();
  test_update_linear<3, 10>();
}

template<int Nx>
void test_predict_linear()
{
  for (auto it = 0; it != 10; ++it) {
    // initial estimate
    Eigen::Matrix<double, Nx, 1> xhat = Eigen::Matrix<double, Nx, 1>::Random();

    // linear system
    smooth::feedback::EKF<smooth::Tn<Nx, double>,
      smooth::diff::Type::NUMERICAL,
      boost::numeric::odeint::runge_kutta4>
      ekf;

    Eigen::Matrix<double, Nx, Nx> P = Eigen::Matrix<double, Nx, 1>::Random().asDiagonal();
    P.diagonal() += Eigen::Matrix<double, Nx, 1>::Constant(1.1);

    ekf.reset(xhat, P);

    // dynamical model
    Eigen::Matrix<double, Nx, Nx> A = Eigen::Matrix<double, Nx, Nx>::Random();
    Eigen::Matrix<double, Nx, Nx> Q = Eigen::Matrix<double, Nx, 1>::Random().asDiagonal();
    Q.diagonal() += Eigen::Matrix<double, Nx, 1>::Constant(1.1);
    Q.setZero();  // linear system solution complicated for Q != 0

    double tau = 0.7;

    // propagate
    ekf.predict([&A](double, const auto & xvar) { return A * xvar.rn(); }, Q, tau, 1e-3);

    // exact solution
    // x_new = exp(A * t) * xhat
    Eigen::Matrix<double, Nx, Nx> F       = (A * tau).exp();
    Eigen::Matrix<double, Nx, 1> xhat_new = F * xhat;
    Eigen::Matrix<double, Nx, Nx> P_new   = F * P * F.transpose() + Q * tau;

    ASSERT_TRUE(xhat_new.isApprox(ekf.state().rn(), 1e-3));
    ASSERT_TRUE(P_new.isApprox(ekf.cov(), 1e-3));
  }
}

TEST(Ekf, PredictLinear)
{
  test_predict_linear<3>();
  test_predict_linear<6>();
  test_predict_linear<9>();
}

TEST(Ekf, PredictTimeCut)
{
  // initial state
  Eigen::Matrix<double, 2, 1> xhat = Eigen::Matrix<double, 2, 1>::Random();
  Eigen::Matrix<double, 2, 2> P    = Eigen::Matrix<double, 2, 1>::Random().asDiagonal();
  P.diagonal() += Eigen::Matrix<double, 2, 1>::Constant(1.1);

  smooth::feedback::EKF<smooth::Tn<2, double>> ekf;
  ekf.reset(xhat, P);

  // dynamical model
  Eigen::Matrix<double, 2, 1> b = Eigen::Matrix<double, 2, 1>::Random();
  Eigen::Matrix<double, 2, 2> Q = Eigen::Matrix<double, 2, 1>::Random().asDiagonal();
  Q.diagonal() += Eigen::Matrix<double, 2, 1>::Constant(1.1);

  // propagate
  double tau = 0.7;
  ekf.predict([&b](double, const auto &) { return b; }, Q, tau, 0.5);

  // exact solution
  ASSERT_TRUE(ekf.state().rn().isApprox(xhat + b * tau));
}
