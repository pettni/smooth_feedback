// Copyright (C) 2022 Petter Nilsson. MIT License.

#include <iostream>

#include <smooth/bundle.hpp>
#include <smooth/feedback/asif.hpp>
#include <smooth/feedback/ekf.hpp>
#include <smooth/feedback/mpc.hpp>
#include <smooth/feedback/pid.hpp>
#include <smooth/feedback/qp.hpp>
#include <smooth/se2.hpp>

using T = std::chrono::duration<double>;
template<typename S>
using X = smooth::Bundle<smooth::SE2<S>, Eigen::Matrix<S, 3, 1>>;
template<typename S>
using U = Eigen::Matrix<S, 2, 1>;

const Eigen::Matrix3d A{
  {-0.2, 0, 0},
  {0, 0, 0},
  {0, 0, -0.4},
};
const Eigen::Matrix<double, 3, 2> B{
  {1, 0},
  {0, 0},
  {0, 1},
};

auto Sigma = []<typename S>(const X<S> & x, const U<S> & u) -> smooth::Tangent<X<S>> {
  smooth::Tangent<X<S>> dx_dt;
  dx_dt.head(3) = x.template part<1>();
  dx_dt.tail(3) = A * x.template part<1>() + B * u;
  return dx_dt;
};

void ekf_snippet()
{
  // variable that holds current input
  U<double> u = U<double>::Random();
  // closed-loop dynamics
  auto SigmaCL = [&u]<typename S>(double, const X<S> & x) -> smooth::Tangent<X<S>> {
    return Sigma(x, u.template cast<S>());
  };

  // create filter
  smooth::feedback::EKF<X<double>> ekf;

  // measurement model
  Eigen::Vector2d landmark(1, 1);
  auto h = [&landmark]<typename S>(const X<S> & x) -> Eigen::Matrix<S, 2, 1> {
    return x.template part<0>().inverse() * landmark;
  };

  // PREDICT STEP: propagate filter over time
  ekf.predict(
    SigmaCL,
    Eigen::Matrix<double, 6, 6>::Identity(),  // motion covariance Q
    1.                                        // time step length
  );

  // UPDATE STEP: register a measurement of the known landmark
  ekf.update(
    h,
    Eigen::Vector2d(0.3, 0.6),   // measurement result y
    Eigen::Matrix2d::Identity()  // measurement covariance R
  );

  // access estimate
  auto x_hat = ekf.estimate();
  auto P_hat = ekf.covariance();
}

void pid_snippet()
{
  smooth::feedback::PID<T, smooth::SE2d> pid;

  // set desired motion
  pid.set_xdes([](T T) -> std::tuple<smooth::SE2d, Eigen::Vector3d, Eigen::Vector3d> {
    return {
      smooth::SE2d::Identity(),  // position
      Eigen::Vector3d::Zero(),   // velocity (right derivative of position w.r.t. t)
      Eigen::Vector3d::Zero(),   // acceleration (second right derivative of position w.r.t. t)
    };
  });

  T t               = T(1);                       // current time
  smooth::SE2d x    = smooth::SE2d::Random();     // current state
  Eigen::Vector3d v = Eigen::Vector3d::Random();  // current body velocity

  Eigen::Vector3d u = pid(t, x, v);
}

void asif_snippet()
{
  smooth::feedback::ASIFilter<X<double>, U<double>, decltype(Sigma)> asif(Sigma);

  // safety set S(t) = { x : h(t, x) >= 0 }
  auto h = []<typename S>(S, const X<S> & x) -> Eigen::Matrix<S, 1, 1> {
    return Eigen::Matrix<S, 1, 1>(x.template part<0>().r2().x() - S(0.2));
  };

  // backup controller
  auto bu = []<typename S>(S, const X<S> &) -> U<S> { return U<S>(1, 1); };

  T t             = T(1);
  X<double> x     = X<double>::Random();
  U<double> u_des = U<double>::Zero();

  // get control input for time t, state x, and reference input u_des
  auto [u_asif, code] = asif(x, u_des, h, bu);
}

void mpc_snippet()
{
  // state/input constraint: -crl \leq cr(x, u) \leq crl
  auto cr = []<typename S>(const X<S> &, const U<S> & u) -> Eigen::Vector<S, 2> { return u; };
  Eigen::Vector2d crl{1, 0.5};

  smooth::feedback::MPC<T, X<double>, U<double>, decltype(Sigma), decltype(cr)> mpc{
    Sigma,
    cr,
    -crl,
    crl,
    {.K = 5, .tf = 5},
  };

  // set desired input and state trajectories
  mpc.set_udes_rel([]<typename S>(S t) -> U<S> { return U<S>::Zero(); });
  mpc.set_xdes_rel([]<typename S>(S t) -> X<S> { return X<S>::Identity(); });

  T t(0);
  X<double> x = X<double>::Identity();

  // get control input for time t and state x
  auto [u, code] = mpc(t, x);
}

void qp_snippet()
{
  int n = 5;
  int m = 10;

  Eigen::MatrixXd P = Eigen::MatrixXd::Random(n, n);
  Eigen::VectorXd q = Eigen::VectorXd::Random(n);

  Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
  Eigen::VectorXd l = Eigen::VectorXd::Random(m);
  Eigen::VectorXd u = Eigen::VectorXd::Random(m);

  // define the QP
  //  min 0.5 x' P x + q' x
  //  s.t l <= Ax <= u
  smooth::feedback::QuadraticProgram<-1, -1> qp{
    .P = P,  // n x n matrix
    .q = q,  // n x 1 vector
    .A = A,  // m x n matrix
    .l = l,  // m x 1 vector
    .u = u,  // m x 1 vector
  };

  smooth::feedback::QPSolverParams prm{};
  auto sol = smooth::feedback::solve_qp(qp, prm);
}

int main()
{
  ekf_snippet();
  pid_snippet();
  asif_snippet();
  mpc_snippet();
  qp_snippet();
}
