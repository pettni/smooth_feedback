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

#include <smooth/bundle.hpp>
#include <smooth/feedback/asif.hpp>
#include <smooth/feedback/ekf.hpp>
#include <smooth/feedback/mpc.hpp>
#include <smooth/feedback/pid.hpp>
#include <smooth/feedback/qp.hpp>
#include <smooth/se2.hpp>

#include <iostream>

using Time = std::chrono::duration<double>;
template<typename T>
using X = smooth::Bundle<smooth::SE2<T>, Eigen::Matrix<T, 3, 1>>;
template<typename T>
using U = Eigen::Matrix<T, 2, 1>;

auto dyn = []<typename T>(Time, const X<T> & x, const U<T> & u) -> smooth::Tangent<X<T>> {
  return smooth::Tangent<X<T>>{
    x.template part<1>().x(),
    x.template part<1>().y(),
    x.template part<1>().z(),
    -T(0.2) * x.template part<1>().x() + u(0),
    T(0),
    -T(0.4) * x.template part<1>().z() + u(1),
  };
};

void ekf_snippet()
{
  U<double> u = U<double>::Random();
  // closed-loop dynamics
  auto dyn_cl = [&u]<typename T>(T t, const X<T> & x) { return dyn(Time(t), x, u.template cast<T>()); };

  smooth::feedback::EKF<X<double>> ekf;

  // measurement model
  Eigen::Vector2d landmark(1, 1);
  auto h = [&landmark]<typename T>(const X<T> & x) -> Eigen::Matrix<T, 2, 1> {
    return x.template part<0>().inverse() * landmark;
  };

  // PREDICT STEP: propagate filter over time
  ekf.predict(dyn_cl,
    Eigen::Matrix<double, 6, 6>::Identity(),  // motion covariance Q
    1.                                        // time step length
  );

  // UPDATE STEP: register a measurement of the known landmark
  ekf.update(h,
    Eigen::Vector2d(0.3, 0.6),   // measurement result y
    Eigen::Matrix2d::Identity()  // measurement covariance R
  );

  // access estimate
  auto x_hat = ekf.estimate();
  auto P_hat = ekf.covariance();
}

void pid_snippet()
{
  smooth::feedback::PID<Time, smooth::SE2d> pid;

  // set desired motion
  pid.set_xdes([](Time Time) -> std::tuple<smooth::SE2d, Eigen::Vector3d, Eigen::Vector3d> {
    return {
      smooth::SE2d::Identity(),  // position
      Eigen::Vector3d::Zero(),   // velocity (right derivative of position w.r.t. t)
      Eigen::Vector3d::Zero(),   // acceleration (second right derivative of position w.r.t. t)
    };
  });

  Time t            = Time(1);                    // current time
  smooth::SE2d x    = smooth::SE2d::Random();     // current state
  Eigen::Vector3d v = Eigen::Vector3d::Random();  // current body velocity

  Eigen::Vector3d u = pid(t, x, v);
}

void asif_snippet()
{
  smooth::feedback::ASIFilter<Time, X<double>, U<double>, decltype(dyn)> asif(dyn);

  // safety set S(t) = { x : h(t, x) >= 0 }
  auto h = []<typename T>(T, const X<T> & x) -> Eigen::Matrix<T, 1, 1> {
    return Eigen::Matrix<T, 1, 1>(x.template part<0>().r2().x() - T(0.2));
  };

  // backup controller
  auto bu = []<typename T>(T, const X<T> &) -> U<T> { return U<T>(1, 1); };

  Time t          = Time(1);
  X<double> x     = X<double>::Random();
  U<double> u_des = U<double>::Zero();

  // get control input for time t, state x, and reference input u_des
  auto [u_asif, code] = asif(t, x, u_des, h, bu);
}

void mpc_snippet()
{
  smooth::feedback::MPC<Time, X<double>, U<double>, decltype(dyn)> mpc(dyn, {.T = 5, .K = 50});

  // set desired input and state trajectories
  mpc.set_udes([]<typename T>(T t) -> U<T> { return U<T>::Zero(); });
  mpc.set_xdes([]<typename T>(T t) -> X<T> { return X<T>::Identity(); });

  Time t(0);
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
