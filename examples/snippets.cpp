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

void ekf_snippet()
{
  smooth::feedback::EKF<smooth::SE2d> ekf;

  // motion model
  auto f = []<typename T>(double, const smooth::SE2<T> &) ->
    typename smooth::SE2<T>::Tangent { return typename smooth::SE2<T>::Tangent(0.4, 0.01, 0.1); };

  // measurement model
  Eigen::Vector2d landmark(1, 1);
  auto h = [&landmark]<typename T>(
             const smooth::SE2<T> & x) -> Eigen::Matrix<T, 2, 1> { return x.inverse() * landmark; };

  // PREDICT STEP: propagate filter over time
  ekf.predict(f,
    Eigen::Matrix3d::Identity(),  // motion covariance Q
    1.                            // time step length
  );

  // UPDATE STEP: register a measurement of the known landmark
  ekf.update(h,
    Eigen::Vector2d(0.3, 0.6),   // measurement result y
    Eigen::Matrix2d::Identity()  // measurement covariance R
  );

  // access estimate
  std::cout << ekf.estimate() << std::endl;
}

void qp_snippet()
{
  int n = 5;
  int m = 10;

  Eigen::MatrixXd P(n, n);
  Eigen::MatrixXd q(n, 1);

  Eigen::MatrixXd A(m, n);
  Eigen::MatrixXd l(m, 1);
  Eigen::MatrixXd u(m, 1);

  P.setRandom();
  q.setRandom();
  A.setRandom();
  l.setRandom();
  u.setRandom();

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

template<typename T>
using X = smooth::Bundle<smooth::SE2<T>, Eigen::Matrix<T, 3, 1>>;
template<typename T>
using U    = Eigen::Matrix<T, 2, 1>;
using Time = std::chrono::duration<double>;

void mpc_snippet()
{
  // dynamics
  auto f = []<typename T>(Time, const X<T> & x, const U<T> & u) -> typename X<T>::Tangent {
    typename X<T>::Tangent dx_dt;
    dx_dt.template head<3>() = x.template part<1>();
    dx_dt.template tail<3>() << -T(0.2) * x.template part<1>().x() + u(0), T(0),
      -T(0.4) * x.template part<1>().z() + u(1);
    return dx_dt;
  };

  // create MPC object
  smooth::feedback::MPCParams prm{.T = 5};
  smooth::feedback::MPC<50, Time, X<double>, U<double>, decltype(f)> mpc(f, prm);

  // set desired state and input trajectories for MPC to track
  mpc.set_xdes([](Time t) -> std::pair<X<double>, smooth::Tangent<X<double>>> {
    return {
      X<double>::Identity(),
      smooth::Tangent<X<double>>::Zero(),
    };
  });
  mpc.set_udes([](Time T) -> U<double> { return U<double>::Zero(); });

  // calculate control input for current time t and current state x
  Time t(0);
  X<double> x = X<double>::Identity();
  U<double> u;
  mpc(u, t, x);
}

void pid_snippet()
{
  smooth::feedback::PID<Time, smooth::SE2d> pid;

  // set desired motion as a function Time -> (position, velocity, acceleration)
  pid.set_xdes([](Time Time) -> std::tuple<smooth::SE2d, Eigen::Vector3d, Eigen::Vector3d> {
    return std::make_tuple(
      smooth::SE2d::Identity(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  });

  Time t(1);          // current time
  smooth::SE2d x;     // current state
  Eigen::Vector3d v;  // current body velocity

  Eigen::Vector3d u = pid(t, x, v);
}

void asif_snippet()
{
  // define dynamics
  auto f = []<typename T>(T, const X<T> & x, const U<T> & u) -> smooth::Tangent<X<T>> {
    typename X<T>::Tangent dx_dt;
    dx_dt.template head<3>() = x.template part<1>();
    dx_dt.template tail<3>() << -T(0.2) * x.template part<1>().x() + u(0), T(0),
      -T(0.4) * x.template part<1>().z() + u(1);
    return dx_dt;
  };

  // define safety set { x : h(x) >= 0 }
  auto h = []<typename T>(T, const X<T> & x) -> Eigen::Vector3<T> {
    return x.template part<0>().log() - Eigen::Vector3<T>(0.2, 0.2, 0.2);
  };

  // backup controller
  auto bu = []<typename T>(T, const X<T> &) -> U<T> { return U<T>(1, 1); };

  using ASIF =
    smooth::feedback::ASIFilter<100, X<double>, U<double>, decltype(f), decltype(h), decltype(bu)>;

  ASIF asif(f, h, bu);

  U<double> u = U<double>::Zero();
  double t    = 0;
  X<double> x = X<double>::Random();
  auto code   = asif(u, 0, x);
}

int main()
{
  ekf_snippet();
  mpc_snippet();
  pid_snippet();
  qp_snippet();
  asif_snippet();
}
