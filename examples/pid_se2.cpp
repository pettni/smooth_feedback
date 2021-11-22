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

#include <boost/numeric/odeint.hpp>
#include <smooth/bundle.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/pid.hpp>
#include <smooth/se2.hpp>

#include <chrono>

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#endif

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;

using Time = std::chrono::duration<double>;

int main()
{
  smooth::feedback::PIDParams prm{};
  smooth::feedback::PID<Time, smooth::SE2d> pid(prm);

  // set desired trajectory
  Eigen::Vector3d vdes(1, 0, 0.4);
  auto xdes = [&vdes](Time t) -> decltype(pid)::TrajectoryReturnT {
    return std::make_tuple(
      smooth::SE2d(smooth::SO2d(M_PI_2), Eigen::Vector2d(2.5, 0)) + (t.count() * vdes),
      vdes,
      Eigen::Vector3d::Zero());
  };

  pid.set_xdes(xdes);

  // input variable
  Eigen::Vector2d u;

  // prepare for integrating the closed-loop system
  using State = smooth::Bundle<smooth::SE2d, Eigen::Vector3d>;
  using Deriv = typename State::Tangent;
  runge_kutta4<State, double, Deriv, double, vector_space_algebra> stepper{};
  const auto ode = [&u](const State & x, Deriv & d, double) {
    d.template head<3>() = x.part<1>();
    d.template tail<3>() << u(0), 0, u(1);
  };

  State x(smooth::SE2d::Identity(), Eigen::Vector3d::Zero());
  std::vector<double> tvec, xvec, yvec, u1vec, u2vec;

  // integrate closed-loop system
  for (std::chrono::milliseconds t = 0s; t < 30s; t += 50ms) {
    // compute input
    Eigen::Vector3d a = pid(t, x.part<0>(), x.part<1>());

    // input allocation from desired acceleration
    u(0) = std::clamp<double>(a(0), -1, 1);               // throtte <- a_x
    u(1) = std::clamp<double>(a(2) + 0.3 * a(1), -1, 1);  // steering <- a_Yaw + 0.3 a_y

    // store data
    tvec.push_back(duration_cast<Time>(t).count());
    xvec.push_back(x.part<0>().r2().x());
    yvec.push_back(x.part<0>().r2().y());

    u1vec.push_back(u(0));
    u2vec.push_back(u(1));

    // step dynamics
    stepper.do_step(ode, x, 0, 0.05);
  }

#ifdef ENABLE_PLOTTING
  matplot::figure();
  matplot::hold(matplot::on);
  matplot::title("Path");

  matplot::plot(xvec, yvec)->line_width(2);
  matplot::plot(
    matplot::transform(tvec, [&](auto t) { return std::get<0>(xdes(Time(t))).r2().x(); }),
    matplot::transform(tvec, [&](auto t) { return std::get<0>(xdes(Time(t))).r2().y(); }),
    "k--")
    ->line_width(2);
  matplot::legend({"actual", "desired"});

  matplot::figure();
  matplot::hold(matplot::on);
  matplot::title("Inputs");
  matplot::plot(tvec, u1vec)->line_width(2);
  matplot::plot(tvec, u2vec)->line_width(2);
  matplot::legend({"u1", "u2"});

  matplot::show();
#else
  std::cout << "TRAJECTORY:" << std::endl;
  for (auto i = 0u; i != tvec.size(); ++i) {
    std::cout << "t=" << tvec[i] << ": x=" << xvec[i] << ", y=" << yvec[i] << std::endl;
  }
#endif
}
