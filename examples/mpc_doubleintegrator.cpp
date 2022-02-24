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
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/mpc.hpp>

#include <chrono>

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#endif

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;

using Time = std::chrono::duration<double>;

template<typename T>
using X = Eigen::Vector2<T>;
template<typename T>
using U = Eigen::Matrix<T, 1, 1>;

using Gd = X<double>;
using Ud = U<double>;

int main()
{
  using std::sin;
  std::srand(5);

  // system variables
  Gd g = Gd::Random();
  Ud u;

  // dynamics
  auto f = []<typename T>(T, const X<T> & x, const U<T> u) -> smooth::Tangent<X<T>> {
    return {x(1), u(0)};
  };

  // running constraints
  auto cr = []<typename T>(T, const X<T> &, const U<T> & u) -> Eigen::Vector<T, 1> { return u; };
  Eigen::Vector<double, 1> crl{-0.5}, cru{0.5};

  // create MPC object and set input bounds, and desired trajectories
  smooth::feedback::MPC<20, Time, Gd, Ud, decltype(f), decltype(cr)> mpc{
    f,
    cr,
    crl,
    cru,
    smooth::feedback::MPCParams{
      .tf = 5,
    },
  };

  mpc.set_weights({
    .Q  = Eigen::Matrix2d::Identity(),
    .QT = 0.1 * Eigen::Matrix2d::Identity(),
    .R  = 0.1 * Eigen::Matrix<double, 1, 1>::Identity(),
  });
  mpc.set_xdes_rel([]<typename T>(T t) -> X<T> { return X<T>{-0.5 * sin(0.3 * t), 0}; });
  mpc.set_udes_rel([]<typename T>(T) -> U<T> { return U<T>::Zero(); });

  // prepare for integrating the closed-loop system
  runge_kutta4<Gd, double, smooth::Tangent<Gd>, double, vector_space_algebra> stepper{};
  const auto ode = [&f, &u](const Gd & x, smooth::Tangent<Gd> & d, double t) -> void {
    d = f(t, x, u);
  };
  std::vector<double> tvec, xvec, vvec, uvec;

  // integrate closed-loop system
  for (std::chrono::milliseconds t = 0s; t < 60s; t += 50ms) {
    // compute MPC input
    auto [u_mpc, code] = mpc(t, g);
    u                  = u_mpc;
    if (code != smooth::feedback::QPSolutionStatus::Optimal) {
      std::cerr << "Solver failed with code " << static_cast<int>(code) << std::endl;
    }

    // store data
    tvec.push_back(duration_cast<Time>(t).count());
    xvec.push_back(g.x());
    vvec.push_back(g.y());
    uvec.push_back(u(0));

    // step dynamics
    stepper.do_step(ode, g, 0, 0.05);
  }

#ifdef ENABLE_PLOTTING
  matplot::figure();
  matplot::hold(matplot::on);

  matplot::plot(tvec, xvec)->line_width(2);
  matplot::plot(tvec, matplot::transform(tvec, [](auto t) { return -0.5 * sin(0.3 * t); }), "k--")
    ->line_width(2);
  matplot::plot(tvec, vvec)->line_width(2);
  matplot::plot(tvec, uvec)->line_width(2);
  matplot::legend({"x", "x_{des}", "v", "u"});

  matplot::show();
#else
  std::cout << "TRAJECTORY:" << std::endl;
  for (auto i = 0u; i != tvec.size(); ++i) {
    std::cout << "t=" << tvec[i] << ": x=" << xvec[i] << ", v=" << vvec[i] << std::endl;
  }
#endif
}
