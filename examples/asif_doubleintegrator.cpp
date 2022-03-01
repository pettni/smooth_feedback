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
#include <smooth/feedback/asif.hpp>

#include <chrono>

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#endif

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;

template<typename T>
using G = Eigen::Matrix<T, 2, 1>;
template<typename T>
using U = Eigen::Matrix<T, 1, 1>;

using Gd = G<double>;
using Ud = U<double>;

int main()
{
  // dynamics
  auto f = []<typename T>(const G<T> & x, const U<T> & u) -> smooth::Tangent<G<T>> {
    return {x(1), u(0)};
  };

  // safety set
  auto h = []<typename T>(T, const G<T> & g) -> Eigen::Vector2<T> {
    return {T(3) - g(0), T(1.5) - g(1)};
  };

  // backup controller
  auto bu = []<typename T>(T, const G<T> &) -> U<T> { return U<T>(-0.6); };

  // parameters
  smooth::feedback::ASIFilterParams<Ud> prm{
    .T  = 0.5,
    .nh = 2,
    .ulim =
      {
        .A = U<double>{{1.}},
        .l = U<double>{{-1.}},
        .u = U<double>{{1.}},
      },
    .asif =
      {
        .K          = 50,
        .alpha      = 1,
        .dt         = 0.01,
        .relax_cost = 100,
      },
    .qp =
      {
        .polish = false,
      },
  };

  // create filter
  smooth::feedback::ASIFilter<Gd, Ud, decltype(f)> asif(f, prm);

  // system variables
  Gd g(-5, 1);
  Ud udes = Ud(1), u;

  // prepare for integrating the closed-loop system
  runge_kutta4<Gd, double, smooth::Tangent<Gd>, double, vector_space_algebra> stepper{};
  const auto ode = [&f, &u](const Gd & x, smooth::Tangent<Gd> & d, double) { d = f(x, u); };
  std::vector<double> tvec, xvec, vvec, uvec;

  // integrate closed-loop system
  for (std::chrono::milliseconds t = 0s; t < 10s; t += 50ms) {
    auto [u_asif, code] = asif(g, udes, h, bu);

    u = u_asif;

    if (code != smooth::feedback::QPSolutionStatus::Optimal) {
      std::cerr << "Solver failed with code " << static_cast<int>(code) << std::endl;
    }

    // store data
    tvec.push_back(duration_cast<std::chrono::duration<double>>(t).count());
    xvec.push_back(g.x());
    vvec.push_back(g.y());
    uvec.push_back(u.x());

    // step dynamics
    stepper.do_step(ode, g, 0, 0.05);
  }

#ifdef ENABLE_PLOTTING
  using namespace matplot;

  figure();
  hold(on);

  plot(tvec, xvec)->line_width(2);
  plot(tvec, vvec)->line_width(2);
  plot(tvec, transform(tvec, [&](auto) { return 3; }), "--")->line_width(2);
  plot(tvec, transform(tvec, [&](auto) { return 1.5; }), "--")->line_width(2);
  legend({"x", "v", "x_{max}", "v_{max}"});

  figure();
  hold(on);
  title("Input");
  plot(tvec, uvec)->line_width(2);
  plot(tvec, transform(tvec, [](auto) { return 1; }), "--")->line_width(2);

  show();
#else
  std::cout << "TRAJECTORY:" << std::endl;
  for (auto i = 0u; i != tvec.size(); ++i) {
    std::cout << "t=" << tvec[i] << ": x=" << xvec[i] << ", v=" << vvec[i] << std::endl;
  }
#endif
}
