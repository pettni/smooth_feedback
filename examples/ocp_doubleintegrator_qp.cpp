// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
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

/**
 * @file Solve a double integrator optimal control problem as a quadratic program.
 */

#include <smooth/feedback/ocp_to_qp.hpp>
#include <smooth/feedback/qp_solver.hpp>

#include <chrono>
#include <iostream>

#include "ocp_doubleintegrator.hpp"

#ifdef ENABLE_PLOTTING
#include "common.hpp"
#include <matplot/matplot.h>
#endif

using namespace std::chrono;

int main()
{
  // define mesh
  smooth::feedback::Mesh<4, 4> mesh;
  mesh.refine_ph(0, 40);

  const auto tf     = ocp_di.cel.x();  // grabbing constraint on tf..
  const auto xl_fun = []<typename T>(T) -> X<T> { return X<T>::Zero(); };
  const auto ul_fun = []<typename T>(T) -> U<T> { return U<T>::Zero(); };

  const auto t0 = high_resolution_clock::now();

  const auto qp = ocp_to_qp(ocp_di, mesh, tf, xl_fun, ul_fun);

  const auto t1 = high_resolution_clock::now();

  const auto qpsol =
    smooth::feedback::solve_qp(qp, smooth::feedback::QPSolverParams{.verbose = true});

  const auto t2 = high_resolution_clock::now();

  const auto ocpsol = smooth::feedback::qpsol_to_ocpsol(ocp_di, mesh, qpsol, tf, xl_fun, ul_fun);

  std::cout << "ocp_to_qp      : " << duration_cast<microseconds>(t1 - t0).count() << '\n';
  std::cout << "solve_qp       : " << duration_cast<microseconds>(t2 - t1).count() << '\n';

#ifdef ENABLE_PLOTTING
  using namespace matplot;

  const auto tt = linspace(0., ocpsol.tf, 500);
  const auto tt_nodes =
    r2v(mesh.all_nodes() | std::views::transform([&](double d) { return d * ocpsol.tf; }));
  const auto tt_weights = r2v(mesh.all_weights());

  figure();
  hold(on);
  plot(tt_nodes, transform(tt_nodes, [](auto) { return 0; }), "xk")->marker_size(10);
  plot(tt_nodes, tt_weights, "or")->marker_size(5);
  plot(tt, transform(tt, [&](double t) { return ocpsol.x(t).x(); }), "-r")->line_width(2.);
  plot(tt, transform(tt, [&](double t) { return ocpsol.x(t).y(); }), "-b")->line_width(2.);
  legend({"nodes", "weights", "pos", "vel"});

  figure();
  hold(on);
  plot(tt, transform(tt, [&ocpsol](double t) { return ocpsol.u(t).x(); }), "-r")->line_width(2.);
  legend(std::vector<std::string>{"input"});

  show();
#endif

  return EXIT_SUCCESS;
}
