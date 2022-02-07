// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
 * @file Solve optimal control problem on SE(2) as a nonlinear program.
 */

#include <smooth/feedback/collocation/dyn_error.hpp>
#include <smooth/feedback/compat/ipopt.hpp>
#include <smooth/feedback/ocp_flatten.hpp>
#include <smooth/feedback/ocp_to_nlp.hpp>

#include <chrono>
#include <iostream>

#include "ocp_se2.hpp"

#ifdef ENABLE_PLOTTING
#include "common.hpp"
#include <matplot/matplot.h>
#endif

int main()
{
  std::cout << "TESTING LIE DERIVATIVES\n";
  smooth::feedback::test_ocp_derivatives(ocp_se2);

  const auto xl = []<typename T>(T) -> X<T> { return X<T>::Identity(); };
  const auto ul = []<typename T>(T) -> U<T> { return Eigen::Vector2<T>::Constant(0.01); };

  const auto flatocp = smooth::feedback::flatten_ocp(ocp_se2, xl, ul);
  std::cout << "TESTING FLAT DERIVATIVES\n";
  smooth::feedback::test_ocp_derivatives(flatocp);

  // target optimality
  const double target_err = 1e-6;

  // define mesh
  smooth::feedback::Mesh<5, 10> mesh;

  // declare solution variable
  std::vector<typename decltype(ocp_se2)::Solution> sols;
  std::optional<smooth::feedback::NLPSolution> nlpsol;

  const auto t0 = std::chrono::high_resolution_clock::now();

  for (auto iter = 0u; iter < 10; ++iter) {
    std::cout << "---------- ITERATION " << iter << " ----------" << std::endl;
    std::cout << "mesh: " << mesh.N_ivals() << " intervals, " << mesh.N_colloc()
              << " collocation pts" << std::endl;

    // transcribe optimal control problem to nonlinear programming problem
    const auto nlp = smooth::feedback::ocp_to_nlp<smooth::diff::Type::Analytic>(flatocp, mesh);

    // solve nonlinear programming problem
    std::cout << "solving..." << std::endl;
    nlpsol = smooth::feedback::solve_nlp_ipopt(
      nlp,
      nlpsol,
      {
        {"print_level", 5},
      },
      {
        {"linear_solver", "mumps"},
        {"hessian_approximation", "exact"},
        // {"derivative_test", "first-order"},
        {"print_timing_statistics", "yes"},
      },
      {
        {"tol", 1e-6},
      });

    // convert solution of nlp insto solution of ocp_se2
    auto flatsol = smooth::feedback::nlpsol_to_ocpsol(flatocp, mesh, nlpsol.value());

    // store unflattened solution
    sols.push_back(smooth::feedback::unflatten_ocpsol<X<double>, U<double>>(flatsol, xl, ul));

    // calculate errors
    mesh.increase_degrees();
    auto errs = smooth::feedback::mesh_dyn_error(
      flatocp.f, mesh, flatsol.t0, flatsol.tf, flatsol.x, flatsol.u);
    mesh.decrease_degrees();

    std::cout << "interval errors " << errs.transpose() << std::endl;

    if (errs.maxCoeff() > target_err) {
      mesh.refine_errors(errs, 0.1 * target_err);
      nlpsol = smooth::feedback::ocpsol_to_nlpsol(flatocp, mesh, flatsol);
    } else {
      break;
    }
  }

  const auto dur = std::chrono::high_resolution_clock::now() - t0;

  std::cout << "TOTAL TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()
            << "ms" << std::endl;

#ifdef ENABLE_PLOTTING
  using namespace matplot;

  const auto tt = linspace(0., sols.back().tf, 500);
  const auto tt_nodes =
    r2v(mesh.all_nodes() | std::views::transform([&](double d) { return d * sols.back().tf; }));

  figure();
  hold(on);
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(
      transform(tt, [&](double t) { return sol.x(t).part<0>().r2().x(); }),
      transform(tt, [&](double t) { return sol.x(t).part<0>().r2().y(); }),
      "-r")
      ->line_width(lw);
  }
  matplot::legend(std::vector<std::string>{"path"});

  figure();
  hold(on);
  plot(tt_nodes, transform(tt_nodes, [](auto) { return 0; }), "xk")->marker_size(10);
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(tt, transform(tt, [&](double t) { return sol.x(t).part<1>().x(); }), "-r")->line_width(lw);
    plot(tt, transform(tt, [&](double t) { return sol.x(t).part<1>().y(); }), "-b")->line_width(lw);
  }
  matplot::legend({"nodes", "vx", "wz"});

  figure();
  hold(on);
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(tt, transform(tt, [&](double t) { return sol.lambda_dyn(t).x(); }), "-r")->line_width(lw);
    plot(tt, transform(tt, [&](double t) { return sol.lambda_dyn(t).y(); }), "-b")->line_width(lw);
  }
  matplot::legend({"lambda_x", "lambda_y"});

  figure();
  hold(on);
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(tt, transform(tt, [&](double t) { return sol.lambda_cr(t).x(); }), "-r")->line_width(lw);
  }
  matplot::legend(std::vector<std::string>{"lambda_{cr}"});

  figure();
  hold(on);
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(tt, transform(tt, [&sol](double t) { return sol.u(t).x(); }), "-r")->line_width(lw);
    plot(tt, transform(tt, [&sol](double t) { return sol.u(t).y(); }), "-b")->line_width(lw);
  }
  matplot::legend({"throttle", "steering"});

  show();
#endif

  return EXIT_SUCCESS;
}
