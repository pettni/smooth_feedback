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

#include <Eigen/Core>
#include <smooth/feedback/compat/ipopt.hpp>
#include <smooth/feedback/ocp.hpp>

#include <chrono>
#include <iostream>

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#endif

template<typename T>
using Vec = Eigen::VectorX<T>;

/// @brief Objective function
const auto obj = []<typename T>(T tf, const Vec<T> &, const Vec<T> &, const Vec<T> & q) -> T {
  return tf + q.x();
};

/// @brief Dynamics
const auto f = []<typename T>(T, const Vec<T> & x, const Vec<T> & u) -> Vec<T> {
  return Vec<T>{{x.y(), u.x()}};
};

/// @brief Integrals
const auto g = []<typename T>(T, const Vec<T> & x, const Vec<T> & u) -> Vec<T> {
  return Vec<T>{{x.squaredNorm() + u.squaredNorm()}};
};

/// @brief Running constraints
const auto cr = []<typename T>(T, const Vec<T> &, const Vec<T> & u) -> Vec<T> {
  return Vec<T>{{u.x()}};
};

/// @brief End constraints
const auto ce =
  []<typename T>(T tf, const Vec<T> & x0, const Vec<T> & xf, const Vec<T> &) -> Vec<T> {
  Vec<T> ret(5);
  ret << tf, x0, xf;
  return ret;
};

/// @brief Range to std::vector
const auto r2v = [](std::ranges::range auto && r) {
  return std::vector(std::ranges::begin(r), std::ranges::end(r));
};

int main()
{
  // define optimal control problem
  smooth::feedback::FlatOCP<decltype(obj), decltype(f), decltype(g), decltype(cr), decltype(ce)>
    ocp{
      .nx    = 2,
      .nu    = 1,
      .nq    = 1,
      .ncr   = 1,
      .nce   = 5,
      .theta = obj,
      .f     = f,
      .g     = g,
      .cr    = cr,
      .crl   = Vec<double>{{-1}},
      .cru   = Vec<double>{{1}},
      .ce    = ce,
      .cel   = Vec<double>{{3, 1, 1, 0.1, 0}},
      .ceu   = Vec<double>{{15, 1, 1, 0.1, 0}},
    };

  // target optimality
  double target_err = 1e-6;

  // define mesh
  smooth::feedback::Mesh<5, 10> mesh;

  // declare solution variable
  std::vector<smooth::feedback::FlatOCPSolution> sols;
  std::optional<smooth::feedback::NLPSolution> nlpsol;

  const auto t0 = std::chrono::high_resolution_clock::now();

  for (auto iter = 0u; iter < 10; ++iter) {
    std::cout << "---------- ITERATION " << iter << " ----------" << std::endl;
    std::cout << "mesh: " << mesh.N_ivals() << " intervals, " << mesh.N_colloc()
              << " collocation pts" << std::endl;

    // transcribe optimal control problem to nonlinear programming problem
    const auto nlp = smooth::feedback::ocp_to_nlp(ocp, mesh);

    // solve nonlinear programming problem
    std::cout << "solving..." << std::endl;
    nlpsol = smooth::feedback::solve_nlp_ipopt(
      nlp,
      nlpsol,
      {
        {"print_level", 5},
      },
      {
        {"linear_solver", "mumps"}, {"hessian_approximation", "limited-memory"},
        // {"derivative_test", "first-order"},
        // {"print_timing_statistics", "yes"},
      },
      {
        {"tol", 1e-6},
      });

    // convert solution of nlp insto solution of ocp
    auto sol = smooth::feedback::nlpsol_to_ocpsol(ocp, mesh, nlpsol.value());
    sols.push_back(sol);

    // calculate errors
    auto errs = smooth::feedback::mesh_dyn_error(ocp.nx, f, mesh, sol.t0, sol.tf, sol.x, sol.u);

    std::cout << "interval errors " << errs.transpose() << std::endl;

    if (errs.maxCoeff() > target_err) {
      smooth::feedback::mesh_refine(mesh, errs, 0.1 * target_err);
      nlpsol = smooth::feedback::ocpsol_to_nlpsol(ocp, mesh, sol);
    } else {
      break;
    }
  }

  const auto dur = std::chrono::high_resolution_clock::now() - t0;

  std::cout << "TOTAL TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()
            << "ms" << std::endl;

#ifdef ENABLE_PLOTTING
  using namespace matplot;

  const auto [nodes, weights] = mesh.all_nodes_and_weights();

  const auto tt         = linspace(0., sols.back().tf, 500);
  const auto tt_nodes   = r2v(sols.back().tf * nodes);
  const auto tt_weights = r2v(weights);

  figure();
  hold(on);
  plot(tt_nodes, transform(tt_nodes, [](auto) { return 0; }), "xk")->marker_size(10);
  plot(tt_nodes, tt_weights, "or")->marker_size(5);
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(tt, transform(tt, [&](double t) { return sol.x(t).x(); }), "-r")->line_width(lw);
    plot(tt, transform(tt, [&](double t) { return sol.x(t).y(); }), "-b")->line_width(lw);
  }
  legend({"nodes", "weights", "pos", "vel"});

  figure();
  hold(on);
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(tt, transform(tt, [&](double t) { return sol.lambda_dyn(t).x(); }), "-r")->line_width(lw);
    plot(tt, transform(tt, [&](double t) { return sol.lambda_dyn(t).y(); }), "-b")->line_width(lw);
  }
  legend({"lambda_x", "lambda_y"});

  figure();
  hold(on);
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(tt, transform(tt, [&](double t) { return sol.lambda_cr(t).x(); }), "-r")->line_width(lw);
  }
  legend(std::vector<std::string>{"lambda_{cr}"});

  figure();
  hold(on);
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(tt, transform(tt, [&sol](double t) { return sol.u(t).x(); }), "-r")->line_width(lw);
  }
  legend(std::vector<std::string>{"input"});

  show();
#endif

  return EXIT_SUCCESS;
}
