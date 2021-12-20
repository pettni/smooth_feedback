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
 * @file Solve a optimal control problem on SE(2) as a nonlinear program.
 */

#include <Eigen/Core>
#include <smooth/bundle.hpp>
#include <smooth/feedback/compat/ipopt.hpp>
#include <smooth/feedback/ocp_to_nlp.hpp>
#include <smooth/se2.hpp>

#include <chrono>
#include <iostream>

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#endif

template<typename T>
using X = smooth::Bundle<smooth::SE2<T>, Eigen::Vector3<T>>;

template<typename T>
using U = Eigen::Vector2<T>;

template<typename T, std::size_t N>
using Vec = Eigen::Vector<T, N>;

/// @brief Objective function
const auto obj = []<typename T>(T tf, const X<T> &, const X<T> &, const Vec<T, 1> & q) -> T {
  return tf + q.x();
};

/// @brief Dynamics
const auto f = []<typename T>(T, const X<T> & x, const U<T> & u) -> smooth::Tangent<X<T>> {
  smooth::Tangent<X<T>> ret;
  ret.segment(0, 3) = x.template part<1>();
  ret(3)            = u.x();
  ret(4)            = T(0);
  ret(5)            = u.y();
  return ret;
};

/// @brief Integrals
const auto g = []<typename T>(T, const X<T> &, const U<T> & u) -> Vec<T, 1> {
  return Vec<T, 1>{{u.squaredNorm()}};
};

/// @brief Running constraints
const auto cr = []<typename T>(T, const X<T> &, const U<T> & u) -> Vec<T, 2> { return u; };

/// @brief End constraints
const auto ce = []<typename T>(T tf, const X<T> & x0, const X<T> & xf, const Vec<T, 1> &) -> Vec<T, 10> {
  const smooth::SE2<T> target(smooth::SO2<T>(-0.5), Eigen::Vector2<T>{2, 0.5});
  Vec<T, 10> ret(10);
  ret << tf, x0.template part<0>().log(), x0.template part<1>(), xf.template part<0>() - target;
  return ret;
};

/// @brief Range to std::vector
const auto r2v = [](std::ranges::range auto && r) {
  std::vector<std::ranges::range_value_t<decltype(r)>> ret;
  for (auto x : r) { ret.push_back(x); }
  return ret;
};

int main()
{
  // define optimal control problem
  smooth::feedback::
    OCP<X<double>, U<double>, decltype(obj), decltype(f), decltype(g), decltype(cr), decltype(ce)>
      ocp{
        .theta = obj,
        .f     = f,
        .g     = g,
        .cr    = cr,
        .crl   = Vec<double, 2>{{-1, -1}},
        .cru   = Vec<double, 2>{{1, 1}},
        .ce    = ce,
        .cel   = Vec<double, 10>{{3, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        .ceu   = Vec<double, 10>{{15, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
      };

  const auto xl = []<typename T>(T) -> X<T> { return X<T>::Identity(); };
  const auto ul = []<typename T>(T) -> U<T> { return Eigen::Vector2<T>::Constant(0.01); };

  const auto flatocp = smooth::feedback::flatten_ocp(ocp, xl, ul);

  // target optimality
  const double target_err = 1e-6;

  // define mesh
  smooth::feedback::Mesh<5, 10> mesh;

  // declare solution variable
  std::vector<typename decltype(ocp)::Solution> sols;
  std::optional<smooth::feedback::NLPSolution> nlpsol;

  const auto t0 = std::chrono::high_resolution_clock::now();

  for (auto iter = 0u; iter < 10; ++iter) {
    std::cout << "---------- ITERATION " << iter << " ----------" << std::endl;
    std::cout << "mesh: " << mesh.N_ivals() << " intervals, " << mesh.N_colloc()
              << " collocation pts" << std::endl;

    // transcribe optimal control problem to nonlinear programming problem
    const auto nlp = smooth::feedback::ocp_to_nlp(flatocp, mesh);

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
        {"hessian_approximation", "limited-memory"},
        {"derivative_test", "first-order"},
        // {"print_timing_statistics", "yes"},
      },
      {
        {"tol", 1e-6},
      });

    // convert solution of nlp insto solution of ocp
    auto flatsol = smooth::feedback::nlpsol_to_ocpsol(flatocp, mesh, nlpsol.value());

    // store unflattened solution
    sols.push_back(smooth::feedback::unflatten_ocpsol<X<double>, U<double>>(flatsol, xl, ul));

    // calculate errors
    auto errs = smooth::feedback::mesh_dyn_error(
      flatocp.Nx, flatocp.f, mesh, flatsol.t0, flatsol.tf, flatsol.x, flatsol.u);

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
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(tt, transform(tt, [&](double t) { return sol.x(t).part<1>().x(); }), "-r")->line_width(lw);
    plot(tt, transform(tt, [&](double t) { return sol.x(t).part<1>().y(); }), "-g")->line_width(lw);
    plot(tt, transform(tt, [&](double t) { return sol.x(t).part<1>().z(); }), "-b")->line_width(lw);
  }
  matplot::legend({"vx", "vy", "wz"});

  figure();
  hold(on);
  plot(tt_nodes, transform(tt_nodes, [](auto) { return 0; }), "xk")->marker_size(10);
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(tt, transform(tt, [&](double t) { return sol.lambda_dyn(t).x(); }), "-r")->line_width(lw);
    plot(tt, transform(tt, [&](double t) { return sol.lambda_dyn(t).y(); }), "-b")->line_width(lw);
  }
  matplot::legend({"nodes", "lambda_x", "lambda_y"});

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
