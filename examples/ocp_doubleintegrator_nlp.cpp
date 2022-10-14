// Copyright (C) 2022 Petter Nilsson. MIT License.

/**
 * @file Solve a double integrator optimal control problem as a nonlinear program.
 */

#include <chrono>
#include <iostream>

#include <Eigen/Core>
#include <smooth/feedback/collocation/dyn_error.hpp>
#include <smooth/feedback/compat/ipopt.hpp>
#include <smooth/feedback/ocp_to_nlp.hpp>

#include "ocp_doubleintegrator.hpp"

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>

#include "common.hpp"
#endif

int main()
{
  smooth::feedback::test_ocp_derivatives(ocp_di);

  // target optimality
  double target_err = 1e-6;

  // define mesh
  smooth::feedback::Mesh<5, 10> mesh;

  // declare solution variable
  std::vector<decltype(ocp_di)::Solution> sols;
  std::optional<smooth::feedback::NLPSolution> nlpsol;

  const auto t0 = std::chrono::high_resolution_clock::now();

  for (auto iter = 0u; iter < 10; ++iter) {
    std::cout << "---------- ITERATION " << iter << " ----------" << std::endl;
    std::cout << "mesh: " << mesh.N_ivals() << " intervals, " << mesh.N_colloc() << " collocation pts" << std::endl;

    // transcribe optimal control problem to nonlinear programming problem
    const auto nlp = smooth::feedback::ocp_to_nlp<smooth::diff::Type::Analytic>(ocp_di, mesh);

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
        // {"derivative_test_print_all", "yes"},
        {"print_timing_statistics", "yes"},
      },
      {
        {"tol", 1e-6},
      });

    // convert solution of nlp insto solution of ocp_di
    auto sol = smooth::feedback::nlpsol_to_ocpsol(ocp_di, mesh, nlpsol.value());
    sols.push_back(sol);

    // calculate errors
    mesh.increase_degrees();
    auto errs = smooth::feedback::mesh_dyn_error(ocp_di.f, mesh, sol.t0, sol.tf, sol.x, sol.u);
    mesh.decrease_degrees();

    std::cout << "interval errors " << errs.transpose() << std::endl;

    if (errs.maxCoeff() > target_err) {
      mesh.refine_errors(errs, 0.1 * target_err);
      nlpsol = smooth::feedback::ocpsol_to_nlpsol(ocp_di, mesh, sol);
    } else {
      break;
    }
  }

  const auto dur = std::chrono::high_resolution_clock::now() - t0;

  std::cout << "TOTAL TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << "ms"
            << std::endl;

#ifdef ENABLE_PLOTTING
  using namespace matplot;

  const auto tt         = linspace(0., sols.back().tf, 500);
  const auto tt_nodes   = r2v(mesh.all_nodes() | std::views::transform([&](double d) { return d * sols.back().tf; }));
  const auto tt_weights = r2v(mesh.all_weights());

  figure();
  hold(on);
  plot(tt_nodes, transform(tt_nodes, [](auto) { return 0; }), "xk")->marker_size(10);
  plot(tt_nodes, tt_weights, "or")->marker_size(5);
  for (auto it = 0u; const auto & sol : sols) {
    int lw = it++ + 1 < sols.size() ? 1 : 2;
    plot(tt, transform(tt, [&](double t) { return sol.x(t).x(); }), "-r")->line_width(lw);
    plot(tt, transform(tt, [&](double t) { return sol.x(t).y(); }), "-b")->line_width(lw);
  }
  matplot::legend({"nodes", "weights", "pos", "vel"});

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
  }
  matplot::legend(std::vector<std::string>{"input"});

  show();
#endif

  return EXIT_SUCCESS;
}
