// Copyright (C) 2022 Petter Nilsson. MIT License.

/**
 * @file Solve optimal control problem on SE(2) as a quadratic program.
 */

#include <chrono>
#include <iostream>

#include <smooth/feedback/ocp_to_qp.hpp>
#include <smooth/feedback/qp_solver.hpp>

#include "ocp_se2.hpp"

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>

#include "common.hpp"
#endif

using namespace std::chrono;

int main()
{
  smooth::feedback::Mesh<5, 5> mesh;
  mesh.refine_ph(0, 50);

  const auto tf     = ocp_se2.cel.x();  // grabbing constraint on tf..
  const auto ul_fun = []<typename T>(T) -> U<T> { return Eigen::Vector2<T>{0, 0}; };

  const auto t0 = high_resolution_clock::now();

  const auto qp = ocp_to_qp(ocp_se2, mesh, tf, xdes, ul_fun);  // linearize around desired x

  Eigen::MatrixXd lu(qp.u.rows(), 2);
  lu.col(0) = qp.l;
  lu.col(1) = qp.u;

  const auto t1 = high_resolution_clock::now();

  const auto qpsol = smooth::feedback::solve_qp(qp, smooth::feedback::QPSolverParams{.verbose = true});

  const auto t2 = high_resolution_clock::now();

  const auto ocpsol = smooth::feedback::qpsol_to_ocpsol(ocp_se2, mesh, qpsol, tf, xdes, ul_fun);

  std::cout << "ocp_to_qp      : " << duration_cast<microseconds>(t1 - t0).count() << '\n';
  std::cout << "solve_qp       : " << duration_cast<microseconds>(t2 - t1).count() << '\n';

#ifdef ENABLE_PLOTTING
  using namespace matplot;

  const auto tt         = linspace(0., ocpsol.tf, 500);
  const auto tt_nodes   = r2v(mesh.all_nodes() | std::views::transform([&](double d) { return d * ocpsol.tf; }));
  const auto tt_weights = r2v(mesh.all_weights());

  figure();
  hold(on);
  plot(
    transform(tt, [&](double t) { return ocpsol.x(t).part<0>().r2().x(); }),
    transform(tt, [&](double t) { return ocpsol.x(t).part<0>().r2().y(); }),
    "-r")
    ->line_width(2);
  matplot::legend(std::vector<std::string>{"path"});

  figure();
  hold(on);
  plot(tt_nodes, transform(tt_nodes, [](auto) { return 0; }), "xk")->marker_size(10);
  plot(tt, transform(tt, [&](double t) { return ocpsol.x(t).part<1>().x(); }), "-r")->line_width(2);
  plot(tt, transform(tt, [&](double t) { return ocpsol.x(t).part<1>().y(); }), "-b")->line_width(2);
  matplot::legend({"nodes", "vx", "wz"});

  figure();
  hold(on);
  plot(tt, transform(tt, [&](double t) { return ocpsol.u(t).x(); }), "-r")->line_width(2);
  plot(tt, transform(tt, [&](double t) { return ocpsol.u(t).y(); }), "-b")->line_width(2);
  matplot::legend({"throttle", "steering"});

  show();
#endif

  return EXIT_SUCCESS;
}
