#include <Eigen/Core>
#define HAVE_CSTDDEF
#include <coin/IpIpoptApplication.hpp>
#undef HAVE_CSTDDEF
#include "smooth/feedback/compat/ipopt.hpp"
#include "smooth/feedback/ocp.hpp"
#include <iostream>
#include <matplot/matplot.h>

template<typename T>
using Vec = Eigen::VectorX<T>;

/// @brief Objective function
const auto theta = []<typename T>(
                     T, T, const Vec<T> &, const Vec<T> &, const Vec<T> & q) -> T { return q.x(); };

/// @brief Dynamics
const auto f = []<typename T>(T, const Vec<T> & x, const Vec<T> & u) -> Vec<T> {
  return Vec<T>{{x.y(), u.x()}};
};

/// @brief Integrals
const auto g = []<typename T>(T, const Vec<T> & x, const Vec<T> & u) -> Vec<T> {
  return Vec<T>{{x.squaredNorm() + u.squaredNorm()}};
};

/// @brief Running constraints
const auto cr = []<typename T>(
                  T, const Vec<T> &, const Vec<T> & u) -> Vec<T> { return Vec<T>{{u.x()}}; };

/// @brief End constraints
const auto ce = []<typename T>(
                  T, T tf, const Vec<T> & x0, const Vec<T> & xf, const Vec<T> &) -> Vec<T> {
  Vec<T> ret(5);
  ret << tf, x0, xf;
  return ret;
};

/// @brief Range to std::vector
const auto r2v = []<std::ranges::range R>(
                   const R & r) { return std::vector(std::ranges::begin(r), std::ranges::end(r)); };

int main()
{
  // define optimal control problem
  smooth::feedback::OCP<decltype(theta), decltype(f), decltype(g), decltype(cr), decltype(ce)> ocp{
    .nx    = 2,
    .nu    = 1,
    .nq    = 1,
    .ncr   = 1,
    .nce   = 5,
    .theta = theta,
    .f     = f,
    .g     = g,
    .cr    = cr,
    .crl   = Vec<double>{{-1}},
    .cru   = Vec<double>{{1}},
    .ce    = ce,
    .cel   = Vec<double>{{3, 1, 1, 0, 0}},
    .ceu   = Vec<double>{{6, 1, 1, 0, 0}},
  };

  // target optimality
  double target_err = 1e-6;

  // define mesh
  smooth::feedback::Mesh mesh(5, 10);

  // declare solution variable
  smooth::feedback::OCPSolution sol;

  for (auto iter = 0u; iter < 10; ++iter) {
    std::cout << "---------- ITERATION " << iter << " ----------" << std::endl;
    std::cout << "mesh: " << mesh.N_ivals() << " intervals" << std::endl;
    std::cout << "colloc pts: " << mesh.all_nodes_and_weights().first.transpose() << std::endl;
    // transcribe optimal control problem to nonlinear programming problem
    const auto nlp = smooth::feedback::ocp_to_nlp(ocp, mesh);

    // solve nonlinear programming problem
    std::cout << "solving..." << std::endl;
    const auto nlp_sol = smooth::feedback::solve_nlp_ipopt(nlp,
      std::nullopt,
      {
        {"print_level", 0},
      },
      {
        {"linear_solver", "mumps"},
        {"hessian_approximation", "limited-memory"},
      },
      {
        {"tol", 1e-8},
      });

    // convert solution of nlp insto solution of ocp
    sol = smooth::feedback::nlpsol_to_ocpsol(ocp, mesh, nlp_sol);

    // calculate errors
    const auto errs =
      smooth::feedback::mesh_dyn_error(ocp.nx, f, mesh, sol.t0, sol.tf, sol.x, sol.u);
    const double maxerr = errs.maxCoeff();
    std::cout << "interval errors " << errs.transpose() << std::endl;

    if (maxerr > target_err) {
      smooth::feedback::mesh_refine(mesh, errs, target_err);
    } else {
      break;
    }
  }

#ifdef ENABLE_PLOTTING
  using namespace matplot;

  const auto [nodes, weights] = mesh.all_nodes_and_weights();

  const auto tt       = linspace(sol.t0, sol.tf, 500);
  const auto tt_nodes = r2v(sol.tf * nodes);

  figure();
  hold(on);
  plot(tt, transform(tt, [&](double t) { return sol.x(t).x(); }), "-r")->line_width(2);
  plot(tt, transform(tt, [&](double t) { return sol.x(t).y(); }), "-b")->line_width(2);
  plot(tt_nodes, transform(tt_nodes, [](auto) { return 0; }), "xk")->marker_size(10);
  matplot::legend({"pos", "vel", "nodes"});

  figure();
  hold(on);
  plot(tt, transform(tt, [&](double t) { return sol.lambda_dyn(t).x(); }), "-r")->line_width(2);
  plot(tt, transform(tt, [&](double t) { return sol.lambda_dyn(t).y(); }), "-b")->line_width(2);
  matplot::legend({"lambda_x", "lambda_y"});

  figure();
  hold(on);
  plot(tt, transform(tt, [&](double t) { return sol.lambda_cr(t).x(); }), "-r")->line_width(2);
  matplot::legend(std::vector<std::string>{"lambda_{cr}"});

  figure();
  plot(tt, transform(tt, [&sol](double t) { return sol.u(t).x(); }), "-")->line_width(2);
  matplot::legend(std::vector<std::string>{"input"});

  show();
#endif

  return EXIT_SUCCESS;
}
