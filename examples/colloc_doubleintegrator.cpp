#include <Eigen/Core>
#define HAVE_CSTDDEF
#include <coin/IpIpoptApplication.hpp>
#undef HAVE_CSTDDEF
#include <iostream>
#include <matplot/matplot.h>

#include "smooth/feedback/compat/ipopt.hpp"
#include "smooth/feedback/nlp.hpp"

template<typename T>
using Vec = Eigen::VectorX<T>;

/// @brief Objective function
const auto theta = []<typename T>(
                     T t0, T tf, const Vec<T> & x0, const Vec<T> & xf, const Vec<T> & q) -> T {
  return q.x();
};

/// @brief Dynamics
const auto f = []<typename T>(T t, const Vec<T> & x, const Vec<T> & u) -> Vec<T> {
  return Vec<T>{{x.y(), u.x()}};
};

/// @brief Integrals
const auto g = []<typename T>(T t, const Vec<T> & x, const Vec<T> & u) -> Vec<T> {
  return Vec<T>{{x.squaredNorm() + u.squaredNorm()}};
};

/// @brief Running constraints
const auto cr = []<typename T>(
                  T, const Vec<T> & x, const Vec<T> & u) -> Vec<T> { return Vec<T>{{u.x()}}; };

/// @brief End constraints
const auto ce = []<typename T>(
                  T t0, T tf, const Vec<T> & x0, const Vec<T> & xf, const Vec<T> & q) -> Vec<T> {
  Vec<T> ret(5);
  ret << tf, x0, xf;
  return ret;
};

/// @brief Range to std::vector
const auto r2v = []<std::ranges::range R>(
                   const R & r) { return std::vector(std::ranges::begin(r), std::ranges::end(r)); };

int main()
{
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

  smooth::feedback::Mesh<5, 10> mesh;
  mesh.refine_ph(0, 8 * 5);

  const auto nlp = smooth::feedback::ocp_to_nlp(ocp, mesh);

  const auto nlp_sol = smooth::feedback::solve_nlp_ipopt(nlp);

  const auto [nodes, weights] = mesh.all_nodes_and_weights();

  const auto ocp_sol = smooth::feedback::nlp_sol_to_ocp_sol(ocp, mesh, nlp_sol);

  matplot::figure();
  matplot::plot(r2v(ocp_sol.tf * nodes), r2v(ocp_sol.X.row(0)), "-x")->line_width(2);
  matplot::title("pos");

  matplot::figure();
  matplot::plot(r2v(ocp_sol.tf * nodes), r2v(ocp_sol.X.row(1)), "-x")->line_width(2);
  matplot::title("vel");

  matplot::figure();
  matplot::plot(r2v(ocp_sol.tf * nodes), r2v(ocp_sol.U.row(0)), "-x")->line_width(2);
  matplot::title("input");

  matplot::show();

  return EXIT_SUCCESS;
}
