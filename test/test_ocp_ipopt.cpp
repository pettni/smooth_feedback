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
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EVecPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "smooth/feedback/compat/ipopt.hpp"
#include "smooth/feedback/ocp.hpp"

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

TEST(OcpIpopt, Solve)
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

  // define mesh
  smooth::feedback::Mesh mesh;
  mesh.refine_ph(0, 4 * 5);
  const auto [nodes, weights] = mesh.all_nodes_and_weights();

  // transcribe optimal control problem to nonlinear programming problem
  const auto nlp = smooth::feedback::ocp_to_nlp(ocp, mesh);

  // solve nonlinear programming problem
  const auto nlp_sol = smooth::feedback::solve_nlp_ipopt(nlp,
    std::nullopt,
    {
      {"print_level", 0},
    },
    {
      {"linear_solver", "mumps"},
      {"hessian_approximation", "limited-memory"},
      {"print_timing_statistics", "yes"},
      {"derivative_test", "first-order"},
    },
    {
      {"tol", 1e-8},
    });

  ASSERT_EQ(nlp_sol.status, smooth::feedback::NLPSolution::Status::Optimal);

  // convert solution of nlp insto solution of ocp
  const auto ocp_sol = smooth::feedback::nlpsol_to_ocpsol(ocp, mesh, nlp_sol);

  const auto nlp_sol_copy = smooth::feedback::ocpsol_to_nlpsol(ocp, mesh, ocp_sol);

  std::cout << (nlp_sol.x - nlp_sol_copy.x).transpose() << std::endl;

  ASSERT_LE((nlp_sol_copy.x - nlp_sol.x).norm(), 1e-8);
  ASSERT_LE((nlp_sol_copy.zl - nlp_sol.zl).norm(), 1e-8);
  ASSERT_LE((nlp_sol_copy.zu - nlp_sol.zu).norm(), 1e-8);
  ASSERT_LE((nlp_sol_copy.lambda - nlp_sol.lambda).norm(), 1e-8);

  // solve again with warmstart
  const auto nlp_sol_warm = smooth::feedback::solve_nlp_ipopt(nlp,
    nlp_sol_copy,
    {
      {"print_level", 0},
    },
    {
      {"linear_solver", "mumps"},
      {"hessian_approximation", "limited-memory"},
      {"print_timing_statistics", "yes"},
      {"derivative_test", "first-order"},
    },
    {
      {"tol", 1e-8},
    });

  ASSERT_LE(nlp_sol_warm.iter, 8);
}
