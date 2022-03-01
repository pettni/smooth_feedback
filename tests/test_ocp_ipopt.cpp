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
#include "smooth/feedback/ocp_to_nlp.hpp"

template<typename T, std::size_t N>
using Vec = Eigen::Vector<T, N>;

/// @brief Objective function
const auto theta =
  []<typename T>(T, const Vec<T, 2> &, const Vec<T, 2> &, const Vec<T, 1> & q) -> T {
  return q.x();
};

/// @brief Dynamics
const auto f = []<typename T>(T, const Vec<T, 2> & x, const Vec<T, 1> & u) -> Vec<T, 2> {
  return Vec<T, 2>{{x.y(), u.x()}};
};

/// @brief Integrals
const auto g = []<typename T>(T, const Vec<T, 2> & x, const Vec<T, 1> & u) -> Vec<T, 1> {
  return Vec<T, 1>{{x.squaredNorm() + u.squaredNorm()}};
};

/// @brief Running constraints
const auto cr = []<typename T>(T, const Vec<T, 2> &, const Vec<T, 1> & u) -> Vec<T, 1> {
  return Vec<T, 1>{{u.x()}};
};

/// @brief End constraints
const auto ce =
  []<typename T>(T tf, const Vec<T, 2> & x0, const Vec<T, 2> & xf, const Vec<T, 1> &) -> Vec<T, 5> {
  Vec<T, 5> ret(5);
  ret << tf, x0, xf;
  return ret;
};

/// @brief Range to std::vector
const auto r2v = []<std::ranges::range R>(const R & r) {
  return std::vector(std::ranges::begin(r), std::ranges::end(r));
};

TEST(OcpIpopt, Solve)
{
  // define optimal control problem
  smooth::feedback::OCP<
    Eigen::Vector2d,
    Vec<double, 1>,
    decltype(theta),
    decltype(f),
    decltype(g),
    decltype(cr),
    decltype(ce)>
    ocp{
      .theta = theta,
      .f     = f,
      .g     = g,
      .cr    = cr,
      .crl   = Vec<double, 1>{{-1}},
      .cru   = Vec<double, 1>{{1}},
      .ce    = ce,
      .cel   = Vec<double, 5>{{3, 1, 1, 0, 0}},
      .ceu   = Vec<double, 5>{{6, 1, 1, 0, 0}},
    };

  // define mesh
  smooth::feedback::Mesh mesh;
  mesh.refine_ph(0, 4 * 5);

  // transcribe optimal control problem to nonlinear programming problem
  const auto nlp = smooth::feedback::ocp_to_nlp(ocp, mesh);

  // solve nonlinear programming problem
  const auto nlp_sol = smooth::feedback::solve_nlp_ipopt(
    nlp,
    std::nullopt,
    {
      {"print_level", 0},
    },
    {
      {"linear_solver", "mumps"},
      {"hessian_approximation", "exact"},
    },
    {
      {"tol", 1e-8},
    });

  ASSERT_EQ(nlp_sol.status, smooth::feedback::NLPSolution::Status::Optimal);

  // convert solution of nlp insto solution of ocp
  const auto ocp_sol = smooth::feedback::nlpsol_to_ocpsol(ocp, mesh, nlp_sol);

  const auto nlp_sol_copy = smooth::feedback::ocpsol_to_nlpsol(ocp, mesh, ocp_sol);

  ASSERT_LE((nlp_sol_copy.x - nlp_sol.x).norm(), 1e-8);
  ASSERT_LE((nlp_sol_copy.zl - nlp_sol.zl).norm(), 1e-8);
  ASSERT_LE((nlp_sol_copy.zu - nlp_sol.zu).norm(), 1e-8);
  ASSERT_LE((nlp_sol_copy.lambda - nlp_sol.lambda).norm(), 1e-8);

  // solve again with warmstart
  const auto nlp_sol_warm = smooth::feedback::solve_nlp_ipopt(
    nlp,
    nlp_sol_copy,
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

  ASSERT_LE(nlp_sol_warm.iter, 6);
}
