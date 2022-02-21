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
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef SMOOTH__FEEDBACK__COLLOCATION__DYN_ERROR_HPP_
#define SMOOTH__FEEDBACK__COLLOCATION__DYN_ERROR_HPP_

/**
 * @file
 * @brief Collocation dynamics constraints.
 */

#include <Eigen/Core>

#include "mesh.hpp"

namespace smooth::feedback {

/**
 * @brief Calculate relative dynamics errors for each interval in mesh.
 *
 * @param f dynamics function
 * @param m Mesh
 * @param t0 initial time variable
 * @param tf final time variable
 * @param xfun state trajectory
 * @param ufun input trajectory
 *
 * @return vector with relative errors for every interval in m
 */
Eigen::VectorXd mesh_dyn_error(
  auto && f, const MeshType auto & m, const double t0, const double tf, auto && xfun, auto && ufun)
{
  using smooth::utils::zip;
  static constexpr auto Nx = Dof<std::invoke_result_t<decltype(xfun), double>>;

  static_assert(Nx > 0, "Static size required");

  const auto N = m.N_ivals();

  Eigen::VectorXd ival_errs(N);

  // for each interval
  for (auto ival = 0u, M = 0u; ival < N; M += m.N_colloc_ival(ival), ++ival) {
    const auto Kext = m.N_colloc_ival(ival);

    // evaluate xs and F at those points
    Eigen::Matrix<double, Nx, -1> Fval(Nx, Kext + 1);
    Eigen::Matrix<double, Nx, -1> Xval(Nx, Kext + 1);
    for (const auto & [j, tau] : zip(std::views::iota(0u, Kext + 1), m.interval_nodes(ival))) {
      const double tj = t0 + (tf - t0) * tau;

      // evaluate x and u values at tj using current degree polynomials
      const auto Xj = xfun(tj);
      const auto Uj = ufun(tj);

      // evaluate right-hand side of dynamics at tj
      Fval.col(j) = f(tj, Xj, Uj);

      // store x values for later comparison
      Xval.col(j) = Xj;
    }

    // "integrate" system inside interval
    const Eigen::MatrixXd Xval_est =
      Xval.col(0).replicate(1, Kext) + (tf - t0) * Fval.leftCols(Kext) * m.interval_intmat(ival);

    // absolute error in interval
    Eigen::VectorXd e_abs = (Xval_est - Xval.rightCols(Kext)).colwise().norm();
    Eigen::VectorXd e_rel = e_abs / (1. + Xval.rightCols(Kext).colwise().norm().maxCoeff());

    // mex relative error on interval
    ival_errs(ival) = e_rel.maxCoeff();
  }

  return ival_errs;
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION__DYN_ERROR_HPP_
