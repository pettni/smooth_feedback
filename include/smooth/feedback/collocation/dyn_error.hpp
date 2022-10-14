// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

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
Eigen::VectorXd
mesh_dyn_error(auto && f, const MeshType auto & m, const double t0, const double tf, auto && xfun, auto && ufun)
{
  using smooth::utils::zip;
  static constexpr auto Nx = std::invoke_result_t<decltype(xfun), double>::SizeAtCompileTime;

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
