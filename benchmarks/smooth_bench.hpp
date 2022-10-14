// Copyright (C) 2022 Petter Nilsson, John B. Mains. MIT License.

#pragma once

#include <smooth/feedback/qp_solver.hpp>

#include "bench_types.hpp"

template<typename Problem>
struct SmoothWrapper
{
  SmoothWrapper(const Problem & pbm, const smooth::feedback::QPSolverParams & prm) : qp_(pbm), prm_(prm) {}

  BenchResult operator()() const
  {
    auto t0  = std::chrono::high_resolution_clock::now();
    auto sol = smooth::feedback::solve_qp(qp_, prm_);
    auto t1  = std::chrono::high_resolution_clock::now();

    return {
      .status    = sol.code,
      .dt        = t1 - t0,
      .iter      = sol.iter,
      .solution  = sol.primal,
      .objective = (qp_.P.template selfadjointView<Eigen::Upper>() * (0.5 * sol.primal) + qp_.q).dot(sol.primal),
    };
  }

  Problem qp_;
  smooth::feedback::QPSolverParams prm_;
};
