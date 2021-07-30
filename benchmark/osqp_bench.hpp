#ifndef BENCHMARK__OSQP_BENCH_HPP_
#define BENCHMARK__OSQP_BENCH_HPP_

#include "bench_types.hpp"
#include <osqp/osqp.h>

#include "smooth/feedback/compat/osqp.hpp"

template<typename Scalar, typename Problem>
struct OsqpWrapper
{
  OsqpWrapper(const Problem & pbm, const smooth::feedback::SolverParams & prm) : qp_(pbm), prm_(prm)
  {}

  BenchResult operator()()
  {
    auto t0  = std::chrono::high_resolution_clock::now();
    auto sol = smooth::feedback::solve_qp_osqp(qp_, prm_);
    auto t1  = std::chrono::high_resolution_clock::now();
    return {.dt = t1 - t0,
      .iter     = sol.iter,
      .solution = sol.primal,
      .success  = sol.code == smooth::feedback::ExitCode::Optimal,
      .objective =
        (qp_.P.template selfadjointView<Eigen::Upper>() * (0.5 * sol.primal) + qp_.q).dot(sol.primal)};
  }

  Problem qp_;
  smooth::feedback::SolverParams prm_;
};

#endif  // BENCHMARK__OSQP_BENCH_HPP_
