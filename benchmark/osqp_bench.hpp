#ifndef BENCHMARK__OSQP_BENCH_HPP_
#define BENCHMARK__OSQP_BENCH_HPP_

#include <cbr_control/osqp-cpp.hpp>

#include "bench_types.hpp"

template<typename Problem>
osqp::OsqpInstance smoothProblemToOsqpInstance(const Problem & pbm)
{
  if constexpr (std::is_base_of_v<Eigen::SparseMatrixBase<decltype(pbm.P)>, decltype(pbm.P)>) {

    return {.objective_matrix = pbm.P.template cast<double>(),
      .objective_vector       = pbm.q.template cast<double>(),
      .constraint_matrix      = pbm.A.template cast<double>(),
      .lower_bounds           = pbm.l.template cast<double>(),
      .upper_bounds           = pbm.u.template cast<double>()};
  } else {
    return {.objective_matrix = pbm.P.template cast<double>().sparseView(),
      .objective_vector       = pbm.q.template cast<double>().sparseView(),
      .constraint_matrix      = pbm.A.template cast<double>().sparseView(),
      .lower_bounds           = pbm.l.template cast<double>().sparseView(),
      .upper_bounds           = pbm.u.template cast<double>().sparseView()};
  }
}

inline osqp::OsqpSettings smoothSettingsToOsqpSettings(const smooth::feedback::SolverParams & prm)
{
  osqp::OsqpSettings ret;
  ret.set_default();
  ret.alpha              = prm.alpha;
  ret.rho                = prm.rho;
  ret.sigma              = prm.sigma;
  ret.eps_abs            = prm.eps_abs;
  ret.eps_rel            = prm.eps_rel;
  ret.eps_prim_inf       = prm.eps_primal_inf;
  ret.eps_dual_inf       = prm.eps_dual_inf;
  ret.max_iter           = prm.max_iter;
  ret.check_termination  = prm.stop_check_iter;
  ret.polish             = prm.polish;
  ret.polish_refine_iter = prm.polish_iter;
  ret.delta              = prm.delta;
  ret.adaptive_rho       = false;
  ret.scaled_termination = false;
  return ret;
}

template<typename Scalar, typename Problem>
struct OsqpWrapper
{
  OsqpWrapper(const Problem & pbm, const smooth::feedback::SolverParams & prm)
      : pbm_(pbm), prm_(prm)
  {}
  BenchResult operator()()
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    solver_.Init(smoothProblemToOsqpInstance(pbm_), smoothSettingsToOsqpSettings(prm_));
    osqp::OsqpExitCode code = solver_.Solve();
    auto t1                 = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<double, -1, 1> qp_primal(solver_.primal_solution());
    return {.dt = t1 - t0,
      .solution = qp_primal,
      .success =
        (code == osqp::OsqpExitCode::kOptimal) || (code == osqp::OsqpExitCode::kMaxIterations),
      .objective =
        (qp_primal.transpose() * pbm_.P * qp_primal + pbm_.q.transpose() * qp_primal).eval()[0]};
  }
  osqp::OsqpSolver solver_;
  Problem pbm_;
  smooth::feedback::SolverParams prm_;
};

#endif  // BENCHMARK__OSQP_BENCH_HPP_