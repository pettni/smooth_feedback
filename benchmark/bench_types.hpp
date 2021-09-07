#ifndef BENCHMARK__BENCH_TYPES_HPP_
#define BENCHMARK__BENCH_TYPES_HPP_

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <smooth/feedback/qp.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

/**
 * @brief Create a random quadratic program of dimension m x n.
 * @param m number on inequalities
 * @param n number of variables
 * @param density approximate proportion of nonzeros in P and A
 * @param RNG random number generator
 */
template<typename RNG>
smooth::feedback::QuadraticProgram<-1, -1> random_qp(int m, int n, double density, RNG & rng)
{
  std::bernoulli_distribution bdist(density);
  std::uniform_real_distribution<double> udist(-1, 1);

  Eigen::MatrixXd A =
    Eigen::MatrixXd::NullaryExpr(m, n, [&](int, int) { return bdist(rng) ? udist(rng) : 0.; });
  Eigen::MatrixXd P =
    Eigen::MatrixXd::NullaryExpr(n, n, [&](int, int) { return bdist(rng) ? udist(rng) : 0.; });

  Eigen::VectorXd v     = Eigen::VectorXd::NullaryExpr(n, [&](int) { return udist(rng); });
  Eigen::VectorXd delta = Eigen::VectorXd::NullaryExpr(m, [&](int) { return udist(rng); });

  return smooth::feedback::QuadraticProgram<-1, -1>{
    .P = P * P.transpose() + 0.01 * Eigen::MatrixXd::Identity(n, n),
    .q = Eigen::VectorXd::NullaryExpr(n, [&](int) { return udist(rng); }),
    .A = A,
    .l = Eigen::VectorXd::Constant(m, -std::numeric_limits<double>::infinity()),
    .u = A * v + delta,
  };
}

/**
 * @brief Create a random sparse quadratic program of dimension m x n.
 */
inline smooth::feedback::QuadraticProgramSparse<double> qp_dense_to_sparse(
  const smooth::feedback::QuadraticProgram<-1, -1, double> & qp)
{
  smooth::feedback::QuadraticProgramSparse<double> qps;

  qps.P = qp.P.sparseView();
  qps.P.prune(1e-6);
  qps.P.makeCompressed();
  qps.q = qp.q;
  qps.A = qp.A.sparseView();
  qps.A.prune(1e-6);
  qps.A.makeCompressed();
  qps.l = qp.l;
  qps.u = qp.u;

  return qps;
}

struct BenchResult
{
  smooth::feedback::QPSolutionStatus status;
  std::chrono::high_resolution_clock::duration dt;
  uint64_t iter;
  Eigen::VectorXd solution;
  double objective;
};

struct BatchResult
{
  std::vector<BenchResult> results{};

  std::size_t num_optimal{0};

  double total_duration_success{0};
  double total_duration_fail{0};

  double avg_duration_success{0};
  double avg_duration_fail{0};
};

template<typename SolverWrapper, std::ranges::range R>
BatchResult solve_batch(const R & qps, const smooth::feedback::QPSolverParams & prm)
{
  auto work_range =
    std::views::transform(qps, [&prm](const auto & qp) { return SolverWrapper(qp, prm); });

  BatchResult ret{};

  for (const auto & work : work_range) { ret.results.push_back(work()); }

  // calculate accuracies
  for (const auto & result : ret.results) {
    if (result.status == smooth::feedback::QPSolutionStatus::Optimal) {
      ++ret.num_optimal;
      ret.total_duration_success += std::chrono::duration<double>(result.dt).count();
    } else {
      ret.total_duration_fail += std::chrono::duration<double>(result.dt).count();
    }
  }

  std::size_t num_fail = ret.results.size() - ret.num_optimal;

  ret.avg_duration_success =
    ret.num_optimal > 0 ? ret.total_duration_success / ret.num_optimal : -1;
  ret.avg_duration_fail = num_fail > 0 ? ret.total_duration_fail / num_fail : -1;

  return ret;
}

#endif  // BENCHMARK__BENCH_TYPES_HPP_
