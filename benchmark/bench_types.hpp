#ifndef BENCHMARK__BENCH_TYPES_HPP_
#define BENCHMARK__BENCH_TYPES_HPP_

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <smooth/feedback/qp.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

template<Eigen::Index Start, Eigen::Index Finish, int Spacing = 1>
struct static_for
{
  template<typename Func>
  void operator()(Func const & func) const
  {
    if constexpr (Start < Finish) {
      func(std::integral_constant<int, Start>{});
      static_for<Start + Spacing, Finish>()(func);
    }
  }
};

// Note: assume default seed across all gens
template<typename Scalar>
Eigen::SparseMatrix<Scalar, Eigen::ColMajor> randomSparseMatrix(std::default_random_engine & gen,
  const Eigen::Index rows,
  const Eigen::Index cols,
  const double density = 1.,
  const Scalar eps     = 1e-3,
  const Scalar max     = 1.)
{
  std::uniform_real_distribution<Scalar> probRand(0., 1.);
  std::normal_distribution<Scalar> valRand(0., max / 2);

  std::vector<Eigen::Triplet<double>> triplets;
  for (auto i = 0u; i < rows; ++i) {
    for (auto j = 0u; j < cols; ++j) {
      auto rval = probRand(gen);
      if (rval > (1 - density)) { triplets.emplace_back(i, j, valRand(gen)); }
    }
  }
  Eigen::SparseMatrix<Scalar, Eigen::ColMajor> A(rows, cols);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

struct ReportProblem
{
  bool sparse;
  double density;
  smooth::feedback::QuadraticProgram<-1, -1, double> qp;
};

template<Eigen::Index M, Eigen::Index N, typename Scalar>
struct SparseBase
{
public:
  smooth::feedback::QuadraticProgramSparse<Scalar> qp_;
  Eigen::Matrix<Scalar, -1, 1> v_;  // solution
  ReportProblem toReportProblem()
  {
    smooth::feedback::QuadraticProgram<-1, -1, double> rqp;
    rqp.P = qp_.P;
    rqp.A = qp_.A;
    rqp.q = qp_.q;
    rqp.l = qp_.l;
    rqp.u = qp_.u;
    return {.sparse = true,
      .density      = static_cast<double>(qp_.P.nonZeros()) / static_cast<double>(qp_.P.size()),
      .qp           = rqp};
  }
};

template<Eigen::Index M, Eigen::Index N, typename Scalar>
struct RandomSparse : public SparseBase<M, N, Scalar>
{
  RandomSparse(std::default_random_engine & gen, const double density)
  {

    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> A = randomSparseMatrix<Scalar>(gen, M, N, density);

    Eigen::SparseMatrix<Scalar, Eigen::ColMajor> P = randomSparseMatrix<Scalar>(gen, N, N, density);
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor> Inn(N, N);
    Inn.setIdentity();
    P                                  = P * P.transpose() + 1e-02 * Inn;
    Eigen::Matrix<Scalar, -1, 1> q     = Eigen::Matrix<Scalar, -1, 1>::Random(N);
    Eigen::Matrix<Scalar, -1, 1> v     = Eigen::Matrix<Scalar, -1, 1>::Random(N);
    Eigen::Matrix<Scalar, -1, 1> delta = Eigen::Matrix<Scalar, -1, 1>::Random(M);
    Eigen::Matrix<Scalar, -1, 1> u     = A * v + delta;
    Eigen::Matrix<Scalar, -1, 1> l =
      Eigen::Matrix<Scalar, -1, 1>::Constant(M, -std::numeric_limits<Scalar>::infinity());

    this->qp_ = smooth::feedback::QuadraticProgramSparse{.P = P, .q = q, .A = A, .l = l, .u = u};
    this->v_  = v;
  }
  RandomSparse(const ReportProblem & copy)
  {
    this->qp_.A = copy.qp.A;
    this->qp_.P = copy.qp.P;
    this->qp_.q = copy.qp.q;
    this->qp_.u = copy.qp.u;
    this->qp_.l = copy.qp.l;
  }
};

template<Eigen::Index M, Eigen::Index N, typename Scalar>
struct DenseBase
{
public:
  smooth::feedback::QuadraticProgram<-1, -1, Scalar> qp_;
  Eigen::Matrix<Scalar, -1, 1> v_;
  ReportProblem toReportProblem()
  {
    smooth::feedback::QuadraticProgram<-1, -1, double> rqp;
    rqp.P = qp_.P;
    rqp.A = qp_.A;
    rqp.q = qp_.q;
    rqp.l = qp_.l;
    rqp.u = qp_.u;
    return {.sparse = false, .density = 1., .qp = rqp};
  }
};

template<Eigen::Index M, Eigen::Index N, typename Scalar>
struct RandomDense : public DenseBase<M, N, Scalar>
{
  RandomDense(std::default_random_engine & gen)
  {

    using Rmat = Eigen::Matrix<Scalar, -1, -1>;
    using Rvec = Eigen::Matrix<Scalar, -1, 1>;
    Rmat A     = Rmat::Random(M, N);
    Rmat P     = Rmat::Random(N, N);
    P          = P * P.transpose() + 1e-02 * Rmat::Identity(N, N);
    Rvec q     = Rvec::Random(N);
    Rvec v     = Rvec::Random(N);
    Rvec delta = Rvec::Random(M);
    Rvec u     = A * v + delta;
    Rvec l     = Rvec::Constant(M, -std::numeric_limits<Scalar>::infinity());
    this->qp_ =
      smooth::feedback::QuadraticProgram<-1, -1, Scalar>{.P = P, .q = q, .A = A, .l = l, .u = u};
    this->v_ = v;
  }
  RandomDense(const ReportProblem & copy)
  {
    this->qp_.A = copy.qp.A;
    this->qp_.P = copy.qp.P;
    this->qp_.q = copy.qp.q;
    this->qp_.u = copy.qp.u;
    this->qp_.l = copy.qp.l;
  }
};

template<Eigen::Index M, Eigen::Index N, typename Scalar>
struct StaticBase
{
public:
  smooth::feedback::QuadraticProgram<M, N, Scalar> qp_;
  Eigen::Matrix<Scalar, N, 1> v_;

  ReportProblem toReportProblem()
  {
    smooth::feedback::QuadraticProgram<-1, -1, double> rqp;
    rqp.P = qp_.P;
    rqp.A = qp_.A;
    rqp.q = qp_.q;
    rqp.l = qp_.l;
    rqp.u = qp_.u;
    return {.sparse = false, .density = 1., .qp = rqp};
  }
};

template<Eigen::Index M, Eigen::Index N, typename Scalar>
struct RandomStatic : public StaticBase<M, N, Scalar>
{
  RandomStatic(std::default_random_engine & gen)
  {
    using Rn  = Eigen::Matrix<Scalar, N, 1>;
    using Rm  = Eigen::Matrix<Scalar, M, 1>;
    using Rnn = Eigen::Matrix<Scalar, N, N>;
    using Rmn = Eigen::Matrix<Scalar, M, N>;

    this->qp_.A = Rmn::Random();
    Rnn P       = Rnn::Random();
    this->qp_.P = P * P.transpose() + 1e-02 * Rnn::Identity();
    this->qp_.q = Rn::Random();
    this->v_    = Rn::Random();
    Rm delta    = Rm::Random();
    this->qp_.u = this->qp_.A * this->v_ + delta;
    this->qp_.l = Rm::Constant(-std::numeric_limits<Scalar>::infinity());
  }

  RandomStatic(const ReportProblem & copy)
  {
    this->qp_.A = copy.qp.A;
    this->qp_.P = copy.qp.P;
    this->qp_.q = copy.qp.q;
    this->qp_.u = copy.qp.u;
    this->qp_.l = copy.qp.l;
  }
};

struct BenchResult
{
  std::chrono::high_resolution_clock::duration dt;
  uint64_t iter;
  Eigen::Matrix<double, -1, -1> solution;
  bool success;
  double objective;
};

using TrialVec = std::pair<std::vector<ReportProblem>, std::vector<std::optional<BenchResult>>>;

struct BatchResult
{
  TrialVec batch;
  double avg_accuracy;
  double avg_duration;

  friend std::ostream & operator<<(std::ostream & os, const BatchResult &);
};

inline std::ostream & operator<<(std::ostream & os, const BatchResult & res)
{
  os << " Accuracy: " << res.avg_accuracy << std::endl;
  os << " Duration: " << res.avg_duration << std::endl;
  return os;
}

template<typename Scalar,
  Eigen::Index M,
  Eigen::Index N,
  template<Eigen::Index, Eigen::Index, typename>
  typename Problem,
  template<typename, typename>
  typename Solver>
BatchResult RandomBatch(std::default_random_engine & gen,
  const smooth::feedback::QPSolverParams & prm,
  const std::size_t n,
  const double density = 1.)
{
  std::uniform_real_distribution<double> densityDist;
  TrialVec results;
  double avg_duration = 0.;
  double avg_accuracy = 0.;
  for (auto i = 0u; i < n; ++i) {
    auto pbm = Problem<M, N, Scalar>(gen, density);
    Solver<Scalar, decltype(pbm.qp_)> solver(pbm.qp_, prm);
    auto res = solver();
    results.first.push_back(pbm.toReportProblem());
    if (res.success) {
      avg_duration += std::chrono::duration<double>(res.dt).count() / (i + 1.);
      avg_accuracy += (res.solution - pbm.v_).norm();
      results.second.push_back(res);
    } else {
      results.second.push_back(std::nullopt);
    }
  }
  return {results, avg_accuracy, avg_duration};
}

template<typename Scalar,
  Eigen::Index M,
  Eigen::Index N,
  template<Eigen::Index, Eigen::Index, typename>
  typename Problem,
  template<typename, typename>
  typename Solver>
BatchResult RandomBatch(
  const BatchResult & src, const smooth::feedback::QPSolverParams & prm, const std::size_t n)
{
  TrialVec results;
  double avg_duration = 0.;
  double avg_accuracy = 0.;
  for (auto i = 0u; i < n; ++i) {
    auto pbm = Problem<M, N, Scalar>(src.batch.first[i]);
    Solver<Scalar, decltype(pbm.qp_)> solver(pbm.qp_, prm);
    auto res = solver();
    results.first.push_back(pbm.toReportProblem());
    if (res.success) {
      avg_duration += std::chrono::duration<double>(res.dt).count() / (i + 1.);
      avg_accuracy += (res.solution - pbm.v_).norm();
      results.second.push_back(res);
    } else {
      results.second.push_back(std::nullopt);
    }
  }
  return {results, avg_accuracy, avg_duration};
}

template<template<typename, typename> typename Solver, Eigen::Index M, Eigen::Index N>
std::pair<std::vector<std::string>, std::unordered_map<std::string, BatchResult>> BenchSuite(
  std::default_random_engine & gen, const int n, const double d)
{
  smooth::feedback::QPSolverParams prm{};
  prm.eps_abs  = 1e-6;
  prm.eps_rel  = 1e-6;
  prm.polish   = true;
  prm.max_iter = 100000;

  auto sparse_res = RandomBatch<double, M, N, RandomSparse, Solver>(gen, prm, n, d);
  auto static_res = RandomBatch<double, M, N, RandomStatic, Solver>(sparse_res, prm, n);
  auto dense_res  = RandomBatch<double, M, N, RandomDense, Solver>(sparse_res, prm, n);
  // auto dense_res  = RandomBatch<double, M, N, RandomDense, Solver>(gen, prm, n);
  // auto static_res = RandomBatch<double, M, N, RandomStatic, Solver>(gen, prm, n);
  std::unordered_map<std::string, BatchResult> ret;
  ret.emplace(std::make_pair("Sparse", sparse_res));
  ret.emplace(std::make_pair("Dense", dense_res));
  ret.emplace(std::make_pair("Static", static_res));
  return {{"Sparse", "Dense", "Static"}, ret};
}

#endif  // BENCHMARK__BENCH_TYPES_HPP_
