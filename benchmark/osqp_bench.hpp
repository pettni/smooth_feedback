#ifndef BENCHMARK__OSQP_BENCH_HPP_
#define BENCHMARK__OSQP_BENCH_HPP_

#include "bench_types.hpp"
#include <osqp/osqp.h>

template<typename Scalar, typename Problem>
struct OsqpWrapper
{
  OsqpWrapper(const Problem & pbm, const smooth::feedback::SolverParams & prm)
  {
    if constexpr (std::is_base_of_v<Eigen::SparseMatrixBase<decltype(pbm.A)>, decltype(pbm.A)>) {
      A_ = pbm.A;
      P_ = pbm.P.template triangularView<Eigen::Upper>();
    } else {
      A_                  = pbm.A.sparseView();
      Eigen::MatrixXd Pup = pbm.P.template triangularView<Eigen::Upper>();
      P_                  = Pup.sparseView();
    }
    l_ = pbm.l;
    u_ = pbm.u;
    q_ = pbm.q;

    A_.makeCompressed();
    P_.makeCompressed();

    settings_ = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    data_     = (OSQPData *)c_malloc(sizeof(OSQPData));

    osqp_set_default_settings(settings_);
    settings_->sigma              = prm.sigma;
    settings_->alpha              = prm.alpha;
    settings_->rho                = prm.rho;
    settings_->sigma              = prm.sigma;
    settings_->eps_abs            = prm.eps_abs;
    settings_->eps_rel            = prm.eps_rel;
    settings_->eps_prim_inf       = prm.eps_primal_inf;
    settings_->eps_dual_inf       = prm.eps_dual_inf;
    settings_->max_iter           = prm.max_iter;
    settings_->check_termination  = prm.stop_check_iter;
    settings_->polish             = prm.polish;
    settings_->polish_refine_iter = prm.polish_iter;
    settings_->delta              = prm.delta;
    settings_->adaptive_rho       = false;
    settings_->scaled_termination = false;
    settings_->linsys_solver      = QDLDL_SOLVER;

    data_->n = A_.cols();
    data_->m = A_.rows();
    data_->A = csc_matrix(
      A_.rows(), A_.cols(), A_.nonZeros(), A_.valuePtr(), A_.innerIndexPtr(), A_.outerIndexPtr());
    data_->q = q_.data();
    data_->P = csc_matrix(
      P_.rows(), P_.cols(), P_.nonZeros(), P_.valuePtr(), P_.innerIndexPtr(), P_.outerIndexPtr());
    data_->l = l_.data();
    data_->u = u_.data();
  }

  ~OsqpWrapper()
  {
    if (data_) {
      if (data_->A) c_free(data_->A);
      if (data_->P) c_free(data_->P);
      c_free(data_);
    }
    if (settings_) c_free(settings_);
  }

  BenchResult operator()()
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    osqp_setup(&work_, data_, settings_);
    auto exit = osqp_solve(work_);
    auto t1   = std::chrono::high_resolution_clock::now();

    Eigen::Matrix<double, -1, 1> qp_primal =
      Eigen::Map<const Eigen::Matrix<double, -1, 1>>(work_->solution->x, data_->n);

    osqp_cleanup(work_);

    return {.dt  = t1 - t0,
      .solution  = qp_primal,
      .success   = (exit == 0) || (exit == OSQP_MAX_ITER_REACHED),
      .objective = (qp_primal.transpose() * P_).dot(qp_primal) + q_.dot(qp_primal)};
  }

  Eigen::SparseMatrix<Scalar, Eigen::ColMajor, long long> P_;
  Eigen::SparseMatrix<Scalar, Eigen::ColMajor, long long> A_;
  Eigen::VectorXd q_, l_, u_;

  OSQPWorkspace * work_;
  OSQPSettings * settings_;
  OSQPData * data_;
};

#endif  // BENCHMARK__OSQP_BENCH_HPP_
