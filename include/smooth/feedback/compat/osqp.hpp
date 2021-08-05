#ifndef SMOOTH__FEEDBACK__COMPAT__OSQP_HPP_
#define SMOOTH__FEEDBACK__COMPAT__OSQP_HPP_

#include <Eigen/Sparse>
#include <osqp/osqp.h>

#include "smooth/feedback/qp.hpp"

namespace smooth::feedback {

/**
 * @brief Solve a QuadraticProgram with the OSQP solver
 *
 * @param pbm QuadraticProgram to solve
 * @param prm solver paramters
 *
 * @return solution
 *
 * @note This is a conveience interface that performs copies and memory allocation in each
 * call. For more fine-grained control use the low-level OSQP interface (https://osqp.org/).
 */
template<typename Problem>
Solution<-1, -1, double> solve_qp_osqp(const Problem & pbm,
  const SolverParams & prm,
  std::optional<std::reference_wrapper<const Solution<-1, -1, double>>> warmstart = {})
{
  // Covert to sparse matrices with OSQP indexing
  Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> P;
  Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> A;
  if constexpr (std::is_base_of_v<Eigen::SparseMatrixBase<decltype(pbm.A)>, decltype(pbm.A)>) {
    A = pbm.A;
    P = pbm.P.template triangularView<Eigen::Upper>();
  } else {
    A                   = pbm.A.sparseView();
    Eigen::MatrixXd Pup = pbm.P.template triangularView<Eigen::Upper>();
    P                   = Pup.sparseView();
    A.prune(1e-6);
    P.prune(1e-6);
  }
  A.makeCompressed();
  P.makeCompressed();

  OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  osqp_set_default_settings(settings);

  settings->verbose            = prm.verbose;
  settings->sigma              = prm.sigma;
  settings->alpha              = prm.alpha;
  settings->rho                = prm.rho;
  settings->eps_abs            = prm.eps_abs;
  settings->eps_rel            = prm.eps_rel;
  settings->eps_prim_inf       = prm.eps_primal_inf;
  settings->eps_dual_inf       = prm.eps_dual_inf;
  settings->scaling            = prm.scaling;
  settings->check_termination  = prm.stop_check_iter;
  settings->polish             = prm.polish;
  settings->polish_refine_iter = prm.polish_iter;
  settings->delta              = prm.delta;

  settings->adaptive_rho       = false;
  settings->linsys_solver      = QDLDL_SOLVER;
  settings->scaled_termination = false;

  if (prm.max_iter) {
    settings->max_iter = prm.max_iter.value();
  } else {
    settings->max_iter = std::numeric_limits<c_int>::max();
  }
  if (prm.max_time) {
    settings->time_limit =
      duration_cast<std::chrono::duration<double>>(prm.max_time.value()).count();
  } else {
    settings->time_limit = 0;
  }

  OSQPData * data = (OSQPData *)c_malloc(sizeof(OSQPData));
  data->n         = A.cols();
  data->m         = A.rows();
  data->A         = csc_matrix(
    A.rows(), A.cols(), A.nonZeros(), A.valuePtr(), A.innerIndexPtr(), A.outerIndexPtr());
  data->q = const_cast<double *>(pbm.q.data());
  data->P = csc_matrix(
    P.rows(), P.cols(), P.nonZeros(), P.valuePtr(), P.innerIndexPtr(), P.outerIndexPtr());
  data->l = const_cast<double *>(pbm.l.data());
  data->u = const_cast<double *>(pbm.u.data());

  OSQPWorkspace * work;

  Solution<-1, -1, double> ret;
  ret.code = ExitCode::Unknown;

  c_int error = osqp_setup(&work, data, settings);

  if (warmstart) {
    osqp_warm_start(
      work, warmstart.value().get().primal.data(), warmstart.value().get().dual.data());
    settings->warm_start = 1;
  } else {
    settings->warm_start = 0;
  }

  if (!error) { error &= osqp_solve(work); }

  if (!error) {
    switch (work->info->status_val) {
    case OSQP_SOLVED: {
      ret.code = ExitCode::Optimal;
      break;
    }
    case OSQP_PRIMAL_INFEASIBLE: {
      ret.code = ExitCode::PrimalInfeasible;
      break;
    }
    case OSQP_DUAL_INFEASIBLE: {
      ret.code = ExitCode::DualInfeasible;
      break;
    }
    case OSQP_MAX_ITER_REACHED: {
      ret.code = ExitCode::MaxIterations;
      break;
    }
    case OSQP_TIME_LIMIT_REACHED: {
      ret.code = ExitCode::MaxTime;
      break;
    }
    default: {
      break;
    }
    }

    ret.iter   = work->info->iter;
    ret.primal = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(work->solution->x, data->n);
    ret.dual   = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(work->solution->y, data->m);
  }

  osqp_cleanup(work);

  c_free(data->A);
  c_free(data->P);
  c_free(data);
  c_free(settings);

  return ret;
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COMPAT__OSQP_HPP_
