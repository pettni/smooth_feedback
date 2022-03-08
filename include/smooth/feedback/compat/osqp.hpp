// smooth_feedback: Control theory on Lie groups
// https://github.com/pettni/smooth_feedback
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson, John B. Mains
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

#ifndef SMOOTH__FEEDBACK__COMPAT__OSQP_HPP_
#define SMOOTH__FEEDBACK__COMPAT__OSQP_HPP_

/**
 * @file
 * @brief Solve quadratiac programs with OSQP.
 */

#include <Eigen/Sparse>
#include <osqp/osqp.h>

#include "smooth/feedback/qp_solver.hpp"

namespace smooth::feedback {

/**
 * @brief Solve a QuadraticProgram with the OSQP solver
 *
 * @param pbm QuadraticProgram to solve
 * @param prm solver paramters
 * @param warmstart initial point to start iterating from
 *
 * @return solution
 *
 * @note This is a convenience interface that performs copies and memory allocation in each
 * call. For more fine-grained control use the low-level OSQP interface (https://osqp.org/).
 */
template<typename Problem>
QPSolution<-1, -1, double> solve_qp_osqp(
  const Problem & pbm,
  const QPSolverParams & prm,
  std::optional<std::reference_wrapper<const QPSolution<-1, -1, double>>> warmstart = {})
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

  QPSolution<-1, -1, double> ret;
  ret.code = QPSolutionStatus::Unknown;

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
      ret.code = QPSolutionStatus::Optimal;
      break;
    }
    case OSQP_PRIMAL_INFEASIBLE: {
      ret.code = QPSolutionStatus::PrimalInfeasible;
      break;
    }
    case OSQP_DUAL_INFEASIBLE: {
      ret.code = QPSolutionStatus::DualInfeasible;
      break;
    }
    case OSQP_MAX_ITER_REACHED: {
      ret.code = QPSolutionStatus::MaxIterations;
      break;
    }
    case OSQP_TIME_LIMIT_REACHED: {
      ret.code = QPSolutionStatus::MaxTime;
      break;
    }
    default: {
      break;
    }
    }

    ret.iter      = work->info->iter;
    ret.primal    = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(work->solution->x, data->n);
    ret.dual      = Eigen::Map<const Eigen::Matrix<double, -1, 1>>(work->solution->y, data->m);
    ret.objective = work->info->obj_val;
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
