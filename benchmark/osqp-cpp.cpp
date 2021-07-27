// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cbr_control/osqp-cpp.hpp"

#include <algorithm>
#include <cassert>
#include <string>
#include <type_traits>


// Fails to compile if OSQP's typedefs change. This lets us avoid including
// osqp.h in osqp++.h.
static_assert(
  std::is_same_v<osqp::c_int, ::c_int>,
  "OSQP's c_int typedef does not match the definition in osqp++.h.");
static_assert(
  std::is_same_v<c_float, double>,
  "OSQP's c_float typedef is unexpectedly not the same as double");

static_assert(
  sizeof(OSQPSettings) == 176,
  "The size of OSQPSettings has changed unexpectedly. Make sure "
  "that the map between ::OSQPSettings and osqp::OsqpSettings "
  "remains up to date.");

namespace osqp
{

namespace
{

using ::Eigen::Map;
using ::Eigen::Ref;
using ::Eigen::VectorXd;


::OSQPSettings ToInternalSettings(const OsqpSettings & settings)
{
  OSQPSettings osqp_settings;
  osqp_set_default_settings(&osqp_settings);
  if (settings.rho) {osqp_settings.rho = settings.rho.value();}
  if (settings.sigma) {osqp_settings.sigma = settings.sigma.value();}
  if (settings.scaling) {osqp_settings.scaling = settings.scaling.value();}
  if (settings.adaptive_rho) {osqp_settings.adaptive_rho = settings.adaptive_rho.value();}
  if (settings.adaptive_rho_interval) {
    osqp_settings.adaptive_rho_interval = settings.adaptive_rho_interval.value();
  }
  if (settings.adaptive_rho_tolerance) {
    osqp_settings.adaptive_rho_tolerance = settings.adaptive_rho_tolerance.value();
  }
  if (settings.adaptive_rho_tolerance) {
    osqp_settings.adaptive_rho_fraction = settings.adaptive_rho_fraction.value();
  }
  if (settings.max_iter) {osqp_settings.max_iter = settings.max_iter.value();}
  if (settings.eps_abs) {osqp_settings.eps_abs = settings.eps_abs.value();}
  if (settings.eps_rel) {osqp_settings.eps_rel = settings.eps_rel.value();}
  if (settings.eps_prim_inf) {osqp_settings.eps_prim_inf = settings.eps_prim_inf.value();}
  if (settings.eps_dual_inf) {osqp_settings.eps_dual_inf = settings.eps_dual_inf.value();}
  if (settings.alpha) {osqp_settings.alpha = settings.alpha.value();}
  if (settings.delta) {osqp_settings.delta = settings.delta.value();}
  if (settings.polish) {osqp_settings.polish = settings.polish.value();}
  if (settings.polish_refine_iter) {
    osqp_settings.polish_refine_iter = settings.polish_refine_iter.value();
  }
  if (settings.verbose) {osqp_settings.verbose = settings.verbose.value();}
  if (settings.scaled_termination) {
    osqp_settings.scaled_termination = settings.scaled_termination.value();
  }
  if (settings.check_termination) {
    osqp_settings.check_termination = settings.check_termination.value();
  }
  if (settings.warm_start) {osqp_settings.warm_start = settings.warm_start.value();}
  if (settings.time_limit) {osqp_settings.time_limit = settings.time_limit.value();}
  osqp_settings.linsys_solver = ::QDLDL_SOLVER;
  return osqp_settings;
}

}  // namespace

void OsqpSettings::set_default()
{
  ::OSQPSettings osqp_settings;
  osqp_set_default_settings(&osqp_settings);
  rho = osqp_settings.rho;
  sigma = osqp_settings.sigma;
  scaling = osqp_settings.scaling;
  adaptive_rho = osqp_settings.adaptive_rho;
  adaptive_rho_interval = osqp_settings.adaptive_rho_interval;
  adaptive_rho_tolerance = osqp_settings.adaptive_rho_tolerance;
  adaptive_rho_fraction = osqp_settings.adaptive_rho_fraction;
  max_iter = osqp_settings.max_iter;
  eps_abs = osqp_settings.eps_abs;
  eps_rel = osqp_settings.eps_rel;
  eps_prim_inf = osqp_settings.eps_prim_inf;
  eps_dual_inf = osqp_settings.eps_dual_inf;
  alpha = osqp_settings.alpha;
  delta = osqp_settings.delta;
  polish = osqp_settings.polish;
  polish_refine_iter = osqp_settings.polish_refine_iter;
  verbose = osqp_settings.verbose;
  scaled_termination = osqp_settings.scaled_termination;
  check_termination = osqp_settings.check_termination;
  warm_start = osqp_settings.warm_start;
  time_limit = osqp_settings.time_limit;
}

struct OSQPWorkspaceHelper : public ::OSQPWorkspace {};

void OsqpSolver::OsqpDeleter::operator()(OSQPWorkspaceHelper * workspace) const
{
  osqp_cleanup(workspace);
}

// OSQP_HANDLE_EXITCODE(x) expands to 'case x: return "x"'. Using this macro
// prevents typos in the strings.
#define OSQP_HANDLE_EXITCODE(x) \
  case x: \
    return #x

std::string ToString(OsqpExitCode exitcode)
{
  switch (exitcode) {
    OSQP_HANDLE_EXITCODE(OsqpExitCode::kOptimal);
    OSQP_HANDLE_EXITCODE(OsqpExitCode::kPrimalInfeasible);
    OSQP_HANDLE_EXITCODE(OsqpExitCode::kDualInfeasible);
    OSQP_HANDLE_EXITCODE(OsqpExitCode::kOptimalInaccurate);
    OSQP_HANDLE_EXITCODE(OsqpExitCode::kPrimalInfeasibleInaccurate);
    OSQP_HANDLE_EXITCODE(OsqpExitCode::kDualInfeasibleInaccurate);
    OSQP_HANDLE_EXITCODE(OsqpExitCode::kMaxIterations);
    OSQP_HANDLE_EXITCODE(OsqpExitCode::kInterrupted);
    OSQP_HANDLE_EXITCODE(OsqpExitCode::kTimeLimitReached);
    OSQP_HANDLE_EXITCODE(OsqpExitCode::kNonConvex);
    OSQP_HANDLE_EXITCODE(OsqpExitCode::kUnknown);
  }
  return "Unknown exit code";
}

#undef OSQP_HANDLE_EXITCODE

namespace
{

Status CheckDimensions(
  const int left_value, const int right_value,
  const std::string & left_name,
  const std::string & right_name)
{
  if (left_value != right_value) {
    return Status::InvalidArgumentError(
      "Dimension mismatch: " + left_name + " (= " + std::to_string(left_value) +
      ") must equal " + right_name + " (= " + std::to_string(right_value) + ").");
  } else {
    return Status::OkStatus();
  }
}

}  // namespace

#define OSQP_CHECK_DIMENSIONS(left_value, right_value) \
  CheckDimensions((left_value), (right_value), #left_value, #right_value)

#define OSQP_RETURN_IF_ERROR(expr) \
  { \
    const Status result = expr; \
    if (!result.ok()) {return result;} \
  }

#define OSQP_CHECK(expr) assert(expr)

Status OsqpSolver::Init(
  const OsqpInstance & instance,
  const OsqpSettings & settings,
  const bool objective_upper_triangular)
{
  const c_int num_variables = instance.num_variables();
  const c_int num_constraints = instance.num_constraints();

  OSQP_RETURN_IF_ERROR(
    OSQP_CHECK_DIMENSIONS(instance.objective_matrix.cols(), num_variables));
  OSQP_RETURN_IF_ERROR(
    OSQP_CHECK_DIMENSIONS(instance.objective_matrix.rows(), num_variables));
  OSQP_RETURN_IF_ERROR(
    OSQP_CHECK_DIMENSIONS(instance.objective_vector.size(), num_variables));
  OSQP_RETURN_IF_ERROR(
    OSQP_CHECK_DIMENSIONS(instance.lower_bounds.size(), num_constraints));
  OSQP_RETURN_IF_ERROR(
    OSQP_CHECK_DIMENSIONS(instance.upper_bounds.size(), num_constraints));

  // Clip bounds using OSQP_INFTY. Failing to do this causes subtle convergence
  // issues instead of producing explicit errors (e.g., the
  // DetectsPrimalInfeasible test fails). The osqp-python interface also clips
  // the bounds in the same way.
  VectorXd clipped_lower_bounds = instance.lower_bounds.cwiseMax(-OSQP_INFTY);
  VectorXd clipped_upper_bounds = instance.upper_bounds.cwiseMin(OSQP_INFTY);

  // OSQP copies all the data, so it's okay to discard this struct after
  // osqp_setup. It also does not modify the input data (note osqp_setup takes a
  // const OSQPData*). const_cast is needed here to fill in the input data
  // structures.
  OSQPData data;
  data.n = num_variables;
  data.m = num_constraints;


  ::csc objective_matrix;
  Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> objective_matrix_upper_triangle;
  if (objective_upper_triangular) {
    if (!instance.objective_matrix.isCompressed()) {
      objective_matrix_upper_triangle = instance.objective_matrix;
      objective_matrix_upper_triangle.makeCompressed();
      objective_matrix = {
        objective_matrix_upper_triangle.outerIndexPtr()[num_variables],
        num_variables,
        num_variables,
        const_cast<c_int *>(objective_matrix_upper_triangle.outerIndexPtr()),
        const_cast<c_int *>(objective_matrix_upper_triangle.innerIndexPtr()),
        const_cast<double *>(objective_matrix_upper_triangle.valuePtr()),
        -1};
    } else {
      objective_matrix = {
        instance.objective_matrix.outerIndexPtr()[num_variables],
        num_variables,
        num_variables,
        const_cast<c_int *>(instance.objective_matrix.outerIndexPtr()),
        const_cast<c_int *>(instance.objective_matrix.innerIndexPtr()),
        const_cast<double *>(instance.objective_matrix.valuePtr()),
        -1};
    }
  } else {
    objective_matrix_upper_triangle =
      instance.objective_matrix.triangularView<Eigen::Upper>();
    if (!objective_matrix_upper_triangle.isCompressed()) {
      objective_matrix_upper_triangle.makeCompressed();
    }
    objective_matrix = {
      objective_matrix_upper_triangle.outerIndexPtr()[num_variables],
      num_variables,
      num_variables,
      const_cast<c_int *>(objective_matrix_upper_triangle.outerIndexPtr()),
      const_cast<c_int *>(objective_matrix_upper_triangle.innerIndexPtr()),
      const_cast<double *>(objective_matrix_upper_triangle.valuePtr()),
      -1};
  }

  data.P = &objective_matrix;

  ::csc constraint_matrix;
  if (!instance.constraint_matrix.isCompressed()) {
    Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> constraint_matrix_comp =
      instance.constraint_matrix;
    constraint_matrix_comp.makeCompressed();
    constraint_matrix = {
      constraint_matrix_comp.outerIndexPtr()[num_variables],
      num_constraints,
      num_variables,
      const_cast<c_int *>(constraint_matrix_comp.outerIndexPtr()),
      const_cast<c_int *>(constraint_matrix_comp.innerIndexPtr()),
      const_cast<double *>(constraint_matrix_comp.valuePtr()),
      -1};
  } else {
    constraint_matrix = {
      instance.constraint_matrix.outerIndexPtr()[num_variables],
      num_constraints,
      num_variables,
      const_cast<c_int *>(instance.constraint_matrix.outerIndexPtr()),
      const_cast<c_int *>(instance.constraint_matrix.innerIndexPtr()),
      const_cast<double *>(instance.constraint_matrix.valuePtr()),
      -1};
  }

  data.A = &constraint_matrix;

  data.q = const_cast<double *>(instance.objective_vector.data());
  data.l = clipped_lower_bounds.data();
  data.u = clipped_upper_bounds.data();

  ::OSQPSettings osqp_settings = ToInternalSettings(settings);

  OSQPWorkspace * workspace = nullptr;
  const int return_code = osqp_setup(&workspace, &data, &osqp_settings);
  workspace_.reset(static_cast<OSQPWorkspaceHelper *>(workspace));
  if (return_code == 0) {
    return Status::OkStatus();
  }
  switch (static_cast<osqp_error_type>(return_code)) {
    case OSQP_DATA_VALIDATION_ERROR:
      return Status::InvalidArgumentError(
        "Unable to initialize OSQP: data validation error.");
    case OSQP_SETTINGS_VALIDATION_ERROR:
      return Status::InvalidArgumentError(
        "Unable to initialize OSQP: invalid settings.");
    case OSQP_LINSYS_SOLVER_LOAD_ERROR:
      // This should never happen because qdldl is statically linked in.
      return Status::UnknownError(
        "Unable to initialize OSQP: unable to load linear solver.");
    case OSQP_LINSYS_SOLVER_INIT_ERROR:
      return Status::UnknownError(
        "Unable to initialize OSQP: unable to initialize linear solver.");
    case OSQP_NONCVX_ERROR:
      return Status::InvalidArgumentError(
        "Unable to initialize OSQP: the problem appears non-convex.");
    case OSQP_MEM_ALLOC_ERROR:
      return Status::UnknownError(
        "Unable to initialize OSQP: memory allocation error.");
    case OSQP_WORKSPACE_NOT_INIT_ERROR:
      return Status::UnknownError(
        "Unable to initialize OSQP: workspace not initialized.");
  }
  return Status::UnknownError(
    "Unable to initialize OSQP: unrecognized error code.");
}

namespace
{

Status VerifySameSparsity(
  const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> & new_matrix,
  const csc * ref_matrix, size_t num_variables)
{
  if (new_matrix.nonZeros() != ref_matrix->p[num_variables]) {
    return Status::InvalidArgumentError(
      "The new new matrix should have the same number of non-zero "
      "elements.");
  }

  for (size_t i = 0; i < num_variables; ++i) {
    if (ref_matrix->p[i] != new_matrix.outerIndexPtr()[i]) {
      return Status::InvalidArgumentError(
        "Sparsity of the new matrix differs from the previously "
        "defined matrix.");
    }
  }
  for (int64_t i = 0; i < new_matrix.innerSize(); ++i) {
    if (ref_matrix->i[i] != new_matrix.innerIndexPtr()[i]) {
      return Status::InvalidArgumentError(
        "Sparsity of the new matrix differs from the previously "
        "defined matrix.");
    }
  }

  return Status::OkStatus();
}

// Helper function for calling osqp_update_P with an upper triangular objective
// matrix. Assumes objective_matrix_upper_triangle is always upper triangular.
Status UpdateUpperTriangularObjectiveMatrix(
  const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> &
  objective_matrix_upper_triangle,
  OSQPWorkspaceHelper * workspace)
{
  const c_int num_variables = workspace->data->n;

  if (objective_matrix_upper_triangle.rows() !=
    objective_matrix_upper_triangle.cols() ||
    objective_matrix_upper_triangle.rows() != num_variables)
  {
    return Status::InvalidArgumentError(
      "The new objective matrix should be square with dimension equal to the "
      "number of variables. Matrix dimensions: " +
      std::to_string(objective_matrix_upper_triangle.rows()) +
      " x " + std::to_string(objective_matrix_upper_triangle.cols()) +
      ", num_variables=" + std::to_string(num_variables));
  }

  OSQP_RETURN_IF_ERROR(
    VerifySameSparsity(
      objective_matrix_upper_triangle,
      workspace->data->P, num_variables));

  c_int nnzP = objective_matrix_upper_triangle.nonZeros();

  const int return_code = osqp_update_P(
    workspace, objective_matrix_upper_triangle.valuePtr(), OSQP_NULL, nnzP);
  if (return_code == 0) {
    return Status::OkStatus();
  }
  return Status::UnknownError(
    "Unable to update OSQP P matrix: unrecognized error code.");
}

// Helper function for calling osqp_update_P_A with an upper triangular
// objective matrix. Assumes objective_matrix_upper_triangle is always upper
// triangular.
Status UpdateUpperTriangularObjectiveMatrixAndConstraintMatrix(
  const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> &
  objective_matrix_upper_triangle,
  const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> &
  constraint_matrix,
  OSQPWorkspaceHelper * workspace)
{
  const c_int num_variables = workspace->data->n;

  if (objective_matrix_upper_triangle.rows() !=
    objective_matrix_upper_triangle.cols() ||
    objective_matrix_upper_triangle.rows() != num_variables)
  {
    return Status::InvalidArgumentError(
      "The new objective matrix should be square with dimension equal to the "
      "number of variables. Matrix dimensions: " +
      std::to_string(objective_matrix_upper_triangle.rows()) +
      " x " + std::to_string(objective_matrix_upper_triangle.cols()) +
      ", num_variables=" + std::to_string(num_variables));
  }
  if (constraint_matrix.cols() != num_variables) {
    return Status::InvalidArgumentError(
      "The new constraint matrix should column size equal to the "
      "number of variables. Matrix dimensions: " +
      std::to_string(constraint_matrix.rows()) +
      " x " + std::to_string(constraint_matrix.cols()) +
      ", num_variables=" + std::to_string(num_variables));
  }

  OSQP_RETURN_IF_ERROR(
    VerifySameSparsity(
      objective_matrix_upper_triangle,
      workspace->data->P, num_variables));

  c_int nnzP = objective_matrix_upper_triangle.nonZeros();

  OSQP_RETURN_IF_ERROR(
    VerifySameSparsity(constraint_matrix, workspace->data->A, num_variables));

  c_int nnzA = constraint_matrix.nonZeros();

  const int return_code = osqp_update_P_A(
    workspace, objective_matrix_upper_triangle.valuePtr(), OSQP_NULL, nnzP,
    constraint_matrix.valuePtr(), OSQP_NULL, nnzA);
  if (return_code == 0) {
    return Status::OkStatus();
  }
  return Status::UnknownError(
    "Unable to update OSQP P and A matrix: unrecognized error code.");
}

// Returns true if the sparse matrix 'matrix' is upper triangular.
bool IsUpperTriangular(
  const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> & matrix)
{
  // Iterate through all non-zero elements, and ensure that their indices are
  // only on the upper-right triangle, including the diagonal.
  for (int i = 0; i < matrix.outerSize(); ++i) {
    for (Eigen::SparseMatrix<double, Eigen::ColMajor, c_int>::InnerIterator it(
        matrix, i);
      it; ++it)
    {
      if (it.col() < it.row()) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

Status OsqpSolver::UpdateObjectiveMatrix(
  const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> &
  objective_matrix)
{
  // If the objective matrix is already upper triangular, we can skip the
  // temporary.
  if (IsUpperTriangular(objective_matrix)) {
    return UpdateUpperTriangularObjectiveMatrix(
      objective_matrix,
      workspace_.get());
  }

  // If not upper triangular, make a temporary.
  Eigen::SparseMatrix<double, Eigen::ColMajor, c_int>
  objective_matrix_upper_triangle =
    objective_matrix.triangularView<Eigen::Upper>();
  return UpdateUpperTriangularObjectiveMatrix(
    objective_matrix_upper_triangle,
    workspace_.get());
}

Status OsqpSolver::UpdateConstraintMatrix(
  const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> &
  constraint_matrix)
{
  const c_int num_variables = workspace_->data->n;

  if (constraint_matrix.cols() != num_variables) {
    return Status::InvalidArgumentError(
      "The new constraint matrix should column size equal to the "
      "number of variables. Matrix dimensions: " +
      std::to_string(constraint_matrix.rows()) +
      " x " + std::to_string(constraint_matrix.cols()) +
      ", num_variables=" + std::to_string(num_variables));
  }

  OSQP_RETURN_IF_ERROR(
    VerifySameSparsity(
      constraint_matrix,
      workspace_->data->A, num_variables));

  c_int nnzA = constraint_matrix.nonZeros();

  const int return_code = osqp_update_A(
    workspace_.get(), constraint_matrix.valuePtr(), OSQP_NULL, nnzA);
  if (return_code == 0) {
    return Status::OkStatus();
  }
  return Status::UnknownError(
    "Unable to update OSQP A matrix: unrecognized error code.");
}

Status OsqpSolver::UpdateObjectiveAndConstraintMatrices(
  const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> & objective_matrix,
  const Eigen::SparseMatrix<double, Eigen::ColMajor, c_int> &
  constraint_matrix)
{
  // If the objective matrix is already upper triangular, we can skip the
  // temporary.
  if (IsUpperTriangular(objective_matrix)) {
    return UpdateUpperTriangularObjectiveMatrixAndConstraintMatrix(
      objective_matrix, constraint_matrix, workspace_.get());
  }

  // If not upper triangular, make a temporary.
  Eigen::SparseMatrix<double, Eigen::ColMajor, c_int>
  objective_matrix_upper_triangle =
    objective_matrix.triangularView<Eigen::Upper>();
  return UpdateUpperTriangularObjectiveMatrixAndConstraintMatrix(
    objective_matrix_upper_triangle, constraint_matrix, workspace_.get());
}

namespace
{
OsqpExitCode StatusToExitCode(const c_int status_val)
{
  switch (status_val) {
    case OSQP_SOLVED:
      return OsqpExitCode::kOptimal;
    case OSQP_SOLVED_INACCURATE:
      return OsqpExitCode::kOptimalInaccurate;
    case OSQP_PRIMAL_INFEASIBLE:
      return OsqpExitCode::kPrimalInfeasible;
    case OSQP_PRIMAL_INFEASIBLE_INACCURATE:
      return OsqpExitCode::kPrimalInfeasibleInaccurate;
    case OSQP_DUAL_INFEASIBLE:
      return OsqpExitCode::kDualInfeasible;
    case OSQP_DUAL_INFEASIBLE_INACCURATE:
      return OsqpExitCode::kDualInfeasibleInaccurate;
    case OSQP_MAX_ITER_REACHED:
      return OsqpExitCode::kMaxIterations;
    case OSQP_SIGINT:
      return OsqpExitCode::kInterrupted;
    case OSQP_NON_CVX:
      return OsqpExitCode::kNonConvex;
    default:
      return OsqpExitCode::kUnknown;
  }
}

}  // namespace

OsqpExitCode OsqpSolver::Solve()
{
  OSQP_CHECK(IsInitialized());
  if (osqp_solve(workspace_.get()) != 0) {
    // From looking at the code, this can happen if the solve is interrupted
    // with ctrl-c or if updating "rho" fails.
    if (osqp_is_interrupted()) {
      return OsqpExitCode::kInterrupted;
    }
    return OsqpExitCode::kUnknown;
  }
  return StatusToExitCode(workspace_->info->status_val);
}

c_int OsqpSolver::iterations() const
{
  OSQP_CHECK(IsInitialized());
  return workspace_->info->iter;
}

double OsqpSolver::objective_value() const
{
  OSQP_CHECK(IsInitialized());
  return workspace_->info->obj_val;
}

Map<const VectorXd> OsqpSolver::primal_solution() const
{
  OSQP_CHECK(IsInitialized());
  return Map<const VectorXd>(workspace_->solution->x, workspace_->data->n);
}

Map<const VectorXd> OsqpSolver::dual_solution() const
{
  OSQP_CHECK(IsInitialized());
  return Map<const VectorXd>(workspace_->solution->y, workspace_->data->m);
}

Map<const VectorXd> OsqpSolver::primal_infeasibility_certificate() const
{
  OSQP_CHECK(IsInitialized());
  const OsqpExitCode exit_code = StatusToExitCode(workspace_->info->status_val);
  OSQP_CHECK(
    exit_code == OsqpExitCode::kPrimalInfeasible ||
    exit_code == OsqpExitCode::kPrimalInfeasibleInaccurate);
  (void)exit_code;
  return Map<const VectorXd>(workspace_->delta_y, workspace_->data->m);
}

Status OsqpSolver::SetWarmStart(
  const Ref<const VectorXd> & primal_vector,
  const Ref<const VectorXd> & dual_vector)
{
  // This is identical to calling osqp_warm_start with both vectors at once.
  OSQP_RETURN_IF_ERROR(SetPrimalWarmStart(primal_vector));
  OSQP_RETURN_IF_ERROR(SetDualWarmStart(dual_vector));
  return Status::OkStatus();
}

Status OsqpSolver::SetPrimalWarmStart(
  const Ref<const VectorXd> & primal_vector)
{
  if (!IsInitialized()) {
    return Status::FailedPreconditionError("OsqpSolver is not initialized.");
  }
  const c_int num_variables = workspace_->data->n;
  OSQP_RETURN_IF_ERROR(
    OSQP_CHECK_DIMENSIONS(primal_vector.size(), num_variables));

  const int return_code =
    osqp_warm_start_x(workspace_.get(), primal_vector.data());
  if (return_code != 0) {
    return Status::UnknownError("osqp_warm_start_x unexpectedly failed.");
  }
  return Status::OkStatus();
}

Status OsqpSolver::SetDualWarmStart(
  const Ref<const VectorXd> & dual_vector)
{
  if (!IsInitialized()) {
    return Status::FailedPreconditionError("OsqpSolver is not initialized.");
  }
  const c_int num_constraints = workspace_->data->m;
  OSQP_RETURN_IF_ERROR(
    OSQP_CHECK_DIMENSIONS(dual_vector.size(), num_constraints));

  const int return_code =
    osqp_warm_start_y(workspace_.get(), dual_vector.data());
  if (return_code != 0) {
    return Status::UnknownError("osqp_warm_start_y unexpectedly failed.");
  }
  return Status::OkStatus();
}

Status OsqpSolver::SetObjectiveVector(
  const Ref<const VectorXd> & objective_vector)
{
  if (!IsInitialized()) {
    return Status::FailedPreconditionError("OsqpSolver is not initialized.");
  }
  const c_int num_variables = workspace_->data->n;
  OSQP_RETURN_IF_ERROR(
    OSQP_CHECK_DIMENSIONS(objective_vector.size(), num_variables));

  const int return_code =
    osqp_update_lin_cost(workspace_.get(), objective_vector.data());
  if (return_code != 0) {
    return Status::UnknownError("osqp_update_lin_cost unexpectedly failed.");
  }
  return Status::OkStatus();
}

// NOTE(ml): osqp_update_lower_bound and osqp_update_upper_bound are not
// exposed because they have confusing semantics. They immediately error if a
// new set of bounds is inconsistent with the existing bounds on the other side.
Status OsqpSolver::SetBounds(
  const Ref<const VectorXd> & lower_bounds,
  const Ref<const VectorXd> & upper_bounds)
{
  if (!IsInitialized()) {
    return Status::FailedPreconditionError("OsqpSolver is not initialized.");
  }
  const c_int num_constraints = workspace_->data->m;
  OSQP_RETURN_IF_ERROR(
    OSQP_CHECK_DIMENSIONS(lower_bounds.size(), num_constraints));
  OSQP_RETURN_IF_ERROR(
    OSQP_CHECK_DIMENSIONS(upper_bounds.size(), num_constraints));
  // OSQP does this check internally, but we can return a better error message.
  for (int i = 0; i < num_constraints; i++) {
    if (lower_bounds[i] > upper_bounds[i]) {
      return Status::InvalidArgumentError(
        "Inconsistent bounds at index " + std::to_string(i) + ", " +
        std::to_string(lower_bounds[i]) + " must be <= " + std::to_string(upper_bounds[i]) + ".");
    }
  }

  const int return_code = osqp_update_bounds(
    workspace_.get(), lower_bounds.data(), upper_bounds.data());
  if (return_code != 0) {
    return Status::UnknownError("osqp_update_bounds unexpectedly failed.");
  }
  return Status::OkStatus();
}

Status OsqpSolver::UpdateMaxIter(int max_iter_new)
{
  if (!IsInitialized()) {
    return Status::FailedPreconditionError("OsqpSolver is not initialized.");
  }
  if (max_iter_new <= 0) {
    return Status::InvalidArgumentError("Invalid max_iter value: " + std::to_string(max_iter_new));
  }
  if (osqp_update_max_iter(workspace_.get(), max_iter_new) != 0) {
    return Status::UnknownError("osqp_update_max_iter unexpectedly failed.");
  }
  return Status::OkStatus();
}

Status OsqpSolver::UpdateEpsAbs(double eps_abs_new)
{
  if (!IsInitialized()) {
    return Status::FailedPreconditionError("OsqpSolver is not initialized.");
  }
  if (eps_abs_new <= 0.0) {
    return Status::InvalidArgumentError("Invalid eps_abs value: " + std::to_string(eps_abs_new));
  }
  if (osqp_update_eps_abs(workspace_.get(), eps_abs_new) != 0) {
    return Status::UnknownError("osqp_update_eps_abs unexpectedly failed.");
  }
  return Status::OkStatus();
}

Status OsqpSolver::UpdateTimeLimit(double time_limit_new)
{
  if (!IsInitialized()) {
    return Status::FailedPreconditionError("OsqpSolver is not initialized.");
  }
  if (time_limit_new < 0.0) {
    return Status::InvalidArgumentError(
      "Invalid time_limit value: " + std::to_string(time_limit_new));
  }
  return Status::OkStatus();
}

#undef OSQP_CHECK_DIMENSIONS
#undef OSQP_RETURN_IF_ERROR
#undef OSQP_CHECK

}  // namespace osqp
