// smooth_feedback: Control theory on Lie groups
// https://github.com/pettni/smooth_feedback
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
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

#ifndef SMOOTH__FEEDBACK__COMPAT__IPOPT_HPP_
#define SMOOTH__FEEDBACK__COMPAT__IPOPT_HPP_

/**
 * @file
 * @brief Solve nonlinear programs with Ipopt.
 */

#define HAVE_CSTDDEF
#include <IpIpoptApplication.hpp>
#include <IpIpoptData.hpp>
#include <IpTNLP.hpp>
#undef HAVE_CSTDDEF

#include "smooth/feedback/nlp.hpp"

namespace smooth::feedback {

/**
 * @brief Ipopt interface to solve NLPs
 *
 * @see NLP
 */
template<NLP Problem>
class IpoptNLP : public Ipopt::TNLP
{
public:
  /**
   * @brief Ipopt wrapper for NLP (rvlaue version).
   */
  inline IpoptNLP(Problem && nlp, bool use_hessian = false)
      : nlp_(std::move(nlp)), use_hessian_(use_hessian)
  {}

  /**
   * @brief Ipopt wrapper for NLP (lvalue version).
   */
  inline IpoptNLP(const Problem & nlp, bool use_hessian = false)
      : nlp_(nlp), use_hessian_(use_hessian)
  {}

  /**
   * @brief Access solution.
   */
  inline NLPSolution & sol() { return sol_; }

  /**
   * @brief IPOPT info overload
   */
  inline bool get_nlp_info(
    Ipopt::Index & n,
    Ipopt::Index & m,
    Ipopt::Index & nnz_jac_g,
    Ipopt::Index & nnz_h_lag,
    IndexStyleEnum & index_style) override
  {
    n = nlp_.n();
    m = nlp_.m();

    const auto J = nlp_.dg_dx(Eigen::VectorXd::Zero(n));
    nnz_jac_g    = J.nonZeros();

    nnz_h_lag = 0;  // default
    if constexpr (HessianNLP<Problem>) {
      if (use_hessian_) {
        H_ = nlp_.d2f_dx2(Eigen::VectorXd::Zero(n));
        H_ += nlp_.d2g_dx2(Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(m));
        H_.makeCompressed();
        nnz_h_lag = H_.nonZeros();
      }
    } else {
      assert(use_hessian_ == false);
    }

    index_style = TNLP::C_STYLE;  // zero-based

    return true;
  }

  /**
   * @brief IPOPT bounds overload
   */
  inline bool get_bounds_info(
    Ipopt::Index n,
    Ipopt::Number * x_l,
    Ipopt::Number * x_u,
    Ipopt::Index m,
    Ipopt::Number * g_l,
    Ipopt::Number * g_u) override
  {
    Eigen::Map<Eigen::VectorXd>(x_l, n) = nlp_.xl().cwiseMax(Eigen::VectorXd::Constant(n, -2e19));
    Eigen::Map<Eigen::VectorXd>(x_u, n) = nlp_.xu().cwiseMin(Eigen::VectorXd::Constant(n, 2e19));
    Eigen::Map<Eigen::VectorXd>(g_l, m) = nlp_.gl().cwiseMax(Eigen::VectorXd::Constant(m, -2e19));
    Eigen::Map<Eigen::VectorXd>(g_u, m) = nlp_.gu().cwiseMin(Eigen::VectorXd::Constant(m, 2e19));

    return true;
  }

  /**
   * @brief IPOPT initial guess overload
   */
  inline bool get_starting_point(
    Ipopt::Index n,
    bool init_x,
    Ipopt::Number * x,
    bool init_z,
    Ipopt::Number * z_L,
    Ipopt::Number * z_U,
    Ipopt::Index m,
    bool init_lambda,
    Ipopt::Number * lambda) override
  {
    if (init_x) { Eigen::Map<Eigen::VectorXd>(x, n) = sol_.x; }

    if (init_z) {
      Eigen::Map<Eigen::VectorXd>(z_L, n) = sol_.zl;
      Eigen::Map<Eigen::VectorXd>(z_U, n) = sol_.zu;
    }

    if (init_lambda) { Eigen::Map<Eigen::VectorXd>(lambda, m) = sol_.lambda; }

    return true;
  }

  /**
   * @brief IPOPT method to define objective
   */
  inline bool eval_f(
    Ipopt::Index n,
    const Ipopt::Number * x,
    [[maybe_unused]] bool new_x,
    Ipopt::Number & obj_value) override
  {
    obj_value = nlp_.f(Eigen::Map<const Eigen::VectorXd>(x, n));
    return true;
  }

  /**
   * @brief IPOPT method to define gradient of objective
   */
  inline bool eval_grad_f(
    Ipopt::Index n,
    const Ipopt::Number * x,
    [[maybe_unused]] bool new_x,
    Ipopt::Number * grad_f) override
  {
    const auto & df_dx = nlp_.df_dx(Eigen::Map<const Eigen::VectorXd>(x, n));
    for (auto i = 0; i < n; ++i) { grad_f[i] = df_dx.coeff(0, i); }
    return true;
  }

  /**
   * @brief IPOPT method to define constraint function
   */
  inline bool eval_g(
    Ipopt::Index n,
    const Ipopt::Number * x,
    [[maybe_unused]] bool new_x,
    Ipopt::Index m,
    Ipopt::Number * g) override
  {
    Eigen::Map<Eigen::VectorXd>(g, m) = nlp_.g(Eigen::Map<const Eigen::VectorXd>(x, n));
    return true;
  }

  /**
   * @brief IPOPT method to define jacobian of constraint function
   */
  inline bool eval_jac_g(
    Ipopt::Index n,
    const Ipopt::Number * x,
    [[maybe_unused]] bool new_x,
    [[maybe_unused]] Ipopt::Index m,
    [[maybe_unused]] Ipopt::Index nele_jac,
    Ipopt::Index * iRow,
    Ipopt::Index * jCol,
    Ipopt::Number * values) override
  {
    if (values == NULL) {
      const auto & J = nlp_.dg_dx(Eigen::VectorXd::Zero(n));
      assert(nele_jac == J.nonZeros());

      for (auto cntr = 0u, od = 0u; od < J.outerSize(); ++od) {
        for (Eigen::InnerIterator it(J, od); it; ++it) {
          iRow[cntr]   = it.row();
          jCol[cntr++] = it.col();
        }
      }
    } else {
      const auto & J = nlp_.dg_dx(Eigen::Map<const Eigen::VectorXd>(x, n));
      assert(nele_jac == J.nonZeros());

      for (auto cntr = 0u, od = 0u; od < J.outerSize(); ++od) {
        for (Eigen::InnerIterator it(J, od); it; ++it) { values[cntr++] = it.value(); }
      }
    }
    return true;
  }

  /**
   * @brief IPOPT method to define problem Hessian
   */
  inline bool eval_h(
    Ipopt::Index n,
    const Ipopt::Number * x,
    [[maybe_unused]] bool new_x,
    Ipopt::Number sigma,
    Ipopt::Index m,
    const Ipopt::Number * lambda,
    [[maybe_unused]] bool new_lambda,
    [[maybe_unused]] Ipopt::Index nele_hess,
    Ipopt::Index * iRow,
    Ipopt::Index * jCol,
    Ipopt::Number * values) override
  {
    assert(use_hessian_);
    assert(HessianNLP<Problem>);
    assert(H_.isCompressed());
    assert(H_.nonZeros() == nele_hess);

    H_.coeffs().setZero();

    if (values == NULL) {
      for (auto cntr = 0u, od = 0u; od < H_.outerSize(); ++od) {
        for (Eigen::InnerIterator it(H_, od); it; ++it) {
          // transpose: H upper triangular but ipopt expects lower-triangular
          iRow[cntr]   = it.col();
          jCol[cntr++] = it.row();
        }
      }
    } else {
      Eigen::Map<const Eigen::VectorXd> xvar(x, n);
      Eigen::Map<const Eigen::VectorXd> lvar(lambda, m);

      H_ += nlp_.d2f_dx2(xvar);
      H_ *= sigma;
      H_ += nlp_.d2g_dx2(xvar, lvar);

      for (auto cntr = 0u, od = 0u; od < H_.outerSize(); ++od) {
        for (Eigen::InnerIterator it(H_, od); it; ++it) { values[cntr++] = it.value(); }
      }
    }

    return true;
  }

  /**
   * @brief IPOPT method called after optimization done
   */
  inline void finalize_solution(
    Ipopt::SolverReturn status,
    Ipopt::Index n,
    const Ipopt::Number * x,
    const Ipopt::Number * z_L,
    const Ipopt::Number * z_U,
    Ipopt::Index m,
    [[maybe_unused]] const Ipopt::Number * g,
    const Ipopt::Number * lambda,
    Ipopt::Number obj_value,
    const Ipopt::IpoptData * ip_data,
    [[maybe_unused]] Ipopt::IpoptCalculatedQuantities * ip_cq) override
  {
    switch (status) {
    case Ipopt::SolverReturn::SUCCESS:
      sol_.status = NLPSolution::Status::Optimal;
      break;
    case Ipopt::SolverReturn::STOP_AT_ACCEPTABLE_POINT:
      sol_.status = NLPSolution::Status::Optimal;
      break;
    case Ipopt::SolverReturn::MAXITER_EXCEEDED:
      sol_.status = NLPSolution::Status::MaxIterations;
      break;
    case Ipopt::SolverReturn::CPUTIME_EXCEEDED:
      sol_.status = NLPSolution::Status::MaxTime;
      break;
    case Ipopt::SolverReturn::LOCAL_INFEASIBILITY:
      sol_.status = NLPSolution::Status::PrimalInfeasible;
      break;
    case Ipopt::SolverReturn::DIVERGING_ITERATES:
      sol_.status = NLPSolution::Status::DualInfeasible;
      break;
    default:
      sol_.status = NLPSolution::Status::Unknown;
      break;
    }

    sol_.iter = ip_data->iter_count();

    sol_.x      = Eigen::Map<const Eigen::VectorXd>(x, n);
    sol_.zl     = Eigen::Map<const Eigen::VectorXd>(z_L, n);
    sol_.zu     = Eigen::Map<const Eigen::VectorXd>(z_U, n);
    sol_.lambda = Eigen::Map<const Eigen::VectorXd>(lambda, m);

    sol_.objective = obj_value;
  }

private:
  Problem nlp_;
  bool use_hessian_;
  NLPSolution sol_;
  Eigen::SparseMatrix<double> H_;
};

/**
 * @brief Solve an NLP with the Ipopt solver
 *
 * @param nlp problem to solve
 * @param warmstart problem warm-start point
 * @param opts_integer key-value list of Ipopt integer options
 * @param opts_string key-value list of Ipopt string options
 * @param opts_numeric key-value list of Ipopt numeric options
 *
 * @see https://coin-or.github.io/Ipopt/OPTIONS.html for a list of available options
 */
inline NLPSolution solve_nlp_ipopt(
  NLP auto && nlp,
  std::optional<NLPSolution> warmstart                         = {},
  std::vector<std::pair<std::string, int>> opts_integer        = {},
  std::vector<std::pair<std::string, std::string>> opts_string = {},
  std::vector<std::pair<std::string, double>> opts_numeric     = {})
{
  using nlp_t      = std::decay_t<decltype(nlp)>;
  bool use_hessian = false;
  const auto n     = nlp.n();

  auto it = std::find_if(opts_string.begin(), opts_string.end(), [](const auto & p) {
    return p.first == "hessian_approximation";
  });
  if (it != opts_string.end() && it->second == "exact") { use_hessian = true; }

  Ipopt::SmartPtr<IpoptNLP<nlp_t>> ipopt_nlp =
    new IpoptNLP<nlp_t>(std::forward<decltype(nlp)>(nlp), use_hessian);
  Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

  // silence welcome message
  app->Options()->SetStringValue("sb", "yes");

  if (warmstart.has_value()) {
    // initial guess is given
    app->Options()->SetStringValue("warm_start_init_point", "yes");
    ipopt_nlp->sol() = warmstart.value();
  } else {
    // initial guess not given, set to zero
    app->Options()->SetStringValue("warm_start_init_point", "no");
    ipopt_nlp->sol().x.setZero(n);
  }

  // override with user-provided options
  for (auto [opt, val] : opts_integer) { app->Options()->SetIntegerValue(opt, val); }
  for (auto [opt, val] : opts_string) { app->Options()->SetStringValue(opt, val); }
  for (auto [opt, val] : opts_numeric) { app->Options()->SetNumericValue(opt, val); }

  app->Initialize();
  app->OptimizeTNLP(ipopt_nlp);

  return ipopt_nlp->sol();
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COMPAT__IPOPT_HPP_
