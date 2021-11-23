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

#ifndef SMOOTH__FEEDBACK__COMPAT__IPOPT_HPP_
#define SMOOTH__FEEDBACK__COMPAT__IPOPT_HPP_

#define HAVE_CSTDDEF
#include <IpIpoptApplication.hpp>
#include <IpIpoptData.hpp>
#include <IpTNLP.hpp>
#undef HAVE_CSTDDEF

#include "smooth/feedback/nlp.hpp"

namespace smooth::feedback {

class IpoptNLP : public Ipopt::TNLP
{
public:
  /**
   * @brief Ipopt wrapper for NLP (rvlaue version).
   */
  inline IpoptNLP(NLP && nlp) : nlp_(std::move(nlp)) {}

  /**
   * @brief Ipopt wrapper for NLP (lvalue version).
   */
  inline IpoptNLP(const NLP & nlp) : nlp_(nlp) {}

  /**
   * @brief Access solution.
   */
  inline NLPSolution & sol() { return sol_; }

  /**
   * @brief IPOPT info overload
   */
  inline bool get_nlp_info(Ipopt::Index & n,
    Ipopt::Index & m,
    Ipopt::Index & nnz_jac_g,
    Ipopt::Index & nnz_h_lag,
    IndexStyleEnum & index_style) override
  {
    n = nlp_.n;
    m = nlp_.m;

    const auto J = nlp_.dg_dx(Eigen::VectorXd::Zero(n));

    nnz_jac_g   = J.nonZeros();
    nnz_h_lag   = 0;              // not used
    index_style = TNLP::C_STYLE;  // zero-based

    return true;
  }

  /**
   * @brief IPOPT bounds overload
   */
  inline bool get_bounds_info(Ipopt::Index n,
    Ipopt::Number * x_l,
    Ipopt::Number * x_u,
    Ipopt::Index m,
    Ipopt::Number * g_l,
    Ipopt::Number * g_u) override
  {
    Eigen::Map<Eigen::VectorXd>(x_l, n) = nlp_.xl;
    Eigen::Map<Eigen::VectorXd>(x_u, n) = nlp_.xu;
    Eigen::Map<Eigen::VectorXd>(g_l, m) = nlp_.gl;
    Eigen::Map<Eigen::VectorXd>(g_u, m) = nlp_.gu;

    return true;
  }

  /**
   * @brief IPOPT initial guess overload
   */
  inline bool get_starting_point(Ipopt::Index n,
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
   * @brief IPOPT objective overload
   */
  inline bool eval_f(Ipopt::Index n,
    const Ipopt::Number * x,
    [[maybe_unused]] bool new_x,
    Ipopt::Number & obj_value) override
  {
    obj_value = nlp_.f(Eigen::Map<const Eigen::VectorXd>(x, n));
    return true;
  }

  /**
   * @brief IPOPT method to define initial guess
   */
  inline bool eval_grad_f(Ipopt::Index n,
    const Ipopt::Number * x,
    [[maybe_unused]] bool new_x,
    Ipopt::Number * grad_f) override
  {
    Eigen::Map<Eigen::RowVectorXd>(grad_f, n) =
      Eigen::MatrixXd(nlp_.df_dx(Eigen::Map<const Eigen::VectorXd>(x, n)));
    return true;
  }

  /**
   * @brief IPOPT method to define initial guess
   */
  inline bool eval_g(Ipopt::Index n,
    const Ipopt::Number * x,
    [[maybe_unused]] bool new_x,
    Ipopt::Index m,
    Ipopt::Number * g) override
  {
    Eigen::Map<Eigen::VectorXd>(g, m) = nlp_.g(Eigen::Map<const Eigen::VectorXd>(x, n));
    return true;
  }

  /**
   * @brief IPOPT method to define initial guess
   */
  inline bool eval_jac_g(Ipopt::Index n,
    const Ipopt::Number * x,
    [[maybe_unused]] bool new_x,
    [[maybe_unused]] Ipopt::Index m,
    Ipopt::Index nele_jac,
    Ipopt::Index * iRow,
    Ipopt::Index * jCol,
    Ipopt::Number * values) override
  {
    if (values == NULL) {
      const auto J = nlp_.dg_dx(Eigen::VectorXd::Zero(n));
      assert(nele_jac == J.nonZeros());

      for (auto cntr = 0u, col = 0u; col < J.cols(); ++col) {
        for (typename decltype(J)::InnerIterator it(J, col); it; ++it) {
          iRow[cntr]   = it.index();
          jCol[cntr++] = col;
        }
      }
    } else {
      const auto J = nlp_.dg_dx(Eigen::Map<const Eigen::VectorXd>(x, n));
      assert(nele_jac == J.nonZeros());

      for (auto cntr = 0u, col = 0u; col < J.cols(); ++col) {
        for (typename decltype(J)::InnerIterator it(J, col); it; ++it) {
          values[cntr++] = it.value();
        }
      }
    }
    return true;
  }

  /**
   * @brief IPOPT method called after optimization done
   */
  inline void finalize_solution(Ipopt::SolverReturn status,
    Ipopt::Index n,
    const Ipopt::Number * x,
    [[maybe_unused]] const Ipopt::Number * z_L,
    [[maybe_unused]] const Ipopt::Number * z_U,
    Ipopt::Index m,
    [[maybe_unused]] const Ipopt::Number * g,
    [[maybe_unused]] const Ipopt::Number * lambda,
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
  NLP nlp_;
  NLPSolution sol_;
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
inline NLPSolution solve_nlp_ipopt(const NLP & nlp,
  std::optional<NLPSolution> warmstart                         = {},
  std::vector<std::pair<std::string, int>> opts_integer        = {},
  std::vector<std::pair<std::string, std::string>> opts_string = {},
  std::vector<std::pair<std::string, double>> opts_numeric     = {})
{
  Ipopt::SmartPtr<IpoptNLP> ipopt_nlp          = new IpoptNLP(nlp);
  Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

  for (auto [opt, val] : opts_integer) { app->Options()->SetIntegerValue(opt, val); }
  for (auto [opt, val] : opts_string) { app->Options()->SetStringValue(opt, val); }
  for (auto [opt, val] : opts_numeric) { app->Options()->SetNumericValue(opt, val); }

  if (warmstart.has_value()) {
    // initial guess is given
    app->Options()->SetStringValue("warm_start_init_point", "yes");
    ipopt_nlp->sol() = warmstart.value();
  } else {
    // initial guess is zero
    app->Options()->SetStringValue("warm_start_init_point", "no");
    ipopt_nlp->sol().x.setZero(nlp.n);
  }

  app->Initialize();
  app->OptimizeTNLP(ipopt_nlp);

  return ipopt_nlp->sol();
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COMPAT__IPOPT_HPP_
