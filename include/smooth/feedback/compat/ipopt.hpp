#ifndef SMOOTH__FEEDBACK__COMPAT__IPOPT_HPP_
#define SMOOTH__FEEDBACK__COMPAT__IPOPT_HPP_

#define HAVE_CSTDDEF
#include <coin/IpIpoptApplication.hpp>
#include <coin/IpTNLP.hpp>
#undef HAVE_CSTDDEF

#include "smooth/feedback/nlp.hpp"

namespace smooth::feedback {

using Ipopt::Index, Ipopt::Number;

class IpoptNLP : public Ipopt::TNLP
{
public:
  /**
   * @brief Ipopt wrapper for NLP (rvlaue version).
   */
  IpoptNLP(NLP && nlp) : nlp_(std::move(nlp)) {}

  /**
   * @brief Ipopt wrapper for NLP (lvalue version).
   */
  IpoptNLP(const NLP & nlp) : nlp_(nlp) {}

  /**
   * @brief Access solution.
   */
  const NLPSolution & sol() const { return sol_; }

  /**
   * @brief IPOPT info overload
   */
  bool get_nlp_info(Index & n,
    Index & m,
    Index & nnz_jac_g,
    Index & nnz_h_lag,
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
  bool get_bounds_info(
    Index n, Number * x_l, Number * x_u, Index m, Number * g_l, Number * g_u) override
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
  bool get_starting_point(Index n,
    bool init_x,
    Number * x,
    bool init_z,
    [[maybe_unused]] Number * z_L,
    [[maybe_unused]] Number * z_U,
    [[maybe_unused]] Index m,
    bool init_lambda,
    [[maybe_unused]] Number * lambda) override
  {
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);

    Eigen::Map<Eigen::VectorXd>(x, n).setZero();

    return true;
  }

  /**
   * @brief IPOPT objective overload
   */
  bool eval_f([[maybe_unused]] Index n,
    const Number * x,
    [[maybe_unused]] bool new_x,
    Number & obj_value) override
  {
    obj_value = nlp_.f(Eigen::Map<const Eigen::VectorXd>(x, n));
    return true;
  }

  /**
   * @brief IPOPT method to define initial guess
   */
  bool eval_grad_f(Index n,
    [[maybe_unused]] const Number * x,
    [[maybe_unused]] bool new_x,
    Number * grad_f) override
  {
    Eigen::Map<Eigen::VectorXd>(grad_f, n) =
      Eigen::MatrixXd(nlp_.df_dx(Eigen::Map<const Eigen::VectorXd>(x, n)));
    return true;
  }

  /**
   * @brief IPOPT method to define initial guess
   */
  bool eval_g([[maybe_unused]] Index n,
    const Number * x,
    [[maybe_unused]] bool new_x,
    Index m,
    Number * g) override
  {
    Eigen::Map<Eigen::VectorXd>(g, m) = nlp_.g(Eigen::Map<const Eigen::VectorXd>(x, n));
    return true;
  }

  /**
   * @brief IPOPT method to define initial guess
   */
  bool eval_jac_g(Index n,
    const Number * x,
    [[maybe_unused]] bool new_x,
    Index m,
    Index nele_jac,
    Index * iRow,
    Index * jCol,
    Number * values) override
  {
    assert(n == Nvars_beg().back());
    assert(m == Ncons_beg().back());


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
   * @brief IPOPT method to define initial guess
   */
  void finalize_solution([[maybe_unused]] Ipopt::SolverReturn status,
    [[maybe_unused]] Index n,
    const Number * x,
    [[maybe_unused]] const Number * z_L,
    [[maybe_unused]] const Number * z_U,
    [[maybe_unused]] Index m,
    [[maybe_unused]] const Number * g,
    [[maybe_unused]] const Number * lambda,
    [[maybe_unused]] Number obj_value,
    [[maybe_unused]] const Ipopt::IpoptData * ip_data,
    [[maybe_unused]] Ipopt::IpoptCalculatedQuantities * ip_cq) override
  {
    assert(n == Nvar);

    sol_.x = Eigen::Map<const Eigen::VectorXd>(x, n);
    sol_.z = Eigen::Map<const Eigen::VectorXd>(lambda, m);
  }

private:
  NLP nlp_;
  NLPSolution sol_;
};

inline NLPSolution solve_nlp_ipopt(const NLP & nlp)
{
  Ipopt::SmartPtr<IpoptNLP> ipopt_nlp          = new IpoptNLP(nlp);
  Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

  app->Options()->SetIntegerValue("print_level", 5);
  app->Options()->SetStringValue("print_timing_statistics", "yes");
  app->Options()->SetNumericValue("tol", 1e-6);
  app->Options()->SetStringValue("linear_solver", "mumps");
  app->Options()->SetStringValue("hessian_approximation", "limited-memory");
  app->Options()->SetStringValue("derivative_test", "first-order");
  app->Options()->SetNumericValue("derivative_test_tol", 1e-3);

  app->Initialize();
  app->OptimizeTNLP(ipopt_nlp);

  return ipopt_nlp->sol();
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COMPAT__IPOPT_HPP_
