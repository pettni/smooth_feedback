// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Optimal control problem definition.
 */

#include <iostream>

#include <Eigen/Core>
#include <smooth/concepts/lie_group.hpp>
#include <smooth/diff.hpp>

#include "traits.hpp"

namespace smooth::feedback {

// \cond
// Forward declaration
template<LieGroup _X, Manifold _U, int _Nq, int _Ncr, int _Nce>
struct OCPSolution;
// \endcond

/**
 * @brief Optimal control problem definition
 * @tparam _X state space
 * @tparam _U input space
 *
 * Problem is defined on the interval \f$ t \in [0, t_f] \f$.
 * \f[
 * \begin{cases}
 *  \min              & \theta(t_f, x_0, x_f, q)                                         \\
 *  \text{s.t.}       & x(0) = x_0                                                       \\
 *                    & x(t_f) = x_f                                                     \\
 *                    & \dot x(t) = f(t, x(t), u(t))                                     \\
 *                    & q = \int_{0}^{t_f} g(t, x(t), u(t)) \mathrm{d}t                  \\
 *                    & c_{rl} \leq c_r(t, x(t), u(t)) \leq c_{ru} \quad t \in [0, t_f]  \\
 *                    & c_{el} \leq c_e(t_f, x_0, x_f, q) \leq c_{eu}
 * \end{cases}
 * \f]
 *
 * The optimal control problem depends on arbitrary functions \f$ \theta, f, g, c_r, c_e \f$.
 * The type of those functions are template pararamters in this structure.
 *
 * @note To enable automatic differentiation \f$ \theta, f, g, c_r, c_e \f$ must be templated over
 * the scalar type.
 */
template<LieGroup _X, Manifold _U, typename Theta, typename F, typename G, typename CR, typename CE>
struct OCP
{
  /// @brief State space
  using X = _X;
  /// @brief Input space
  using U = _U;

  /// @brief State space dimension
  static constexpr int Nx = Dof<X>;
  /// @brief Input space dimension
  static constexpr int Nu = Dof<U>;
  /// @brief Number of integrals
  static constexpr int Nq = std::invoke_result_t<G, double, X, U>::SizeAtCompileTime;
  /// @brief Number of running constraints
  static constexpr int Ncr = std::invoke_result_t<CR, double, X, U>::SizeAtCompileTime;
  /// @brief Number of end constraints
  static constexpr int Nce = std::invoke_result_t<CE, double, X, X, Eigen::Matrix<double, Nq, 1>>::SizeAtCompileTime;

  /// @brief Solution type corresponding to this problem
  using Solution = OCPSolution<X, U, Nq, Ncr, Nce>;

  static_assert(Nx > 0, "Static size required");
  static_assert(Nu > 0, "Static size required");
  static_assert(Nq > 0, "Static size required");
  static_assert(Ncr > 0, "Static size required");
  static_assert(Nce > 0, "Static size required");

  /// @brief Objective function \f$ \theta : R \times X \times X \times R^{n_q} \rightarrow R \f$
  Theta theta;

  /// @brief System dynamics \f$ f : R \times X \times U \rightarrow Tangent<X> \f$
  F f;
  /// @brief Integrals \f$ g : R \times X \times U \rightarrow R^{n_q} \f$
  G g;

  /// @brief Running constraint \f$ c_r : R \times X \times U \rightarrow R^{n_{cr}} \f$
  CR cr;
  /// @brief Running constraint lower bound \f$ c_{rl} \in R^{n_{cr}} \f$
  Eigen::Vector<double, Ncr> crl = Eigen::Vector<double, Ncr>::Zero();
  /// @brief Running constraint upper bound \f$ c_{ru} \in R^{n_{cr}} \f$
  Eigen::Vector<double, Ncr> cru = Eigen::Vector<double, Ncr>::Zero();

  /// @brief End constraint \f$ c_e : R \times X \times X \times R^{n_q} \rightarrow R^{n_{ce}} \f$
  CE ce;
  /// @brief End constraint lower bound \f$ c_{el} \in R^{n_{ce}} \f$
  Eigen::Vector<double, Nce> cel = Eigen::Vector<double, Nce>::Zero();
  /// @brief End constraint upper bound \f$ c_{eu} \in R^{n_{ce}} \f$
  Eigen::Vector<double, Nce> ceu = Eigen::Vector<double, Nce>::Zero();
};

/// @brief Concept that is true for OCP specializations
template<typename T>
concept OCPType = traits::is_specialization_of_v<std::decay_t<T>, OCP>;

/// @brief Concept that is true for FlatOCP specializations
template<typename T>
concept FlatOCPType =
  OCPType<T> &&(smooth::RnType<typename std::decay_t<T>::X> && smooth::RnType<typename std::decay_t<T>::U>);

/**
 * @brief Solution to OCP.
 */
template<LieGroup _X, Manifold _U, int _Nq, int _Ncr, int _Nce>
struct OCPSolution
{
  /// @brief State space
  using X = _X;
  /// @brief Input space
  using U = _U;

  /// @brief Number of integrals
  static constexpr int Nq = _Nq;
  /// @brief Number of running constraints
  static constexpr int Ncr = _Ncr;
  /// @brief Number of end constraints
  static constexpr int Nce = _Nce;

  ///@{
  /// @brief Initial and final time
  double t0, tf;
  //}@

  /// @brief Integral values
  Eigen::Vector<double, Nq> Q{};

  ///@{
  /// @brief Callable functions for state and input
  std::function<U(double)> u;
  std::function<X(double)> x;
  //}@

  /// @brief Multipliers for integral constraints
  Eigen::Vector<double, Nq> lambda_q{};

  /// @brief Multipliers for endpoint constraints
  Eigen::Vector<double, Nce> lambda_ce{};

  /// @brief Multipliers for dynamics equality constraint
  std::function<Eigen::Vector<double, Dof<X>>(double)> lambda_dyn{};

  /// @brief Multipliers for active running constraints
  std::function<Eigen::Vector<double, Ncr>(double)> lambda_cr{};
};

/**
 * @brief Test analytic derivatives for an OCP problem.
 *
 * @tparam DT differentiation method to compare against.
 *
 * @param ocp problem to test derivatives for
 * @param num_trials number of random points to test
 *
 * @todo Make it possible to test a subset of derivatives
 */
template<diff::Type DT = diff::Type::Numerical>
bool test_ocp_derivatives(OCPType auto & ocp, uint32_t num_trials = 1, double eps = 1e-4)
{
  using OCP = std::decay_t<decltype(ocp)>;

  using X = typename OCP::X;
  using U = typename OCP::U;
  using Q = Eigen::Vector<double, OCP::Nq>;

  if (!diff::detail::diffable_order1<decltype(ocp.theta), std::tuple<double, X, X, Q>>) {
    std::cout << "no jacobian for theta\n";
  }
  if (!diff::detail::diffable_order2<decltype(ocp.theta), std::tuple<double, X, X, Q>>) {
    std::cout << "no hessian for theta\n";
  }
  if (!diff::detail::diffable_order1<decltype(ocp.f), std::tuple<double, X, U>>) { std::cout << "no jacobian for f\n"; }
  if (!diff::detail::diffable_order2<decltype(ocp.f), std::tuple<double, X, U>>) { std::cout << "no hessian for f\n"; }
  if (!diff::detail::diffable_order1<decltype(ocp.g), std::tuple<double, X, U>>) { std::cout << "no jacobian for g\n"; }
  if (!diff::detail::diffable_order2<decltype(ocp.g), std::tuple<double, X, U>>) { std::cout << "no hessian for g\n"; }
  if (!diff::detail::diffable_order1<decltype(ocp.cr), std::tuple<double, X, U>>) {
    std::cout << "no jacobian for cr\n";
  }
  if (!diff::detail::diffable_order2<decltype(ocp.cr), std::tuple<double, X, U>>) {
    std::cout << "no hessian for cr\n";
  }
  if (!diff::detail::diffable_order1<decltype(ocp.ce), std::tuple<double, X, X, Q>>) {
    std::cout << "no jacobian for ce\n";
  }
  if (!diff::detail::diffable_order2<decltype(ocp.ce), std::tuple<double, X, X, Q>>) {
    std::cout << "no hessian for ce\n";
  }

  const auto cmp = [&eps](const auto & m1, const auto & m2) {
    return (
      // clang-format off
      (m1.cols() == m2.cols())
      && (m1.rows() == m2.rows())
      && (
          m1.isApprox(m2, 1e-4) ||
          Eigen::MatrixXd(m1 - m2).cwiseAbs().maxCoeff() < eps
      )
      // clang-format on
    );
  };

  bool success = true;

  for (auto trial = 0u; trial < num_trials; ++trial) {
    // endpt parameters
    const double tf                        = 1 + static_cast<double>(std::rand()) / RAND_MAX;
    const X x0                             = Random<X>();
    const X xf                             = Random<X>();
    const Eigen::Vector<double, OCP::Nq> q = Eigen::Vector<double, OCP::Nq>::Random();
    const double t                         = 1 + static_cast<double>(std::rand()) / RAND_MAX;
    const X x                              = Random<X>();
    const U u                              = Random<U>();

    // theta
    if constexpr (diff::detail::diffable_order1<decltype(ocp.theta), std::tuple<double, X, X, Q>>) {
      const auto [f_def, df_def] = diff::dr<1, diff::Type::Analytic>(ocp.theta, wrt(tf, x0, xf, q));
      const auto [f_num, df_num] = diff::dr<1, DT>(ocp.theta, wrt(tf, x0, xf, q));

      if (!cmp(df_def, df_num)) {
        std::cout << "Error in 1st derivative of theta: got\n"
                  << Eigen::MatrixXd(df_def) << "\nbut expected\n"
                  << Eigen::MatrixXd(df_num) << '\n';
        success = false;
      };
    }
    if constexpr (diff::detail::diffable_order2<decltype(ocp.theta), std::tuple<double, X, X, Q>>) {
      const auto [f_def, df_def, d2f_def] = diff::dr<2, diff::Type::Analytic>(ocp.theta, wrt(tf, x0, xf, q));
      const auto [f_num, df_num, d2f_num] = diff::dr<2, DT>(ocp.theta, wrt(tf, x0, xf, q));

      if (!cmp(d2f_def, d2f_num)) {
        std::cout << "Error in 2nd derivative of theta: got\n"
                  << Eigen::MatrixXd(d2f_def) << "\nbut expected\n"
                  << Eigen::MatrixXd(d2f_num) << '\n';
        success = false;
      };
    }

    // end constraints
    if constexpr (diff::detail::diffable_order1<decltype(ocp.ce), std::tuple<double, X, X, Q>>) {
      const auto [f_def, df_def] = diff::dr<1, diff::Type::Analytic>(ocp.ce, wrt(tf, x0, xf, q));
      const auto [f_num, df_num] = diff::dr<1, DT>(ocp.ce, wrt(tf, x0, xf, q));

      if (!cmp(df_def, df_num)) {
        std::cout << "Error in 1st derivative of ce: got\n"
                  << Eigen::MatrixXd(df_def) << "\nbut expected\n"
                  << Eigen::MatrixXd(df_num) << '\n';
        success = false;
      };
    }
    if constexpr (diff::detail::diffable_order2<decltype(ocp.ce), std::tuple<double, X, X, Q>>) {
      const auto [f_def, df_def, d2f_def] = diff::dr<2, diff::Type::Analytic>(ocp.ce, wrt(tf, x0, xf, q));
      const auto [f_num, df_num, d2f_num] = diff::dr<2, DT>(ocp.ce, wrt(tf, x0, xf, q));

      if (!cmp(d2f_def, d2f_num)) {
        std::cout << "Error in 2nd derivative of ce: got\n"
                  << Eigen::MatrixXd(d2f_def) << "\nbut expected\n"
                  << Eigen::MatrixXd(d2f_num) << '\n';
        success = false;
      };
    }

    // dynamics
    if constexpr (diff::detail::diffable_order1<decltype(ocp.f), std::tuple<double, X, U>>) {
      const auto [f_def, df_def] = diff::dr<1, diff::Type::Analytic>(ocp.f, wrt(t, x, u));
      const auto [f_num, df_num] = diff::dr<1, DT>(ocp.f, wrt(t, x, u));
      if (!cmp(df_def, df_num)) {
        std::cout << "Error in 1st derivative of f: got\n"
                  << Eigen::MatrixXd(df_def) << "\nbut expected\n"
                  << Eigen::MatrixXd(df_num) << '\n';
        success = false;
      };
    }
    if constexpr (diff::detail::diffable_order2<decltype(ocp.f), std::tuple<double, X, U>>) {
      const auto [f_def, df_def, d2f_def] = diff::dr<2, diff::Type::Analytic>(ocp.f, wrt(t, x, u));
      const auto [f_num, df_num, d2f_num] = diff::dr<2, DT>(ocp.f, wrt(t, x, u));
      if (!cmp(d2f_def, d2f_num)) {
        std::cout << "Error in 2nd derivative of f: got\n"
                  << Eigen::MatrixXd(d2f_def) << "\nbut expected\n"
                  << Eigen::MatrixXd(d2f_num) << '\n';
        success = false;
      };
    }

    // integrand
    if constexpr (diff::detail::diffable_order1<decltype(ocp.g), std::tuple<double, X, U>>) {
      const auto [f_def, df_def] = diff::dr<1, diff::Type::Analytic>(ocp.g, wrt(t, x, u));
      const auto [f_num, df_num] = diff::dr<1, DT>(ocp.g, wrt(t, x, u));
      if (!cmp(df_def, df_num)) {
        std::cout << "Error in 1st derivative of g: got\n"
                  << Eigen::MatrixXd(df_def) << "\nbut expected\n"
                  << Eigen::MatrixXd(df_num) << '\n';
        success = false;
      };
    }
    if constexpr (diff::detail::diffable_order2<decltype(ocp.g), std::tuple<double, X, U>>) {
      const auto [f_def, df_def, d2f_def] = diff::dr<2, diff::Type::Analytic>(ocp.g, wrt(t, x, u));
      const auto [f_num, df_num, d2f_num] = diff::dr<2, DT>(ocp.g, wrt(t, x, u));
      if (!cmp(d2f_def, d2f_num)) {
        std::cout << "Error in 2nd derivative of g: got\n"
                  << Eigen::MatrixXd(d2f_def) << "\nbut expected\n"
                  << Eigen::MatrixXd(d2f_num) << '\n';
        success = false;
      };
    }

    // running constraints
    if constexpr (diff::detail::diffable_order1<decltype(ocp.cr), std::tuple<double, X, U>>) {
      const auto [f_def, df_def] = diff::dr<1, diff::Type::Analytic>(ocp.cr, wrt(t, x, u));
      const auto [f_num, df_num] = diff::dr<1, DT>(ocp.cr, wrt(t, x, u));
      if (!cmp(df_def, df_num)) {
        std::cout << "Error in 1st derivative of cr: got\n"
                  << Eigen::MatrixXd(df_def) << "\nbut expected\n"
                  << Eigen::MatrixXd(df_num) << '\n';
        success = false;
      };
    }
    if constexpr (diff::detail::diffable_order2<decltype(ocp.cr), std::tuple<double, X, U>>) {
      const auto [f_def, df_def, d2f_def] = diff::dr<2, diff::Type::Analytic>(ocp.cr, wrt(t, x, u));
      const auto [f_num, df_num, d2f_num] = diff::dr<2, DT>(ocp.cr, wrt(t, x, u));
      if (!cmp(d2f_def, d2f_num)) {
        std::cout << "Error in 2nd derivative of cr: got\n"
                  << Eigen::MatrixXd(d2f_def) << "\nbut expected\n"
                  << Eigen::MatrixXd(d2f_num) << '\n';
        success = false;
      };
    }
  }

  return success;
}

}  // namespace smooth::feedback
