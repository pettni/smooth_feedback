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

#ifndef SMOOTH__FEEDBACK__COLLOCATION_HPP_
#define SMOOTH__FEEDBACK__COLLOCATION_HPP_

/**
 * TODO list
 *
 * # Mesh refinement policy
 *   - Read literature
 *   - Convergence criteria
 *   - Rules for polynomial degrees
 *   - Easy way to evaluate functions on mesh to carry warmstart to finer mesh
 * # Calculate jacobians on demand
 * # Calculate Hessian function..
 * # Problem scaling
 * # Generalize to endpoint constraints b(t0, x0, tf, xf, q)
 * # Add parameters s?
 */

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <smooth/polynomial/quadrature.hpp>

#include <cstddef>
#include <numeric>
#include <ranges>
#include <vector>

#include "utils/sparse.hpp"

namespace smooth::feedback {

namespace detail {

/**
 * @brief Legendre-Gauss-Radau nodes including an extra node at +1.
 */
template<std::size_t K, std::size_t I = 8>
  requires(K <= 40)
constexpr std::pair<std::array<double, K + 1>, std::array<double, K + 1>> lgr_plus_one()
{
  auto lgr_norm = ::smooth::lgr_nodes<K, I>();

  std::array<double, K + 1> ns, ws;
  for (auto i = 0u; i < K; ++i) {
    ns[i] = lgr_norm.first[i];
    ws[i] = lgr_norm.second[i];
  }
  ns[K] = 1;
  ws[K] = 0;
  return {ns, ws};
}

}  // namespace detail

/**
 * @brief Collocation mesh of interval [0, 1].
 *
 * [0, 1] is divided into non-overlapping intervals I_i, and each interval I_i has K_i LGR
 * collocation points.
 */
class Mesh
{
public:
  /**
   * @brief Create a mesh consisting of a single interval [0, 1].
   */
  inline Mesh() { intervals_ = {Interval{.K = 5, .tau0 = 0}}; }

  /**
   * @brief Number of intervals in mesh.
   */
  inline std::size_t N_ivals() const { return intervals_.size(); }

  /**
   * @brief Number of collocation points in mesh.
   */
  inline std::size_t N_colloc() const
  {
    return std::accumulate(
      intervals_.begin(), intervals_.end(), 0u, [](std::size_t curr, const auto & x) {
        return curr + x.K;
      });
  }

  /**
   * @brief Number of collocation points in interval i.
   *
   * @note This is also equal to the polynomial degree inside interval i, since the polynomial is
   * fitted with an "extra" point belonging to the subsequent interval.
   */
  inline std::size_t N_colloc_ival(std::size_t i) const { return intervals_[i].K; }

  /**
   * @breif Split interval i into n equal parts.
   */
  inline void split(std::size_t i, std::size_t n = 2)
  {
    const double tau0 = intervals_[i].tau0;
    const double tauf = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;
    const double taum = tau0 + (tauf - tau0) / n;

    while (n-- > 1) {
      intervals_.insert(intervals_.begin() + i + 1, Interval{.K = 5, .tau0 = n * taum});
    }
  }

  /**
   * @brief Split all intervals.
   */
  inline void split_all(std::size_t n = 2)
  {
    const std::size_t S0 = intervals_.size();
    for (auto i = 0u; i < S0; ++i) { split(S0 - 1 - i, n); }
  }

  /**
   * @brief Interval nodes and weights (DOES include extra point)
   */
  inline std::pair<Eigen::VectorXd, Eigen::VectorXd> interval_nodes_and_weights(std::size_t i) const
  {
    const std::size_t K = intervals_[i].K;

    const double tau0 = intervals_[i].tau0;
    const double tauf = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;

    Eigen::VectorXd n_n;
    Eigen::VectorXd w_n;

    if (K == 5) {
      constexpr auto Ks   = 5u;
      constexpr auto nw_s = detail::lgr_plus_one<Ks>();

      n_n = Eigen::Map<const Eigen::VectorXd>(nw_s.first.data(), Ks + 1);
      w_n = Eigen::Map<const Eigen::VectorXd>(nw_s.second.data(), Ks + 1);
    } else {
      throw std::runtime_error("Size not available");
    }

    return {
      Eigen::VectorXd::Constant(n_n.size(), tau0)
        + (tauf - tau0) * (n_n + Eigen::VectorXd::Constant(n_n.size(), 1)) / 2,
      ((tauf - tau0) / 2.) * w_n,
    };
  }

  /**
   * @brief All nodes (DOES include extra point)
   */
  inline std::pair<Eigen::VectorXd, Eigen::VectorXd> all_nodes_and_weights() const
  {
    Eigen::VectorXd n(N_colloc() + 1), w(N_colloc() + 1);

    std::size_t cntr = 0;
    for (auto i = 0u; i < intervals_.size(); ++i) {
      auto [ni, wi] = interval_nodes_and_weights(i);

      const std::size_t Ni = ni.size();

      // exclude last point that belongs to next interval..
      n.segment(cntr, Ni - 1) = ni.head(Ni - 1);
      w.segment(cntr, Ni - 1) = wi.head(Ni - 1);

      cntr += Ni - 1;
    }

    n.tail(1).setConstant(1);
    w.tail(1).setConstant(0);

    return {n, w};
  }

  /**
   * @brief Interval differentiation matrix w.r.t. [0, 1] timescale.
   *
   * Matrix K+1 x K matrix D s.t.
   * \f[
   *   \begin{bmatrix} y'(\tau_{i, 0}) & y'(\tau_{i, 1}) & \cdots & y'(\tau_{i, K-1}) \end{bmatrix}
   *  =
   *   \begin{bmatrix} y_{i, 0} & y_{i, 1} & \cdots & y_{i, K} \end{bmatrix} D
   * \f],
   * where \f$ y_{i, j} \in \mathbb{R}^{d \times 1} \f$ is the function value at scaled time point
   * \f$ \tau_{i, j} \f$.
   */
  inline Eigen::MatrixXd interval_diffmat(std::size_t i) const
  {
    const std::size_t K = intervals_[i].K;

    const double tau0 = intervals_[i].tau0;
    const double tauf = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;

    if (K == 5) {
      constexpr auto Ks   = 5u;
      constexpr auto nw_s = detail::lgr_plus_one<Ks>();
      constexpr auto B    = lagrange_basis<Ks>(nw_s.first);
      constexpr auto D    = polynomial_basis_derivatives<Ks, Ks + 1>(B, nw_s.first);

      Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> m(
        D[0].data(), Ks + 1, Ks + 1);
      return (2 / (tauf - tau0)) * m.block(0, 0, Ks + 1, Ks);
    } else {
      throw std::runtime_error("Wrong size");
    }
  }

private:
  struct Interval
  {
    /// @brief Polynomial degree in interval
    std::size_t K;
    /// @brief Start of interval on [0, 1] timescale
    double tau0;
  };

  /// @brief Mesh intervals
  std::vector<Interval> intervals_;
};

/**
 * @brief Evaluate a function on all collocation points.
 *
 * Returns a nf x N matrix
 *
 *  F= [ f(t_0, X_0, U_0)  f(t_1, X_1, u_1) ... f(t_{N-1}, X_{N-1}, U_{N-1})]
 *
 * with the function evaluated at all collocation points t_i in the Mesh m.
 *
 * @param nf dimensionality of f image
 * @param f function (t, X, U) -> R^nf
 * @param m Mesh of time
 * @param t0 initial time
 * @param tf final time
 * @param X state variables (size nx x N+1)
 * @param U input variables (size nu x N)
 *
 * @return {F, dvecF_dt0, dvecF_dtf, dvecF_dvecX, dvecF_dvecU},
 * where vec(X) stacks the columns of X into a single column vector.
 */
template<typename Fun>
auto colloc_eval(std::size_t nf,
  Fun && f,
  const Mesh & m,
  const double t0,
  const double tf,
  const Eigen::MatrixXd & X,
  const Eigen::MatrixXd & U)
{
  assert(m.N_colloc() + 1 == static_cast<std::size_t>(X.cols()));  //  extra variable at the end
  assert(m.N_colloc() == static_cast<std::size_t>(U.cols()));  // one input per collocation point

  const std::size_t nx = X.rows();
  const std::size_t nu = U.rows();

  // all nodes in mesh
  const auto [tau_s, w_s] = m.all_nodes_and_weights();

  Eigen::MatrixXd Fval(nf, tau_s.size() - 1);

  Eigen::SparseMatrix<double> dvecF_dt0(Fval.size(), 1);
  Eigen::SparseMatrix<double> dvecF_dtf(Fval.size(), 1);
  Eigen::SparseMatrix<double> dvecF_dvecX(Fval.size(), X.size());
  Eigen::SparseMatrix<double> dvecF_dvecU(Fval.size(), U.size());

  Eigen::Matrix<int, -1, 1> FX_pattern = Eigen::Matrix<int, -1, 1>::Constant(X.size(), nf);
  FX_pattern.tail(nx).setZero();
  dvecF_dvecX.reserve(FX_pattern);
  dvecF_dvecU.reserve(Eigen::Matrix<int, -1, 1>::Constant(U.size(), nf));

  Eigen::SparseMatrix<double> d2lF_dvecX2(X.size(), X.size());

  // TODO: calculate Hessian on demand

  for (auto i = 0u; i + 1 < tau_s.size(); ++i) {
    autodiff::VectorXreal var_ad(1 + nx + nu);
    var_ad << autodiff::real(t0 + (tf - t0) * tau_s(i)), X.col(i).template cast<autodiff::real>(),
      U.col(i).template cast<autodiff::real>();

    autodiff::VectorXreal Fval_ad;
    const Eigen::MatrixXd dF = autodiff::jacobian(
      [&](const autodiff::VectorXreal & var) -> autodiff::VectorXreal {
        return f.template operator()<autodiff::real>(
          var(0), var.segment(1, nx), var.segment(1 + nx, nu));
      },
      autodiff::wrt(var_ad),
      autodiff::at(var_ad),
      Fval_ad);

    assert(Fval_ad.rows() == Eigen::Index(nf));
    assert(dF.rows() == Eigen::Index(nf));
    assert(dF.cols() == Eigen::Index(1 + nu + nx));

    Fval.col(i) = Fval_ad.template cast<double>();

    for (auto row = 0u; row < nf; ++row) {
      dvecF_dt0.insert(nf * i + row, 0) = dF(row, 0) * (1. - tau_s(i));
      dvecF_dtf.insert(nf * i + row, 0) = dF(row, 0) * tau_s(i);
      for (auto col = 0u; col < nx; ++col) {
        dvecF_dvecX.insert(nf * i + row, i * nx + col) = dF(row, 1 + col);
      }
      for (auto col = 0u; col < nu; ++col) {
        dvecF_dvecU.insert(nf * i + row, i * nu + col) = dF(row, 1 + nx + col);
      }
    }
  }

  dvecF_dt0.makeCompressed();
  dvecF_dtf.makeCompressed();
  dvecF_dvecX.makeCompressed();
  dvecF_dvecU.makeCompressed();

  return std::make_tuple(std::move(Fval),
    std::move(dvecF_dt0),
    std::move(dvecF_dtf),
    std::move(dvecF_dvecX),
    std::move(dvecF_dvecU));
}

/**
 * @brief Evaluate dynamics constraint in all collocation points of a Mesh.
 *
 * @param f right-hand side of dynamics with signature (t, x, u) -> dx where x and dx are size nx
 * x 1 and u is size nu x 1
 * @param m mesh with a total of N collocation points
 * @param tf final time (variable of size 1)
 * @param x state values (variable of size nx x N+1)
 * @param u input values (variable of size nu x N)
 *
 * @return {F, dvecF_dt0, dvecF_dtf, dvecF_dvecX, dvecF_dvecU},
 * where vec(X) stacks the columns of X into a single column vector.
 */
template<typename F>
auto dynamics_constraint(F && f,
  const Mesh & m,
  const double t0,
  const double tf,
  const Eigen::MatrixXd & X,
  const Eigen::MatrixXd & U)
{
  assert(m.N_colloc() + 1 == static_cast<std::size_t>(X.cols()));  // extra at the end
  assert(m.N_colloc() == static_cast<std::size_t>(U.cols()));      // one per collocation point

  const std::size_t nx = X.rows();

  const auto [Fval, dvecF_dt0, dvecF_dtf, dFvec_dvecX, dvecF_dvecU] =
    colloc_eval(nx, std::forward<F>(f), m, t0, tf, X, U);

  Eigen::MatrixXd XD(nx, m.N_colloc());
  Eigen::SparseMatrix<double> dvecXD_dvecX(XD.size(), X.size());

  // reserve sparsity pattern
  Eigen::Matrix<int, -1, 1> pattern = Eigen::Matrix<int, -1, 1>::Zero(X.size());
  for (auto M = 0u, i = 0u; i < m.N_ivals(); ++i) {
    const std::size_t K = m.N_colloc_ival(i);
    pattern.segment(M, (K + 1) * nx) += Eigen::Matrix<int, -1, 1>::Constant((K + 1) * nx, K);
    M += K * nx;
  }
  dvecXD_dvecX.reserve(pattern);

  for (auto i = 0u, M = 0u; i < m.N_ivals(); M += m.N_colloc_ival(i), ++i) {
    const std::size_t K     = m.N_colloc_ival(i);
    const Eigen::MatrixXd D = m.interval_diffmat(i);
    XD.block(0, M, nx, K)   = X.block(0, M, nx, K + 1) * D;

    // vec(X * D) = kron(D', I) * vec(X), so derivative w.r.t vec(X) = kron(D', I)
    for (auto i = 0u; i < K; ++i) {
      for (auto j = 0u; j < K + 1; ++j) {
        for (auto diag = 0u; diag < nx; ++diag) {
          dvecXD_dvecX.coeffRef(M * nx + i * nx + diag, M * nx + j * nx + diag) += D(j, i);
        }
      }
    }
  }

  dvecXD_dvecX.makeCompressed();

  Eigen::VectorXd Fv = (XD - (tf - t0) * Fval).reshaped();

  Eigen::SparseMatrix<double> dF_dt0 = -(tf - t0) * dvecF_dt0;
  dF_dt0 += Fval.reshaped().sparseView();  // OK since dvecF_dtf is dense

  Eigen::SparseMatrix<double> dF_dtf = -(tf - t0) * dvecF_dtf;
  dF_dtf -= Fval.reshaped().sparseView();  // OK since dvecF_dtf is dense

  Eigen::SparseMatrix<double> dF_dvecX = dvecXD_dvecX;
  dF_dvecX -= (tf - t0) * dFvec_dvecX;

  Eigen::SparseMatrix<double> dF_dvecU = -(tf - t0) * dvecF_dvecU;

  dF_dt0.makeCompressed();
  dF_dtf.makeCompressed();
  dF_dvecX.makeCompressed();
  dF_dvecU.makeCompressed();

  return std::make_tuple(
    std::move(Fv), std::move(dF_dt0), std::move(dF_dtf), std::move(dF_dvecX), std::move(dF_dvecU));
}

/**
 * @brief Evaluate integral constraint on Mesh.
 *
 * @param g integrand with signature (t, x, u) -> double where x is size nx x 1 and u is size nu x
 * 1
 * @param m mesh
 * @param t0 initial time (variable of size 1)
 * @param tf final time (variable of size 1)
 * @param I values (variable of size 1)
 * @param X state values (variable of size nx x N+1)
 * @param U input values (variable of size nu x N)
 *
 * @return {G, dvecG_dt0, dvecG_dtf, dvecG_dvecX, dvecG_dvecU},
 * where vec(X) stacks the columns of X into a single column vector.
 */
template<typename G>
auto integral_constraint(G && g,
  const Mesh & m,
  const double & t0,
  const double & tf,
  const Eigen::MatrixXd & I,
  const Eigen::MatrixXd & X,
  const Eigen::MatrixXd & U)
{
  static constexpr std::size_t mi = 1;
  const std::size_t N             = m.N_colloc();

  const auto [Gv, dvecG_dt0, dvecG_dtf, dvecG_dvecX, dvecG_dvecU] =
    colloc_eval(mi, std::forward<G>(g), m, t0, tf, X, U);

  const auto [n, w]          = m.all_nodes_and_weights();
  const Eigen::VectorXd Iest = Gv * w.head(N);

  const Eigen::SparseMatrix<double> w_kron_I = (tf - t0) * kron_identity(w.head(N).transpose(), mi);

  Eigen::VectorXd Rv = (tf - t0) * Iest - I;

  Eigen::SparseMatrix<double> dR_dt0 = w_kron_I * dvecG_dt0;
  for (auto i = 0u; i < Iest.size(); ++i) { dR_dt0.coeffRef(i, 0) -= Iest(i); }

  Eigen::SparseMatrix<double> dR_dtf = w_kron_I * dvecG_dtf;
  for (auto i = 0u; i < Iest.size(); ++i) { dR_dtf.coeffRef(i, 0) += Iest(i); }

  Eigen::SparseMatrix<double> dR_dvecI = -sparse_identity(mi);

  Eigen::SparseMatrix<double> dR_dvecX = w_kron_I * dvecG_dvecX;

  Eigen::SparseMatrix<double> dR_dvecU = w_kron_I * dvecG_dvecU;

  dR_dt0.makeCompressed();
  dR_dtf.makeCompressed();
  dR_dvecI.makeCompressed();
  dR_dvecX.makeCompressed();
  dR_dvecU.makeCompressed();

  return std::make_tuple(std::move(Rv),
    std::move(dR_dt0),
    std::move(dR_dtf),
    std::move(dR_dvecI),
    std::move(dR_dvecX),
    std::move(dR_dvecU));
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION_HPP_
