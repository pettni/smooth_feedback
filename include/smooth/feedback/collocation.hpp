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
 *   - Convergence criteria
 *   - Rules for polynomial degrees
 *   - Easy way to evaluate functions on mesh to carry warmstart to finer mesh
 * # Calculate jacobians on demand
 * # Calculate Hessian function..
 * # Problem scaling
 * # Add parameters s?
 */

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <smooth/diff.hpp>
#include <smooth/internal/utils.hpp>
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
template<std::size_t Kmin = 5, std::size_t Kmax = 5>
  requires(Kmax >= Kmin)
class Mesh
{
private:
public:
  /**
   * @brief Create a mesh consisting of a single interval [0, 1].
   */
  inline Mesh()
  {
    // store compile-time objects
    utils::static_for<Kmax + 1 - Kmin>([this](auto i) {
      constexpr auto K    = Kmin + i;
      constexpr auto nw_s = detail::lgr_plus_one<K>();
      constexpr auto B_s  = lagrange_basis<K>(nw_s.first);
      constexpr auto D_s  = polynomial_basis_derivatives<K, K + 1>(B_s, nw_s.first);

      Eigen::VectorXd n = Eigen::Map<const Eigen::VectorXd>(nw_s.first.data(), K + 1);
      Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(nw_s.second.data(), K + 1);
      Eigen::MatrixXd D = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
        D_s[0].data(), K + 1, K + 1);

      ns_.push_back(std::move(n));
      ws_.push_back(std::move(w));
      Ds_.push_back(std::move(D));
    });

    // initialize with a single interval
    intervals_ = {Interval{.K = Kmin, .tau0 = 0}};
  }

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
   * @breif Refine interval using the ph strategy.
   *
   * @param i index of interval to refine
   * @param D target number of collocation points in refined interval
   *
   * If D < current degree, then nothing is done.
   * If D <= Kmax,          then the polynomial degree is increased to D.
   * If D > Kmax,           then the interval is divided into n = max(2, ceil(D / Kmin))
   *                        intervals with degree Kmin
   */
  inline void refine_ph(std::size_t i, std::size_t D)
  {
    if (D < intervals_[i].K) {
      return;
    } else if (D <= Kmax) {
      intervals_[i].K = D;
    } else {
      std::size_t n = std::max<std::size_t>(2u, (D + Kmin - 1) / Kmin);

      const double tau0 = intervals_[i].tau0;
      const double tauf = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;
      const double taum = (tauf - tau0) / n;

      while (n-- > 1) {
        intervals_.insert(intervals_.begin() + i + 1, Interval{.K = 5, .tau0 = tau0 + n * taum});
      }
    }
  }

  /**
   * @brief Interval nodes and quadrature weights (DOES include extra point)
   */
  inline std::pair<Eigen::VectorXd, Eigen::VectorXd> interval_nodes_and_weights(std::size_t i) const
  {
    const std::size_t K = intervals_[i].K;

    const double tau0 = intervals_[i].tau0;
    const double tauf = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;

    return {
      Eigen::VectorXd::Constant(ns_[K - Kmin].size(), tau0)
        + (tauf - tau0) * (ns_[K - Kmin] + Eigen::VectorXd::Constant(ns_[K - Kmin].size(), 1)) / 2,
      ((tauf - tau0) / 2.) * ws_[K - Kmin],
    };
  }

  /**
   * @brief All Mesh nodes and quadrature weights (DOES include extra point)
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

    return (2 / (tauf - tau0)) * Ds_[K - Kmin].block(0, 0, K + 1, K);
  }

private:
  // nodes (plus one)
  std::vector<Eigen::VectorXd> ns_;
  // weights (plus one)
  std::vector<Eigen::VectorXd> ws_;
  // differentiation matrix (plus one)
  std::vector<Eigen::MatrixXd> Ds_;

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
template<typename Fun, std::size_t Kmin, std::size_t Kmax>
auto colloc_eval(std::size_t nf,
  Fun && f,
  const Mesh<Kmin, Kmax> & m,
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

  for (auto i = 0u; i + 1 < tau_s.size(); ++i) {
    const double T = t0 + (tf - t0) * tau_s(i);

    const Eigen::VectorXd x = X.col(i);
    const Eigen::VectorXd u = U.col(i);

    const auto [fval, dfval] = diff::dr(f, wrt(T, x, u));

    assert(fval.rows() == Eigen::Index(nf));
    assert(dfval.rows() == Eigen::Index(nf));
    assert(dfval.cols() == Eigen::Index(1 + nu + nx));

    Fval.col(i) = fval;

    for (auto row = 0u; row < nf; ++row) {
      dvecF_dt0.insert(nf * i + row, 0) = dfval(row, 0) * (1. - tau_s(i));
      dvecF_dtf.insert(nf * i + row, 0) = dfval(row, 0) * tau_s(i);
      for (auto col = 0u; col < nx; ++col) {
        dvecF_dvecX.insert(nf * i + row, i * nx + col) = dfval(row, 1 + col);
      }
      for (auto col = 0u; col < nu; ++col) {
        dvecF_dvecU.insert(nf * i + row, i * nu + col) = dfval(row, 1 + nx + col);
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
 * @brief Evaluate a function at endpoints.
 *
 * Returns a nf vector
 *
 *  f(t_0, t_f, x_0, x_f)
 *
 * @param nf dimensionality of f image
 * @param nf state space degrees of freedom
 * @param f function (t0, tf, x0, xf, q) -> R^nf
 * @param t0 initial time
 * @param tf final time
 * @param X state variables (size nx x N+1)
 * @param q integrals
 *
 * @return {F, dF_dt0, dF_dtf, dF_dvecX},
 */
template<typename Fun>
auto endpoint_eval(std::size_t nf,
  std::size_t nx,
  Fun && f,
  const double t0,
  const double tf,
  const Eigen::MatrixXd & X,
  const Eigen::VectorXd & q)
{
  assert(static_cast<std::size_t>(X.rows()) == nx);

  const Eigen::VectorXd x0 = X.leftCols(1);
  const Eigen::VectorXd xf = X.rightCols(1);

  const auto [Fval, J] = diff::dr(f, wrt(t0, tf, x0, xf, q));

  assert(static_cast<std::size_t>(Fval.rows()) == nf);
  assert(static_cast<std::size_t>(J.rows()) == nf);
  assert(static_cast<std::size_t>(J.cols()) == 2 + 2 * nx + q.size());

  Eigen::SparseMatrix<double> dF_dt0(nf, 1);
  dF_dt0.reserve(nf);
  for (auto i = 0u; i < nf; ++i) { dF_dt0.insert(i, 0) = J(i, 0); }

  Eigen::SparseMatrix<double> dF_dtf(nf, 1);
  dF_dtf.reserve(nf);
  for (auto i = 0u; i < nf; ++i) { dF_dtf.insert(i, 0) = J(i, 1); }

  Eigen::SparseMatrix<double> dF_dvecX(nf, X.size());
  Eigen::VectorXi pattern(X.size());
  pattern.head(nx).setConstant(nf);
  pattern.tail(nx).setConstant(nf);
  dF_dvecX.reserve(pattern);

  for (auto row = 0u; row < nf; ++row) {
    for (auto col = 0u; col < nx; ++col) {
      dF_dvecX.insert(row, col)                 = J(row, 2 + col);
      dF_dvecX.insert(row, X.size() - nx + col) = J(row, 2 + nx + col);
    }
  }

  Eigen::SparseMatrix<double> dF_dQ(nf, q.size());
  dF_dQ.reserve(Eigen::VectorXi::Constant(q.size(), nf));

  for (auto row = 0u; row < nf; ++row) {
    for (auto col = 0u; col < q.size(); ++col) {
      dF_dQ.insert(row, col) = J(row, 2 + 2 * nx + col);
    }
  }

  dF_dt0.makeCompressed();
  dF_dtf.makeCompressed();
  dF_dvecX.makeCompressed();
  dF_dQ.makeCompressed();

  return std::make_tuple(Fval, dF_dt0, dF_dtf, dF_dvecX, dF_dQ);
}

/**
 * @brief Evaluate dynamics constraint in all collocation points of a Mesh.
 *
 * @param nx state space degrees of freedom
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
template<typename F, std::size_t Kmin, std::size_t Kmax>
auto dynamics_constraint(std::size_t nx,
  F && f,
  const Mesh<Kmin, Kmax> & m,
  const double t0,
  const double tf,
  const Eigen::MatrixXd & X,
  const Eigen::MatrixXd & U)
{
  assert(m.N_colloc() + 1 == static_cast<std::size_t>(X.cols()));  // extra at the end
  assert(m.N_colloc() == static_cast<std::size_t>(U.cols()));      // one per collocation point
  assert(nx == static_cast<std::size_t>(X.rows()));                // one per collocation point

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
 * @param nq number of integrals
 * @param g integrand with signature (t, x, u) -> R^{nq} where x is size nx x 1 and u is size nu x
 * 1
 * @param m mesh
 * @param t0 initial time (variable of size 1)
 * @param tf final time (variable of size 1)
 * @param I values (variable of size nq)
 * @param X state values (variable of size nx x N+1)
 * @param U input values (variable of size nu x N)
 *
 * @return {G, dvecG_dt0, dvecG_dtf, dvecG_dvecX, dvecG_dvecU},
 * where vec(X) stacks the columns of X into a single column vector.
 */
template<typename G, std::size_t Kmin, std::size_t Kmax>
auto integral_constraint(std::size_t nq,
  G && g,
  const Mesh<Kmin, Kmax> & m,
  const double & t0,
  const double & tf,
  const Eigen::VectorXd & I,
  const Eigen::MatrixXd & X,
  const Eigen::MatrixXd & U)
{
  assert(static_cast<std::size_t>(I.size()) == nq);

  const std::size_t N = m.N_colloc();

  const auto [Gv, dvecG_dt0, dvecG_dtf, dvecG_dvecX, dvecG_dvecU] =
    colloc_eval(nq, std::forward<G>(g), m, t0, tf, X, U);

  const auto [n, w]          = m.all_nodes_and_weights();
  const Eigen::VectorXd Iest = Gv * w.head(N);

  const Eigen::SparseMatrix<double> w_kron_I = (tf - t0) * kron_identity(w.head(N).transpose(), nq);

  Eigen::VectorXd Rv = (tf - t0) * Iest - I;

  Eigen::SparseMatrix<double> dR_dt0 = w_kron_I * dvecG_dt0;
  for (auto i = 0u; i < Iest.size(); ++i) { dR_dt0.coeffRef(i, 0) -= Iest(i); }

  Eigen::SparseMatrix<double> dR_dtf = w_kron_I * dvecG_dtf;
  for (auto i = 0u; i < Iest.size(); ++i) { dR_dtf.coeffRef(i, 0) += Iest(i); }

  Eigen::SparseMatrix<double> dR_dvecI = -sparse_identity(nq);

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
