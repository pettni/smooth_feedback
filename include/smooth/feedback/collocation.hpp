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

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <smooth/diff.hpp>
#include <smooth/internal/utils.hpp>
#include <smooth/polynomial/quadrature.hpp>

#include <cstddef>
#include <numeric>
#include <ranges>
#include <vector>

#include "traits.hpp"
#include "utils/sparse.hpp"

namespace smooth::feedback {

namespace detail {

/**
 * @brief Legendre-Gauss-Radau nodes including an extra node at +1.
 */
template<std::size_t K, std::size_t I = 8>
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
 * @tparam _Kmin minimal number of collocation points per interval
 * @tparam _Kmax maximal number of collocation points per interval
 *
 * [0, 1] is divided into non-overlapping intervals I_i, and each interval I_i has K_i LGR
 * collocation points.
 */
template<std::size_t _Kmin = 5, std::size_t _Kmax = 10>
  requires(_Kmin <= _Kmax)
class Mesh
{
  using MatMap = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>;

public:
  /// @brief Minimal number of collocation points per interval
  static constexpr auto Kmin = _Kmin;
  /// @brief Maximal number of collocation points per interval
  static constexpr auto Kmax = _Kmax;

  /**
   * @brief Create a mesh consisting of a single interval [0, 1].
   *
   * @param Kmin minimal polynomial degree in mesh
   * @param Kmax maximal polynomial degree in mesh
   *
   * @note It must hold that kKmin <= Kmin <= Kmax <= kKmax, where kKmin and kKmax are compile-time
   * constants that define which LGR nodes to pre-compute.
   */
  inline Mesh() : intervals_(1, Interval{.K = Kmin, .tau0 = 0}) {}

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
   * If D > Kmax, or current degree > Kmax    then the interval is divided into
   *                                          n = max(2, ceil(D / Kmin)) intervals with deg Kmin
   * If D < current degree,                   then nothing is done.
   * If D <= Kmax,                            then the polynomial degree is increased to D.
   */
  inline void refine_ph(std::size_t i, std::size_t D)
  {
    if (D > Kmax || intervals_[i].K > Kmax) {
      // refine by splitting interval into n intervals, each with degree Kmin_
      std::size_t n = std::max<std::size_t>(2u, (D + Kmin - 1) / Kmin);

      const double tau0 = intervals_[i].tau0;
      const double tauf = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;
      const double taum = (tauf - tau0) / n;

      while (n-- > 1) {
        intervals_.insert(intervals_.begin() + i + 1, Interval{.K = Kmin, .tau0 = tau0 + n * taum});
      }
    } else if (D < intervals_[i].K) {
      return;
    } else if (D <= Kmax) {
      // refine by increasing degree in interval
      intervals_[i].K = D;
    }
  }

  /**
   * @brief Set the number of collocation points in interval i to K
   * @param i interval index
   * @param K number of collocation points s.t. (Kmin <= K <= Kmax + 1)
   */
  inline void set_N_colloc_ival(std::size_t i, std::size_t K)
  {
    assert(Kmin <= K);
    assert(K <= Kmax + 1);
    intervals_[i].K = K;
  }

  /**
   * @brief Interval nodes and quadrature weights.
   *
   * @note Includes extra point at 1, i.e. size of returned arrays is equal to N_colloc_ival(i)+1
   */
  inline std::pair<Eigen::VectorXd, Eigen::VectorXd> interval_nodes_and_weights(std::size_t i) const
  {
    const std::size_t k = intervals_[i].K;

    Eigen::VectorXd ns, ws;
    utils::static_for<Kmax + 2 - Kmin>([&](auto i) {
      static constexpr auto K = Kmin + i;
      if (K == k) {
        static constexpr auto nw_ext_s = detail::lgr_plus_one<K>();
        ns = Eigen::Map<const Eigen::VectorXd>(nw_ext_s.first.data(), k + 1);
        ws = Eigen::Map<const Eigen::VectorXd>(nw_ext_s.second.data(), k + 1);
      }
    });

    const double tau0  = intervals_[i].tau0;
    const double tauf  = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;
    const double alpha = (tauf - tau0) / 2;

    return {
      Eigen::VectorXd::Constant(ns.size(), tau0) + alpha * (ns + Eigen::VectorXd::Ones(ns.size())),
      alpha * ws,
    };
  }

  /**
   * @brief All Mesh nodes and quadrature weights.
   *
   * @note Includes extra point at 1, i.e. size of returned arrays is equal to N_colloc()+1
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
   * Returns a \f$ (K+1 \times K) \f$ matrix \f$ D \f$ s.t.
   * \f[
   *   \begin{bmatrix} y'(\tau_{i, 0}) & y'(\tau_{i, 1}) & \cdots & y'(\tau_{i, K-1}) \end{bmatrix}
   *  =
   *   \begin{bmatrix} y(\tau_{i, 0}) & y(\tau_{i, 1}) & \cdots & y(\tau_{i, K}) \end{bmatrix} D
   * \f],
   * where \f$ y(\cdot) \in \mathbb{R}^{d \times 1} \f$ is a Lagrange polynomial in interval i.
   */
  inline Eigen::MatrixXd interval_diffmat(std::size_t i) const
  {
    const std::size_t k = intervals_[i].K;

    const double tau0 = intervals_[i].tau0;
    const double tauf = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;

    Eigen::MatrixXd ret;
    utils::static_for<Kmax + 2 - Kmin>([&](auto i) {
      static constexpr auto K = Kmin + i;
      if (K == k) {
        static constexpr auto nw_ext_s = detail::lgr_plus_one<K>();
        static constexpr auto B_ext_s  = lagrange_basis<K>(nw_ext_s.first);
        static constexpr auto D_ext_s =
          polynomial_basis_derivatives<K, K + 1>(B_ext_s, nw_ext_s.first)
            .template block<K + 1, K>(0, 0);
        ret = MatMap(D_ext_s[0].data(), k + 1, k);
        ;
      }
    });

    return (2. / (tauf - tau0)) * ret;
  }

  /**
   * @brief Interval integration matrix w.r.t. [0, 1] timescale.
   *
   * Returns a \f$ (K \times K) \f$ matrix \f$ I \f$ s.t.
   * \f[
   *   \begin{bmatrix}
   *      y(\tau_{i, 1}) & y(\tau_{i, 2}) & \cdots & y(\tau_{i, K})
   *   \end{bmatrix}
   *  = y(\tau_{i, 0}) \begin{bmatrix} 1 & \ldots & 1 \end{bmatrix}
   *    + \begin{bmatrix}
   *        \dot y(\tau_{i, 0}) & \dot y(\tau_{i, 1}) & \cdots & \dot y(\tau_{i, K-1})
   *      \end{bmatrix} I
   * \f],
   * where \f$ y(\cdot) \in \mathbb{R}^{d \times 1} \f$ is a Lagrange
   * polynomial in interval i.
   */
  inline Eigen::MatrixXd interval_intmat(std::size_t i) const
  {
    const std::size_t k = intervals_[i].K;
    return interval_diffmat(i).block(1, 0, k, k).inverse();
  }

  /**
   * @brief Find interval index that contains t
   */
  inline std::size_t interval_find(double t) const
  {
    if (t < 0) { return 0; }
    if (t > 1) { return intervals_.size() - 1; }
    auto it = utils::binary_interval_search(
      intervals_, t, [](const auto & ival, double _t) { return ival.tau0 <=> _t; });
    if (it != intervals_.end()) { return std::distance(intervals_.begin(), it); }
    return 0;
  }

  /**
   * @brief Evaluate a function
   *
   * @tparam RetT return value type
   *
   * @param t time value in [0, 1]
   * @param r values for the collocation points (size N [extend=false] or N+1 [extend=true])
   * @param derivative to evaluate
   * @param extend set to true if a value is provided for t=+1
   */
  template<typename RetT, std::ranges::sized_range R>
  RetT eval(double t, const R & r, std::size_t p = 0, bool extend = true) const
  {
    [[maybe_unused]] const std::size_t N = N_colloc();

    if (extend) {
      assert(std::ranges::size(r) == N + 1);
    } else {
      assert(std::ranges::size(r) == N);
    }

    const std::size_t ival = interval_find(t);
    const std::size_t k    = intervals_[ival].K;

    const double tau0 = intervals_[ival].tau0;
    const double tauf = ival + 1 < intervals_.size() ? intervals_[ival + 1].tau0 : 1.;

    const double u = 2 * (t - tau0) / (tauf - tau0) - 1;

    Eigen::RowVectorXd W;

    utils::static_for<Kmax + 2 - Kmin>([&](auto i) {
      static constexpr auto K = Kmin + i;
      if (K == k) {
        if (extend || ival + 1 < intervals_.size()) {
          static constexpr auto nw_ext_s = detail::lgr_plus_one<K>();
          static constexpr auto B_ext_s  = lagrange_basis<K>(nw_ext_s.first);
          const auto U                   = monomial_derivative<K>(u, p);
          W = MatMap(U[0].data(), 1, k + 1) * MatMap(B_ext_s[0].data(), k + 1, k + 1);
          assert(std::size_t(W.size()) == k + 1);
        } else {
          static constexpr auto nw_s = lgr_nodes<K>();
          static constexpr auto B_s  = lagrange_basis<K - 1>(nw_s.first);
          const auto U               = monomial_derivative<K - 1>(u, p);
          W                          = MatMap(U[0].data(), 1, k) * MatMap(B_s[0].data(), k, k);
          assert(std::size_t(W.size()) == k);
        }
      }
    });

    using namespace std::views;

    std::size_t N_before = 0;
    for (auto i = 0u; i < ival; ++i) { N_before += intervals_[i].K; }
    const auto r_ival = r | drop(int64_t(N_before));
    RetT ret          = W(0) * *std::ranges::begin(r_ival);

    for (auto i = 1u; const auto & v : r_ival | drop(1) | take(W.size() - 1)) { ret += W(i++) * v; }
    return ret;
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

/// @brief MeshType is a specialization of Mesh
template<typename T>
concept MeshType = traits::is_specialization_of_sizet_v<T, Mesh>;

/**
 * @brief Output structure for colloc_eval
 */
struct CollocEvalResult
{
  /**
   * @brief Construct object and allocate memory
   * @param nf dimensionality of result
   * @param nx state space dimension
   * @param nu input space dimension
   * @param N number of collocation points
   */
  inline CollocEvalResult(
    const std::size_t nf, const std::size_t nx, const std::size_t nu, const std::size_t N)
      : nf(nf), nx(nx), nu(nu), N(N)
  {
    allocate();
  }

  inline void allocate()
  {
    F.resize(nf, N);

    // dense column vector
    dvecF_dt0.resize(F.size(), 1);
    dvecF_dt0.reserve(Eigen::VectorXi::Constant(1, F.size()));

    // dense column vector
    dvecF_dtf.resize(F.size(), 1);
    dvecF_dt0.reserve(Eigen::VectorXi::Constant(1, F.size()));

    // block diagonal matrix (blocks have size nf x nx)
    dvecF_dvecX.resize(F.size(), nx * (N + 1));
    Eigen::VectorXi FX_pattern = Eigen::VectorXi::Constant(nx * (N + 1), nf);
    FX_pattern.tail(nx).setZero();
    dvecF_dvecX.reserve(FX_pattern);

    // block diagonal matrix (blocks have size nf x nu)
    dvecF_dvecU.resize(F.size(), nu * N);
    dvecF_dvecU.reserve(Eigen::VectorXi::Constant(nu * N, nf));

    dvecF_dt0.makeCompressed();
    dvecF_dtf.makeCompressed();
    dvecF_dvecX.makeCompressed();
    dvecF_dvecU.makeCompressed();
  }

  inline void setZero()
  {
    F.setZero();
    dvecF_dt0.setZero();
    dvecF_dtf.setZero();
    dvecF_dvecX.setZero();
    dvecF_dvecU.setZero();

    allocate();  // seems like setZero forgets zero pattern, so this might be necessary to
                 // re-establish it
  }

  std::size_t nf;
  std::size_t nx;
  std::size_t nu;
  std::size_t N;

  /// @brief Function values (size nf x N)
  Eigen::MatrixXd F;
  /// @brief Function derivatives w.r.t. t0 (size N*nf x 1)
  Eigen::SparseMatrix<double> dvecF_dt0;
  /// @brief Function derivatives w.r.t. tf (size N*nf x 1)
  Eigen::SparseMatrix<double> dvecF_dtf;
  /// @brief Function derivatives w.r.t. X (size N*nf x nx*(N+1))
  Eigen::SparseMatrix<double> dvecF_dvecX;
  /// @brief Function derivatives w.r.t. X (size N*nf x nu*N)
  Eigen::SparseMatrix<double> dvecF_dvecU;

  /// @brief Second derivatives (only valid if nf == 1) w.r.t. all variables (square matrix with
  /// side 2+nx*(N+1)+nu*N)
  Eigen::SparseMatrix<double> d2F_dt0tfXU;
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
 * @tparam Deriv differentiation order
 *
 * @param[out] result
 * @param[in] f function (t, x, u) -> R^nf
 * @param[in] m Mesh of time
 * @param[in] t0 initial time variable
 * @param[in] tf final time variable
 * @param[in] xs state variables (size N+1)
 * @param[in] us input variables (size N)
 */
template<uint8_t Deriv = 0>
void colloc_eval(
  CollocEvalResult & result,
  auto && f,
  const MeshType auto & m,
  const double t0,
  const double tf,
  std::ranges::sized_range auto && xs,
  std::ranges::sized_range auto && us)
{
  using X = PlainObject<std::ranges::range_value_t<decltype(xs)>>;
  using U = PlainObject<std::ranges::range_value_t<decltype(us)>>;

  const auto numX = std::ranges::size(xs);
  const auto numU = std::ranges::size(us);

  assert(m.N_colloc() + 1 == numX);  //  extra variable at the end
  assert(m.N_colloc() == numU);      // one input per collocation point

  result.setZero();

  const std::size_t nf = result.nf;
  const std::size_t nx = result.nx;
  const std::size_t nu = result.nu;

  const auto [tau_s, w_s] = m.all_nodes_and_weights();

  for (const auto & [ival, tau, x, u] : utils::zip(std::views::iota(0u), tau_s, xs, us)) {
    const double ti = t0 + (tf - t0) * tau;

    const X x_plain = x;
    const U u_plain = u;

    if constexpr (Deriv == 0u) {
      result.F.col(ival) = f(ti, x_plain, u_plain);
    } else if constexpr (Deriv == 1u) {
      const auto [fval, dfval] = diff::dr<1>(f, wrt(ti, x_plain, u_plain));
      result.F.col(ival)       = fval;

      for (auto row = 0u; row < result.nf; ++row) {
        result.dvecF_dt0.insert(nf * ival + row, 0) = dfval(row, 0) * (1. - tau);
        result.dvecF_dtf.insert(nf * ival + row, 0) = dfval(row, 0) * tau;
        for (auto col = 0u; col < nx; ++col) {
          result.dvecF_dvecX.insert(nf * ival + row, ival * nx + col) = dfval(row, 1 + col);
        }
        for (auto col = 0u; col < nu; ++col) {
          result.dvecF_dvecU.insert(nf * ival + row, ival * nu + col) = dfval(row, 1 + nx + col);
        }
      }
    }
  }
}

/**
 * @brief Evaluate a function at endpoints.
 *
 * Returns a nf vector
 *
 *  F = f(t_0, t_f, x_0, x_f, q)
 *
 * @tparam Der return derivatives w.r.t variables
 *
 * @param nf dimensionality of f image
 * @param nf state space degrees of freedom
 * @param f function (t0, tf, x0, xf, q) -> R^nf
 * @param t0 initial time
 * @param tf final time
 * @param xs state variables (size N+1)
 * @param q integrals
 *
 * @return If Deriv == false,
 * If Deriv == true, {F, dF_dt0, dF_dtf, dF_dvecX, dF_dQ},
 */
template<bool Deriv>
auto colloc_eval_endpt(
  const std::size_t nf,
  const std::size_t nx,
  auto && f,
  [[maybe_unused]] const double t0,
  const double tf,
  std::ranges::sized_range auto && xs,
  const Eigen::VectorXd & Q)
{
  using X = PlainObject<std::ranges::range_value_t<decltype(xs)>>;

  const auto numX = std::ranges::size(xs);
  assert(numX >= 2);

  // NOTE: for now t0 = 0 and we don't want t0 in signatures
  assert(t0 == 0);

  const X x0 = *std::ranges::begin(xs);
  const X xf = *std::ranges::next(std::ranges::begin(xs), numX - 1);

  if constexpr (!Deriv) {
    return f.template operator()<double>(tf, x0, xf, Q);
  } else {
    const auto [Fval, J] = diff::dr(f, wrt(tf, x0, xf, Q));

    assert(static_cast<std::size_t>(J.rows()) == nf);
    assert(static_cast<std::size_t>(J.cols()) == 1 + 2 * nx + Q.size());

    Eigen::SparseMatrix<double> dF_dt0, dF_dtf, dF_dvecX, dF_dQ;

    dF_dt0.resize(nf, 1);
    // dF_dt0.reserve(nf);
    // for (auto i = 0u; i < nf; ++i) { dF_dt0.insert(i, 0) = J(i, 0); }

    dF_dtf.resize(nf, 1);
    dF_dtf.reserve(nf);
    for (auto i = 0u; i < nf; ++i) { dF_dtf.insert(i, 0) = J(i, 0); }

    dF_dvecX.resize(nf, nx * numX);
    Eigen::VectorXi pattern = Eigen::VectorXi::Zero(nx * numX);
    pattern.head(nx).setConstant(nf);
    pattern.tail(nx).setConstant(nf);
    dF_dvecX.reserve(pattern);

    for (auto row = 0u; row < nf; ++row) {
      for (auto col = 0u; col < nx; ++col) {
        dF_dvecX.insert(row, col)                  = J(row, 1 + col);
        dF_dvecX.insert(row, nx * numX - nx + col) = J(row, 1 + nx + col);
      }
    }

    dF_dQ.resize(nf, Q.size());
    dF_dQ.reserve(Eigen::VectorXi::Constant(Q.size(), nf));

    for (auto row = 0u; row < nf; ++row) {
      for (auto col = 0u; col < Q.size(); ++col) {
        dF_dQ.insert(row, col) = J(row, 1 + 2 * nx + col);
      }
    }

    dF_dt0.makeCompressed();
    dF_dtf.makeCompressed();
    dF_dvecX.makeCompressed();
    dF_dQ.makeCompressed();

    return std::make_tuple(Fval, dF_dt0, dF_dtf, dF_dvecX, dF_dQ);
  }
}

/**
 * @brief Evaluate dynamics constraint in all collocation points of a Mesh.
 *
 * @tparam Der return derivatives w.r.t variables
 *
 * @param nx state space degrees of freedom
 * @param f right-hand side of dynamics with signature (t, x, u) -> dx where x and dx are size nx
 * x 1 and u is size nu x 1
 * @param m mesh with a total of N collocation points
 * @param tf final time (variable of size 1)
 * @param x state values (matrix of size nx x N+1)
 * @param u input values (matrix of size nu x N)
 *
 * @return {F, dvecF_dt0, dvecF_dtf, dvecF_dvecX, dvecF_dvecU},
 * where vec(xs) stacks the columns of xs into a single column vector.
 *
 * @note This only works on flat spaces, which is why xs and us are matrices rather than ranges.
 */
template<bool Deriv>
auto colloc_dyn(
  const std::size_t nx,
  auto && f,
  const MeshType auto & m,
  const double t0,
  const double tf,
  const Eigen::MatrixXd & xs,
  const Eigen::MatrixXd & us)
{
  assert(m.N_colloc() + 1 == static_cast<std::size_t>(xs.cols()));  // extra at the end
  assert(m.N_colloc() == static_cast<std::size_t>(us.cols()));      // one per collocation point
  assert(nx == static_cast<std::size_t>(xs.rows()));                // one per collocation point

  const auto N  = m.N_colloc();
  const auto nu = us.rows();

  CollocEvalResult feval_res(nx, nx, nu, N);

  Eigen::MatrixXd XD(nx, m.N_colloc());
  Eigen::SparseMatrix<double> dvecXD_dvecX;

  if constexpr (!Deriv) {
    colloc_eval<0>(feval_res, f, m, t0, tf, xs.colwise(), us.colwise());
  } else {
    colloc_eval<1>(feval_res, f, m, t0, tf, xs.colwise(), us.colwise());

    dvecXD_dvecX.resize(XD.size(), xs.size());

    // reserve sparsity pattern
    Eigen::VectorXi pattern = Eigen::VectorXi::Zero(xs.size());
    for (auto M = 0u, i = 0u; i < m.N_ivals(); ++i) {
      const std::size_t K = m.N_colloc_ival(i);
      pattern.segment(M, (K + 1) * nx) += Eigen::VectorXi::Constant((K + 1) * nx, K);
      M += K * nx;
    }
    dvecXD_dvecX.reserve(pattern);
  }

  for (auto i = 0u, M = 0u; i < m.N_ivals(); M += m.N_colloc_ival(i), ++i) {
    const std::size_t K     = m.N_colloc_ival(i);
    const Eigen::MatrixXd D = m.interval_diffmat(i);
    XD.block(0, M, nx, K)   = xs.block(0, M, nx, K + 1) * D;

    if constexpr (Deriv) {
      // vec(xs * D) = kron(D', I) * vec(xs), so derivative w.r.t vec(xs) = kron(D', I)
      for (auto i = 0u; i < K; ++i) {
        for (auto j = 0u; j < K + 1; ++j) {
          for (auto diag = 0u; diag < nx; ++diag) {
            dvecXD_dvecX.coeffRef(M * nx + i * nx + diag, M * nx + j * nx + diag) += D(j, i);
          }
        }
      }
    }
  }

  Eigen::VectorXd Fv = (XD - (tf - t0) * feval_res.F).reshaped();

  // scale equalities by by quadrature weights
  const auto [n, w] = m.all_nodes_and_weights();

  // vec(A * W) = kron(W', I) * vec(A), so we apply kron(W', I) on the left

  Eigen::SparseMatrix<double> W(N, N);
  W.reserve(Eigen::VectorXi::Ones(N));
  for (auto i = 0u; i < N; ++i) { W.insert(i, i) = w(i); }

  const Eigen::SparseMatrix<double> W_kron_I = kron_identity(W, nx);

  Fv.applyOnTheLeft(W_kron_I);

  if constexpr (!Deriv) {
    return Fv;
  } else {
    dvecXD_dvecX.makeCompressed();

    Eigen::SparseMatrix<double> dF_dt0 = -(tf - t0) * feval_res.dvecF_dt0;
    dF_dt0 += feval_res.F.reshaped().sparseView();  // OK since dvecF_dtf is dense
    dF_dt0 = W_kron_I * dF_dt0;

    Eigen::SparseMatrix<double> dF_dtf = -(tf - t0) * feval_res.dvecF_dtf;
    dF_dtf -= feval_res.F.reshaped().sparseView();  // OK since dvecF_dtf is dense
    dF_dtf = W_kron_I * dF_dtf;

    Eigen::SparseMatrix<double> dF_dvecX = dvecXD_dvecX;
    dF_dvecX -= (tf - t0) * feval_res.dvecF_dvecX;
    dF_dvecX = W_kron_I * dF_dvecX;

    Eigen::SparseMatrix<double> dF_dvecU = -(tf - t0) * W_kron_I * feval_res.dvecF_dvecU;

    dF_dt0.makeCompressed();
    dF_dtf.makeCompressed();
    dF_dvecX.makeCompressed();
    dF_dvecU.makeCompressed();

    return std::make_tuple(
      std::move(Fv),
      std::move(dF_dt0),
      std::move(dF_dtf),
      std::move(dF_dvecX),
      std::move(dF_dvecU));
  }
}
/**
 * @brief Evaluate integral constraint on Mesh.
 *
 * @tparam Der return derivatives w.r.t variables
 *
 * @param nq number of integrals
 * @param g integrand with signature (t, x, u) -> R^{nq} where x is size nx x 1 and u is size nu x
 * 1
 * @param m mesh
 * @param t0 initial time (variable of size 1)
 * @param tf final time (variable of size 1)
 * @param I values (variable of size nq)
 * @param xs state values (variable of size N+1)
 * @param us input values (variable of size N)
 *
 * @return {G, dvecG_dt0, dvecG_dtf, dvecG_dvecX, dvecG_dvecU},
 * where vec(xs) stacks the columns of xs into a single column vector.
 */
template<uint8_t Deriv>
auto colloc_int(
  const std::size_t nq,
  auto && g,
  const MeshType auto & m,
  const double t0,
  const double tf,
  const Eigen::VectorXd & I,
  std::ranges::sized_range auto && xs,
  std::ranges::sized_range auto && us)
{
  assert(static_cast<std::size_t>(I.size()) == nq);

  const auto N  = m.N_colloc();
  const auto nx = dof(*std::ranges::begin(xs));
  const auto nu = dof(*std::ranges::begin(us));

  const auto [n, w] = m.all_nodes_and_weights();

  CollocEvalResult geval_res(nq, nx, nu, N);
  colloc_eval<Deriv>(geval_res, g, m, t0, tf, xs, us);

  const Eigen::VectorXd Iest = geval_res.F * w.head(N);
  Eigen::VectorXd Rv         = (tf - t0) * Iest - I;

  if constexpr (Deriv == 0u) {
    return Rv;
  } else if (Deriv == 1u) {
    const Eigen::SparseMatrix<double> w_kron_I =
      (tf - t0) * kron_identity(w.head(N).transpose(), nq);

    Eigen::SparseMatrix<double> dR_dt0 = w_kron_I * geval_res.dvecF_dt0;
    for (auto i = 0u; i < Iest.size(); ++i) { dR_dt0.coeffRef(i, 0) -= Iest(i); }

    Eigen::SparseMatrix<double> dR_dtf = w_kron_I * geval_res.dvecF_dtf;
    for (auto i = 0u; i < Iest.size(); ++i) { dR_dtf.coeffRef(i, 0) += Iest(i); }

    Eigen::SparseMatrix<double> dR_dvecI = -sparse_identity(nq);

    Eigen::SparseMatrix<double> dR_dvecX = w_kron_I * geval_res.dvecF_dvecX;

    Eigen::SparseMatrix<double> dR_dvecU = w_kron_I * geval_res.dvecF_dvecU;

    dR_dt0.makeCompressed();
    dR_dtf.makeCompressed();
    dR_dvecI.makeCompressed();
    dR_dvecX.makeCompressed();
    dR_dvecU.makeCompressed();

    return std::make_tuple(
      std::move(Rv),
      std::move(dR_dt0),
      std::move(dR_dtf),
      std::move(dR_dvecI),
      std::move(dR_dvecX),
      std::move(dR_dvecU));
  }
}

/**
 * @brief Calculate relative dynamics errors for each interval in mesh.
 *
 * @param nx state space dimension
 * @param f dynamics function
 * @param m Mesh
 * @param t0 initial time variable
 * @param tf final time variable
 * @param x state trajectory
 * @param u input trajectory
 *
 * @return vector with relative errors for every interval in m
 */
Eigen::VectorXd mesh_dyn_error(
  const std::size_t nx,
  auto && f,
  const MeshType auto & m,
  const double t0,
  const double tf,
  const std::function<Eigen::VectorXd(double)> xfun,
  const std::function<Eigen::VectorXd(double)> ufun)
{
  const auto N = m.N_ivals();

  // create a new mesh where each interval is extended
  Mesh mext = m;
  for (auto i = 0u; i < N; ++i) {
    const std::size_t K = m.N_colloc_ival(i);
    mext.set_N_colloc_ival(i, K + 1);
  }

  Eigen::VectorXd ival_errs(N);

  // for each interval
  for (auto i = 0u, M = 0u; i < N; M += m.N_colloc_ival(i), ++i) {
    const std::size_t Kext = mext.N_colloc_ival(i);

    const auto [tau_s, weights] = mext.interval_nodes_and_weights(i);

    assert(std::size_t(tau_s.size()) == Kext + 1);

    // evaluate xs and F at those points
    Eigen::MatrixXd Fval(nx, Kext + 1);
    Eigen::MatrixXd Xval(nx, Kext + 1);
    for (auto j = 0u; j < Kext + 1; ++j) {
      const double tj = t0 + (tf - t0) * tau_s(j);

      // evaluate x and u values at tj using current degree polynomials
      const auto Xj = xfun(tj);
      const auto Uj = ufun(tj);

      // evaluate right-hand side of dynamics at tj
      Fval.col(j) = f(tj, Xj, Uj);

      // store x values for later comparison
      Xval.col(j) = Xj;
    }

    // "integrate" system inside interval
    const Eigen::MatrixXd Xval_est =
      Xval.col(0).replicate(1, Kext) + (tf - t0) * Fval.leftCols(Kext) * mext.interval_intmat(i);

    // absolute error in interval
    Eigen::VectorXd e_abs = (Xval_est - Xval.rightCols(Kext)).colwise().norm();
    Eigen::VectorXd e_rel = e_abs / (1. + Xval.rightCols(Kext).colwise().norm().maxCoeff());

    // mex relative error on interval
    ival_errs(i) = e_rel.maxCoeff();
  }

  return ival_errs;
}

/**
 * @brief Refine intervals in mesh to satisfy a target error criterion.
 * @param[in, out] m mesh to refine
 * @param[in] errs relative errors for all intervals (@see mesh_dyn_error())
 * @param[in] target_err target relative error
 */
void mesh_refine(MeshType auto & m, const Eigen::VectorXd & errs, const double target_err)
{
  const auto N = m.N_ivals();

  assert(N == std::size_t(errs.size()));

  for (auto i = 0u; i < N; ++i) {
    const auto Nmi = N - 1 - i;
    const auto Ki  = m.N_colloc_ival(Nmi);

    if (errs(Nmi) > target_err) {
      const auto Ktarget = Ki + std::lround(std::log(errs(Nmi) / target_err) / std::log(Ki) + 1);
      m.refine_ph(Nmi, Ktarget);
    }
  }
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION_HPP_
