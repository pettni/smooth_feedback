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

#ifndef SMOOTH__FEEDBACK__COLLOCATION__MESH_HPP_
#define SMOOTH__FEEDBACK__COLLOCATION__MESH_HPP_

/**
 * @file
 * @brief Refinable Legendre-Gauss-Radau mesh of time interval [0, 1]
 */

#include <Eigen/Core>
#include <Eigen/LU>
#include <smooth/internal/utils.hpp>
#include <smooth/polynomial/quadrature.hpp>

#include <ranges>
#include <span>
#include <vector>

#include "smooth/feedback/traits.hpp"
#include "smooth/feedback/utils/sparse.hpp"

namespace smooth::feedback {

using smooth::utils::zip;

using std::views::iota, std::views::drop, std::views::reverse, std::views::take,
  std::views::transform, std::views::join;

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
   *
   * @note Allocates heap memory.
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
  inline std::size_t N_colloc_ival(std::size_t i) const
  {
    assert(i < intervals_.size());
    return intervals_[i].K;
  }

  /**
   * @brief Refine interval using the ph strategy.
   *
   * @param i index of interval to refine
   * @param D target number of collocation points in refined interval
   *
   * If D > Kmax, or current degree > Kmax    then the interval is divided into
   *                                          n = max(2, ceil(D / Kmin)) intervals with deg Kmin
   * If D < current degree,                   then nothing is done.
   * If D <= Kmax,                            then the polynomial degree is increased to D.
   *
   * @note May allocate heap memory due to vector resizing if the number of intervals is increased.
   */
  inline void refine_ph(std::size_t i, std::size_t D)
  {
    assert(i < intervals_.size());
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
   * @brief Refine intervals in mesh to satisfy a target error criterion.
   * @param errs relative errors for all intervals (@see mesh_dyn_error())
   * @param target_err target relative error
   */
  inline void refine_errors(std::ranges::sized_range auto && errs, double target_err)
  {
    const auto N = N_ivals();

    assert(N == std::size_t(std::ranges::size(errs)));

    for (const auto & [i, e] : zip(iota(0u, N) | reverse, errs | reverse)) {

      const auto Ki = N_colloc_ival(i);

      if (e > target_err) {
        const auto Ktarget = Ki + std::lround(std::log(e / target_err) / std::log(Ki) + 1);
        refine_ph(i, Ktarget);
      }
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
   * @brief Interval nodes (as range of doubles).
   *
   * @note Includes extra point at 1, i.e. size of returned range is equal to N_colloc_ival()+1
   */
  inline auto interval_nodes(std::size_t i) const
  {
    const std::size_t k = intervals_[i].K;

    assert(Kmin <= k && k <= Kmax + 1);

    std::span<const double> sp;

    utils::static_for<Kmax + 2 - Kmin>([&](auto ivar) {
      static constexpr auto K = Kmin + ivar;
      if (K == k) {
        static constexpr auto nw_ext_s = detail::lgr_plus_one<K>();

        sp = std::span<const double>(nw_ext_s.first.data(), k + 1);
      }
    });

    const double tau0 = intervals_[i].tau0;
    const double tauf = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;
    const double al   = (tauf - tau0) / 2;

    return transform(std::move(sp), [tau0, al](double d) -> double { return tau0 + al * (d + 1); });
  }

  /**
   * @brief Nodes (as range of doubles).
   *
   * @note Includes extra point at 1, i.e. size of returned range is equal to N_colloc()+1
   *
   * @note The result is an input range
   */
  inline auto all_nodes() const
  {
    const auto n_ivals = N_ivals();
    auto all_views     = iota(0u, n_ivals) | transform([this, n_ivals = n_ivals](auto i) {
                       const auto n_ival    = N_colloc_ival(i);
                       const int64_t n_take = i + 1 < n_ivals ? n_ival : n_ival + 1;
                       return interval_nodes(i) | take(n_take);
                     });

    return join(std::move(all_views));
  }

  /**
   * @brief Interval weights (as range of doubles)
   *
   * @note Includes zero weight at 1, i.e. size of returned range is equal to N_colloc_ival()+1
   */
  inline auto interval_weights(std::size_t i) const
  {
    const std::size_t k = intervals_[i].K;

    assert(Kmin <= k && k <= Kmax + 1);

    std::span<const double> sp;

    utils::static_for<Kmax + 2 - Kmin>([&](auto ivar) {
      static constexpr auto K = Kmin + ivar;
      if (K == k) {
        static constexpr auto nw_ext_s = detail::lgr_plus_one<K>();

        sp = std::span<const double>(nw_ext_s.second.data(), k + 1);
      }
    });

    const double tau0 = intervals_[i].tau0;
    const double tauf = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;
    const double al   = (tauf - tau0) / 2;

    return transform(std::move(sp), [al](double d) -> double { return al * d; });
  }

  /**
   * @brief Weights (as range of doubles)
   *
   * @note Includes zero weight at 1, i.e. size of returned range is equal to N_colloc()+1
   *
   * @note The result is an input range
   */
  inline auto all_weights() const
  {
    const auto n_ivals = N_ivals();
    auto all_views     = iota(0u, n_ivals) | transform([this, n_ivals = n_ivals](auto i) {
                       const auto n_ival    = N_colloc_ival(i);
                       const int64_t n_take = i + 1 < n_ivals ? n_ival : n_ival + 1;
                       return interval_weights(i) | take(n_take);
                     });

    return join(std::move(all_views));
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
   *
   * @note Allocates heap memory for return value.
   */
  inline Eigen::MatrixXd interval_diffmat(std::size_t i) const
  {
    const std::size_t k = intervals_[i].K;

    const double tau0 = intervals_[i].tau0;
    const double tauf = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;

    Eigen::MatrixXd ret(k + 1, k);

    utils::static_for<Kmax + 2 - Kmin>([&](auto ivar) {
      static constexpr auto K = Kmin + ivar;
      if (K == k) {
        static constexpr auto nw_ext_s = detail::lgr_plus_one<K>();
        static constexpr auto B_ext_s  = lagrange_basis<K>(nw_ext_s.first);
        static constexpr auto D_ext_s =
          polynomial_basis_derivatives<K, K + 1>(B_ext_s, nw_ext_s.first)
            .template block<K + 1, K>(0, 0);
        ret = MatMap(D_ext_s[0].data(), k + 1, k);
      }
    });

    ret *= 2. / (tauf - tau0);
    return ret;
  }

  /**
   * @brief Interval differentiation matrix (unscaled).
   *
   * Returns a Map D_us and a scalar alpha s.t. D = alpha * D_us
   *
   * @see interval_diffmat
   */
  inline std::pair<double, MatMap> interval_diffmat_unscaled(std::size_t i) const
  {
    const std::size_t k = intervals_[i].K;

    const double tau0 = intervals_[i].tau0;
    const double tauf = i + 1 < intervals_.size() ? intervals_[i + 1].tau0 : 1.;

    MatMap ret(nullptr, 0, 0);

    utils::static_for<Kmax + 2 - Kmin>([&](auto ivar) {
      static constexpr auto K = Kmin + ivar;
      if (K == k) {
        static constexpr auto nw_ext_s = detail::lgr_plus_one<K>();
        static constexpr auto B_ext_s  = lagrange_basis<K>(nw_ext_s.first);
        static constexpr auto D_ext_s =
          polynomial_basis_derivatives<K, K + 1>(B_ext_s, nw_ext_s.first)
            .template block<K + 1, K>(0, 0);
        // Eigen maps don't have copy constructors so use placement new
        new (&ret) MatMap(D_ext_s[0].data(), k + 1, k);
      }
    });

    return {2. / (tauf - tau0), ret};
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
   *
   * @note Allocates heap memory for return value.
   *
   * @note Performs a matrix inverse.
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
   * @brief Increase the degree of all intervals by one (up to maximal degree Kmax + 1)
   */
  void increase_degrees()
  {
    for (auto & ival : intervals_) { ival.K = std::min(ival.K + 1, Kmax + 1); }
  }

  /**
   * @brief Decrease the degree of all intervals by one (down to minimal degree Kmin)
   */
  void decrease_degrees()
  {
    for (auto & ival : intervals_) { ival.K = std::max(ival.K - 1, Kmin); }
  }

  /**
   * @brief Evaluate a function
   *
   * @tparam RetT return value type
   *
   * @param t time value in [0, 1]
   * @param r values for the collocation points (size N [extend=false] or N+1 [extend=true])
   * @param p derivative to evaluate
   * @param extend set to true if a value is provided for t=+1
   */
  template<smooth::traits::RnType RetT>
  RetT eval(double t, std::ranges::range auto && r, std::size_t p = 0, bool extend = true) const
  {
    const std::size_t ival = interval_find(t);
    const std::size_t k    = intervals_[ival].K;

    const double tau0 = intervals_[ival].tau0;
    const double tauf = ival + 1 < intervals_.size() ? intervals_[ival + 1].tau0 : 1.;

    const double u = 2 * (t - tau0) / (tauf - tau0) - 1;

    int64_t N_before = 0;
    for (auto i = 0u; i < ival; ++i) { N_before += intervals_[i].K; }

    // initialize output variable
    RetT ret = RetT::Zero(dof(*std::ranges::begin(r)));

    utils::static_for<Kmax + 2 - Kmin>([&](auto i) {
      static constexpr auto K = Kmin + i;
      if (K == k) {
        if (extend || ival + 1 < intervals_.size()) {
          static constexpr auto nw_ext_s = detail::lgr_plus_one<K>();
          static constexpr auto B_ext_s  = lagrange_basis<K>(nw_ext_s.first);  // K+1 x K+1
          const auto U                   = monomial_derivative<K>(u, p);       // 1 x K+1
          const auto W                   = U * B_ext_s;                        // 1 x K+1

          for (const auto & [w, v] : zip(std::span(W[0].data(), k + 1), r | drop(N_before))) {
            ret += w * v;
          }
        } else {
          static constexpr auto nw_s = lgr_nodes<K>();
          static constexpr auto B_s  = lagrange_basis<K - 1>(nw_s.first);  // K x K
          const auto U               = monomial_derivative<K - 1>(u, p);   // 1 x K
          const auto W               = U * B_s;                            // 1 x K

          for (const auto & [w, v] : zip(std::span(W[0].data(), k), r | drop(N_before))) {
            ret += w * v;
          }
        }
      }
    });

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
concept MeshType = traits::is_specialization_of_sizet_v<std::decay_t<T>, Mesh>;

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__COLLOCATION__MESH_HPP_
