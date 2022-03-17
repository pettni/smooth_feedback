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

#ifndef SMOOTH__FEEDBACK__FLATTEN_OCP_HPP_
#define SMOOTH__FEEDBACK__FLATTEN_OCP_HPP_

/**
 * @file
 * @brief Reformulate an optimal control problem on a Lie group as an optimal control problem in the
 * tangent space around a reference trajectory.
 *
 * @todo More efficient implementation of Hessian in FlatDyn.
 * @todo Accept dxl as a template argument to avoid double differentiation.
 */

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <smooth/diff.hpp>

#include "ocp.hpp"
#include "utils/d2r_exp_sparse.hpp"
#include "utils/dr_exp_sparse.hpp"
#include "utils/sparse.hpp"

namespace smooth::feedback {

// \cond
namespace detail {

/// @brief The first Bernoulli numbers
static constexpr std::array<double, 23> kBn{
  1,               // 0
  -1. / 2,         // 1
  1. / 6,          // 2
  0.,              // 3
  -1. / 30,        // 4
  0,               // 5
  1. / 42,         // 6
  0,               // 7
  -1. / 30,        // 8
  0,               // 9
  5. / 66,         // 10
  0,               // 11
  -691. / 2730,    // 12
  0,               // 13
  7. / 6,          // 14
  0,               // 15
  -3617. / 510,    // 16
  0,               // 17
  43867. / 798,    // 18
  0,               // 19
  -174611. / 330,  // 20
  0,               // 21
  854513. / 138,   // 22
};

/**
 * @brief Algebra generators as sparse matrices
 */
template<LieGroup G>
inline auto algebra_generators_sparse = []() -> std::array<Eigen::SparseMatrix<double>, Dof<G>> {
  std::array<Eigen::SparseMatrix<double>, Dof<G>> ret;
  for (auto i = 0u; i < Dof<G>; ++i) {
    ret[i] = ad<G>(Tangent<G>::Unit(i)).sparseView();
    ret[i].makeCompressed();
  }
  return ret;
}();

/**
 * @brief Sparse version of ad.
 */
template<LieGroup G>
void ad_sparse(Eigen::SparseMatrix<double> & out, const Tangent<G> & a)
{
  if (out.isCompressed()) {
    out.coeffs().setZero();
  } else {
    out.setZero();
  }
  out.resize(Dof<G>, Dof<G>);

  for (auto i = 0u; i < Dof<G>; ++i) { out += a(i) * algebra_generators_sparse<G>[i]; }

  out.makeCompressed();
}

/**
 * @brief Sparse matrices containing reordered rows of algebra generators.
 */
template<LieGroup G>
inline auto algebra_generators_sparse_reordered =
  []() -> std::array<Eigen::SparseMatrix<double, Eigen::RowMajor>, Dof<G>> {
  std::array<Eigen::SparseMatrix<double, Eigen::RowMajor>, Dof<G>> ret;
  for (auto k = 0u; k < Dof<G>; ++k) {
    ret[k].resize(Dof<G>, Dof<G>);
    for (auto i = 0u; i < Dof<G>; ++i) { ret[k].row(i) = algebra_generators_sparse<G>[i].row(k); }
    ret[k].makeCompressed();
  }
  return ret;
}();

/**
 * @brief Derivative of the adjoint as sparse matrix
 */
template<LieGroup G>
inline Eigen::SparseMatrix<double> d_ad = []() -> Eigen::SparseMatrix<double> {
  Eigen::SparseMatrix<double> ret;
  ret.resize(Dof<G>, Dof<G> * Dof<G>);
  for (auto i = 0u; i < Dof<G>; ++i) {
    for (auto j = 0u; j < Dof<G>; ++j) {
      ret.col(j * Dof<G> + i) = detail::algebra_generators_sparse<G>[i].row(j).transpose();
    }
  }
  ret.makeCompressed();
  return ret;
}();

/**
 * @brief Flattening of dynamics function (t, x, u) -> Tangent, and its derivatives.
 *
 * @note Do not use numeric differentiation of operator() (it differentiates inside)
 * @note Only considers first derivative of xl and ul
 */
template<LieGroup X, Manifold U, typename F, typename Xl, typename Ul>
class FlatDyn
{
private:
  static constexpr auto Nx    = Dof<X>;
  static constexpr auto Nu    = Dof<U>;
  static constexpr auto Nouts = Nx;
  static constexpr auto Nvars = 1 + Nx + Nu;

  static constexpr auto t_B = 0;
  static constexpr auto x_B = t_B + 1;
  static constexpr auto u_B = x_B + Nx;

  using E = Tangent<X>;
  using V = Tangent<U>;

  F f;
  Xl xl;
  Ul ul;

  Eigen::SparseMatrix<double> Joplus_, Hoplus_, J_, H_;

  // Would ideally like to remove these temporaries...
  Eigen::SparseMatrix<double> ji_, ji_tmp_, hi_, hi_tmp_;
  Eigen::SparseMatrix<double> ad_e, ad_vi;
  Eigen::SparseMatrix<double> dexpinv_e_;
  Eigen::SparseMatrix<double> d2exp_e_, d2exp_v_, d2expinv_e_;

  /// @brief Calculate jacobian of (t, x(t)+e, u(t)+v) w.r.t. (t, e, v)
  void update_joplus(const E & e, const V & v, const E & dxl, const V & dul)
  {
    set_zero(Joplus_);
    block_add_identity(Joplus_, t_B, t_B, 1);
    block_add(Joplus_, x_B, t_B, Ad<X>(smooth::exp<X>(-e)) * dxl);
    block_add(Joplus_, u_B, t_B, Ad<U>(smooth::exp<U>(-v)) * dul);
    dr_exp_sparse<X>(Joplus_, e, x_B, x_B);
    dr_exp_sparse<U>(Joplus_, v, u_B, u_B);
    Joplus_.makeCompressed();
  }

  /// @brief Calculate hessian of (t, x(t)+e, u(t)+v) w.r.t. (t, e, v)
  void update_hoplus(const E & e, const V & v, const E & dxl, const V & dul)
  {
    d2r_exp_sparse<X>(d2exp_e_, e);
    d2r_exp_sparse<U>(d2exp_v_, v);

    // d (Ad_X b) = -ad_(Ad_X b) * Ad_X
    const TangentMap<X> Adexp_X  = Ad<X>(smooth::exp<X>(-e));
    const TangentMap<X> dAdexp_X = ad<X>(Adexp_X * dxl) * Adexp_X * dr_exp<X>(-e);
    const TangentMap<U> Adexp_U  = Ad<U>(smooth::exp<U>(-v));
    const TangentMap<U> dAdexp_U = ad<U>(Adexp_U * dul) * Adexp_U * dr_exp<U>(-v);

    set_zero(Hoplus_);
    for (auto nx = 0u; nx < Nx; ++nx) {
      const auto b0 = Nvars * (x_B + nx);
      block_add(Hoplus_, t_B, b0 + x_B, dAdexp_X.middleRows(nx, 1));
    }
    for (auto nu = 0u; nu < Nu; ++nu) {
      const auto b0 = Nvars * (u_B + nu);
      block_add(Hoplus_, t_B, b0 + u_B, dAdexp_U.middleRows(nu, 1));
    }
    for (auto nx = 0u; nx < Nx; ++nx) {
      const auto b0 = Nvars * (x_B + nx);
      block_add(Hoplus_, x_B, b0 + x_B, d2exp_e_.middleCols(nx * Nx, Nx));
    }
    for (auto nu = 0u; nu < Nu; ++nu) {
      const auto b0 = Nvars * (u_B + nu);
      block_add(Hoplus_, u_B, b0 + u_B, d2exp_v_.middleCols(nu * Nu, Nu));
    }
    Hoplus_.makeCompressed();
  }

public:
  template<typename A1, typename A2, typename A3>
  FlatDyn(A1 && a1, A2 && a2, A3 && a3)
      : f(std::forward<A1>(a1)), xl(std::forward<A2>(a2)), ul(std::forward<A3>(a3)),
        Joplus_(Nvars, Nvars), Hoplus_(Nvars, Nvars * Nvars), J_(Nouts, Nvars),
        H_(Nvars, Nouts * Nvars), ji_(Nouts, Nvars), ji_tmp_(Nouts, Nvars),
        hi_(Nvars, Nouts * Nvars), hi_tmp_(Nvars, Nouts * Nvars), dexpinv_e_(Nx, Nx),
        d2exp_e_(Nx, Nx * Nx), d2exp_v_(Nu, Nu * Nu), d2expinv_e_(Nx, Nx * Nx)
  {}

  template<typename T>
  CastT<T, E> operator()(const T & t, const CastT<T, E> & e, const CastT<T, V> & v) const
  {
    using XT = CastT<T, X>;

    // can not double-differentiate, so we hide derivative of xl w.r.t. t
    const double tdbl           = static_cast<double>(t);
    const auto [unused, dxlval] = diff::dr(xl, wrt(tdbl));

    return dr_expinv<XT>(e) * (f(t, rplus(xl(t), e), rplus(ul(t), v)) - dxlval.template cast<T>())
         + ad<XT>(e) * dxlval.template cast<T>();
  }

  // First derivative
  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  jacobian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>>)
  {
    const double tdbl          = static_cast<double>(t);
    const auto [xlval, dxlval] = diff::dr(xl, wrt(tdbl));
    const auto [ulval, dulval] = diff::dr(ul, wrt(tdbl));
    const auto x               = rplus(xlval, e);
    const auto u               = rplus(ulval, v);

    // zero working memory
    set_zero(dexpinv_e_);
    set_zero(d2expinv_e_);
    set_zero(J_);

    // left stuff
    dr_expinv_sparse<X>(dexpinv_e_, e);

    // value and derivative of f
    const auto fval = f(t, x, u);
    const auto & Jf = f.jacobian(t, x, u);
    // derivatives of +
    update_joplus(e, v, dxlval, dulval);

    // Add drexpinv * d (f \circ +)
    J_ = dexpinv_e_ * Jf * Joplus_;

    d2r_expinv_sparse<X>(d2expinv_e_, e);

    // Add d ( drexpinv ) * (f \circ (+) - dxl) + d ( ad ) * dxl
    for (auto i = 0u; i < d2expinv_e_.outerSize(); ++i) {
      for (Eigen::InnerIterator it(d2expinv_e_, i); it; ++it) {
        J_.coeffRef(it.col() / Nx, 1 + (it.col() % Nx)) +=
          (fval(it.row()) - dxlval(it.row())) * it.value();
      }
    }
    for (auto i = 0u; i < d_ad<X>.outerSize(); ++i) {
      for (Eigen::InnerIterator it(d_ad<X>, i); it; ++it) {
        J_.coeffRef(it.col() / Nx, 1 + (it.col() % Nx)) += dxlval(it.row()) * it.value();
      }
    }

    // compress working memory
    dexpinv_e_.makeCompressed();
    d2expinv_e_.makeCompressed();

    J_.makeCompressed();
    return J_;
  }

  // Second derivative
  //    \sum Bn (-1)^n / n! d2r (ad_a^n f)_aa - \sum Bn / n! d2r(ad_a^n dxl)_aa
  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  hessian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>> &&
      diff::detail::diffable_order2<F, std::tuple<double, X, U>>)
  {
    const double tdbl          = static_cast<double>(t);
    const auto [xlval, dxlval] = diff::dr(xl, wrt(tdbl));
    const auto [ulval, dulval] = diff::dr(ul, wrt(tdbl));

    const auto x    = rplus(xlval, e);
    const auto u    = rplus(ul(t), v);
    const auto & Jf = f.jacobian(t, x, u);  // nx x (1 + nx + nu)
    const auto & Hf = f.hessian(t, x, u);   // (1 + nx + nu) x (nx * (1 + nx + nu))
    ad_sparse<X>(ad_e, e);

    update_joplus(e, v, dxlval, dulval);
    update_hoplus(e, v, dxlval, dulval);

    double coef   = 1;                    // hold (-1)^i / i!
    Tangent<X> vi = f(t, x, u) - dxlval;  // (ad_a)^i * (f - dxl)
    ji_           = Jf * Joplus_;         // dr (vi)_{t, e, v}
    set_zero(hi_);
    d2r_fog(hi_, Jf, Hf, Joplus_, Hoplus_);  // d2r (vi)_{t, e, v}

    set_zero(H_);
    for (auto iter = 0u; iter < std::tuple_size_v<decltype(kBn)>; ++iter) {
      if (kBn[iter] != 0) { block_add(H_, 0, 0, hi_, kBn[iter] * coef); }

      // update hi_
      hi_tmp_.setZero();
      for (auto i = 0u; i < ad_e.outerSize(); ++i) {
        for (Eigen::InnerIterator it(ad_e, i); it; ++it) {
          const auto b0 = it.row() * Nvars;
          block_add(hi_tmp_, 0, b0, hi_.middleCols(it.col() * Nvars, Nvars), it.value());
        }
      }
      for (auto k = 0u; k < Nx; ++k) {
        const auto b0 = k * Nvars;
        block_add(hi_tmp_, 1, b0, algebra_generators_sparse_reordered<X>[k] * ji_);
        block_add(
          hi_tmp_, 0, b0 + 1, ji_.transpose() * algebra_generators_sparse_reordered<X>[k], -1);
      }
      std::swap(hi_, hi_tmp_);

      // update ji
      ji_tmp_.setZero();
      ji_tmp_ = ad_e * ji_;
      ad_sparse<X>(ad_vi, vi);
      block_add(ji_tmp_, 0, 1, ad_vi, -1);
      std::swap(ji_, ji_tmp_);

      // update vi
      vi.applyOnTheLeft(ad_e);

      coef *= (-1.) / (iter + 1);
    }

    H_.makeCompressed();
    return H_;
  }
};

/**
 * @brief Flattening of inner function (t, x, u) -> Vector, and its derivatives.
 *
 * @note Only considers first derivative of xl and ul
 */
template<LieGroup X, Manifold U, std::size_t Nouts, typename F, typename Xl, typename Ul>
class FlatInnerFun
{
private:
  F f;
  Xl xl;
  Ul ul;

  static constexpr auto Nx    = Dof<X>;
  static constexpr auto Nu    = Dof<U>;
  static constexpr auto Nvars = 1 + Nx + Nu;

  using E = Tangent<X>;
  using V = Tangent<U>;

  static constexpr auto t_B = 0;
  static constexpr auto x_B = t_B + 1;
  static constexpr auto u_B = x_B + Nx;

  Eigen::SparseMatrix<double> Joplus_, Hoplus_, J_, H_;
  Eigen::SparseMatrix<double> d2exp_e_, d2exp_v_;

  /// @brief Calculate jacobian of (t, x(t)+e, u(t)+v) w.r.t. (t, e, v)
  void update_joplus(const E & e, const V & v, const E & dxl, const V & dul)
  {
    set_zero(Joplus_);
    block_add_identity(Joplus_, t_B, t_B, 1);
    block_add(Joplus_, x_B, t_B, Ad<X>(smooth::exp<X>(-e)) * dxl);
    block_add(Joplus_, u_B, t_B, Ad<U>(smooth::exp<U>(-v)) * dul);
    dr_exp_sparse<X>(Joplus_, e, x_B, x_B);
    dr_exp_sparse<U>(Joplus_, v, u_B, u_B);
    Joplus_.makeCompressed();
  }

  /// @brief Calculate hessian of (t, x(t)+e, u(t)+v) w.r.t. (t, e, v)
  void update_hoplus(const E & e, const V & v, const E & dxl, const V & dul)
  {
    d2r_exp_sparse<X>(d2exp_e_, e);
    d2r_exp_sparse<U>(d2exp_v_, v);

    // d (Ad_X b) = -ad_(Ad_X b) * Ad_X
    const TangentMap<X> Adexp_X  = Ad<X>(smooth::exp<X>(-e));
    const TangentMap<X> dAdexp_X = ad<X>(Adexp_X * dxl) * Adexp_X * dr_exp<X>(-e);
    const TangentMap<U> Adexp_U  = Ad<U>(smooth::exp<U>(-v));
    const TangentMap<U> dAdexp_U = ad<U>(Adexp_U * dul) * Adexp_U * dr_exp<U>(-v);

    set_zero(Hoplus_);
    for (auto nx = 0u; nx < Nx; ++nx) {
      const auto b0 = Nvars * (x_B + nx);
      block_add(Hoplus_, t_B, b0 + x_B, dAdexp_X.middleRows(nx, 1));
    }
    for (auto nu = 0u; nu < Nu; ++nu) {
      const auto b0 = Nvars * (u_B + nu);
      block_add(Hoplus_, t_B, b0 + u_B, dAdexp_U.middleRows(nu, 1));
    }
    for (auto nx = 0u; nx < Nx; ++nx) {
      const auto b0 = Nvars * (x_B + nx);
      block_add(Hoplus_, x_B, b0 + x_B, d2exp_e_.middleCols(nx * Nx, Nx));
    }
    for (auto nu = 0u; nu < Nu; ++nu) {
      const auto b0 = Nvars * (u_B + nu);
      block_add(Hoplus_, u_B, b0 + u_B, d2exp_v_.middleCols(nu * Nu, Nu));
    }
    Hoplus_.makeCompressed();
  }

public:
  template<typename A1, typename A2, typename A3>
  FlatInnerFun(A1 && a1, A2 && a2, A3 && a3)
      : f(std::forward<A1>(a1)), xl(std::forward<A2>(a2)), ul(std::forward<A3>(a3)),
        Joplus_(Nvars, Nvars), Hoplus_(Nvars, Nvars * Nvars), J_(Nouts, Nvars),
        H_(Nvars, Nouts * Nvars), d2exp_e_(Nx, Nx * Nx), d2exp_v_(Nu, Nu * Nu)
  {}

  template<typename T>
  Eigen::Vector<T, Nouts>
  operator()(const T & t, const CastT<T, E> & e, const CastT<T, V> & v) const
  {
    return f.template operator()<T>(t, rplus(xl(t), e), rplus(ul(t), v));
  }

  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  jacobian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>>)
  {
    const auto & [xlval, dxlval] = diff::dr(xl, wrt(t));
    const auto & [ulval, dulval] = diff::dr(ul, wrt(t));

    // derivative of f
    const auto & Jf = f.jacobian(t, rplus(xlval, e), rplus(ulval, v));
    // derivative of +
    update_joplus(e, v, dxlval, dulval);
    // update result
    set_zero(J_);
    J_ = Jf * Joplus_;
    J_.makeCompressed();
    return J_;
  }

  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  hessian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>> &&
      diff::detail::diffable_order2<F, std::tuple<double, X, U>>)
  {
    const auto & [xlval, dxlval] = diff::dr(xl, wrt(t));
    const auto & [ulval, dulval] = diff::dr(ul, wrt(t));

    const auto x = rplus(xlval, e);
    const auto u = rplus(ulval, v);

    // derivatives of f
    const auto & Jf = f.jacobian(t, x, u);
    const auto & Hf = f.hessian(t, x, u);
    // derivatives of +
    update_joplus(e, v, dxlval, dulval);
    update_hoplus(e, v, dxlval, dulval);
    // update result
    set_zero(H_);
    d2r_fog(H_, Jf, Hf, Joplus_, Hoplus_);
    H_.makeCompressed();
    return H_;
  }
};

/**
 * @brief Flattening of endpoint function (tf, x0, xf, q) -> Vector, and its derivatives.
 *
 * @note Only considers first derivative of xl
 */
template<LieGroup X, Manifold U, std::size_t Nq, std::size_t Nouts, typename F, typename Xl>
class FlatEndptFun
{
private:
  F f;
  Xl xl;

  static constexpr auto Nx    = Dof<X>;
  static constexpr auto Nvars = 1 + 2 * Nx + Nq;

  using E = Tangent<X>;
  using Q = Eigen::Vector<Scalar<X>, Nq>;

  static constexpr auto tf_B = 0;
  static constexpr auto x0_B = tf_B + 1;
  static constexpr auto xf_B = x0_B + Nx;
  static constexpr auto q_B  = xf_B + Nx;

  Eigen::SparseMatrix<double> Joplus_, Hoplus_, J_, H_;

  Eigen::SparseMatrix<double> d2exp_e_;

  /// @brief Calculate jacobian of (tf, xl(0.)+e0, xl(tf)+ef, q) w.r.t. (tf, e0, ef, q)
  void update_joplus(const E & e0, const E & ef, const E & dxlf)
  {
    set_zero(Joplus_);
    block_add_identity(Joplus_, tf_B, tf_B, 1);
    block_add(Joplus_, xf_B, tf_B, Ad<X>(smooth::exp<X>(-ef)) * dxlf);
    dr_exp_sparse<X>(Joplus_, e0, x0_B, x0_B);
    dr_exp_sparse<X>(Joplus_, ef, xf_B, xf_B);
    block_add_identity(Joplus_, q_B, q_B, Nq);
    Joplus_.makeCompressed();
  }

  /// @brief Calculate hessian of (tf, xl(0.)+e0, xl(tf)+ef, q) w.r.t. (tf, e0, ef, q)
  void update_hoplus(const E & e0, const E & ef, [[maybe_unused]] const E & dxlf)
  {
    // dr (Ad_X b)_X = -ad_{Ad_X b} Ad_X
    const TangentMap<X> Adexp_f  = Ad<X>(smooth::exp<X>(-ef));
    const TangentMap<X> dAdexp_f = ad<X>(Adexp_f * dxlf) * Adexp_f * dr_exp<X>(-ef);

    set_zero(Hoplus_);

    for (auto nx = 0u; nx < Nx; ++nx) {
      const auto b0 = Nvars * (xf_B + nx);
      block_add(Hoplus_, tf_B, b0 + xf_B, dAdexp_f.middleRows(nx, 1));
    }
    d2r_exp_sparse<X>(d2exp_e_, e0);
    for (auto nx = 0u; nx < Nx; ++nx) {
      const auto b0 = Nvars * (x0_B + nx);
      block_add(Hoplus_, x0_B, b0 + x0_B, d2exp_e_.middleCols(nx * Nx, Nx));
    }
    d2r_exp_sparse<X>(d2exp_e_, ef);
    for (auto nx = 0u; nx < Nx; ++nx) {
      const auto b0 = Nvars * (xf_B + nx);
      block_add(Hoplus_, xf_B, b0 + xf_B, d2exp_e_.middleCols(nx * Nx, Nx));
    }
    Hoplus_.makeCompressed();
  }

public:
  template<typename A1, typename A2>
  FlatEndptFun(A1 && a1, A2 && a2)
      : f(std::forward<A1>(a1)), xl(std::forward<A2>(a2)), Joplus_(Nvars, Nvars),
        Hoplus_(Nvars, Nvars * Nvars), J_(Nouts, Nvars), H_(Nvars, Nouts * Nvars),
        d2exp_e_(Nx, Nx * Nx)
  {}
  template<typename T>
  auto operator()(
    const T & tf, const CastT<T, E> & e0, const CastT<T, E> & ef, const CastT<T, Q> & q) const
  {
    return f.template operator()<T>(tf, rplus(xl(T(0.)), e0), rplus(xl(tf), ef), q);
  }

  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  jacobian(double tf, const E & e0, const E & ef, const Q & q) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, X, Q>>)
  {
    const auto & [xlfval, dxlfval] = diff::dr(xl, wrt(tf));

    const auto x0 = rplus(xl(0.), e0);
    const auto xf = rplus(xlfval, ef);

    // derivative of f
    const auto & Jf = f.jacobian(tf, x0, xf, q);
    // derivative of +
    update_joplus(e0, ef, dxlfval);
    // update result
    set_zero(J_);
    J_ = Jf * Joplus_;
    J_.makeCompressed();
    return J_;
  }

  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  hessian(double tf, const E & e0, const E & ef, const Q & q) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, X, Q>> &&
      diff::detail::diffable_order2<F, std::tuple<double, X, X, Q>>)
  {
    const auto & [xlfval, dxlfval] = diff::dr(xl, wrt(tf));

    const auto x0 = rplus(xl(0.), e0);
    const auto xf = rplus(xlfval, ef);

    // derivatives of f
    const auto & Jf = f.jacobian(tf, x0, xf, q);  // Nouts x Nx
    const auto & Hf = f.hessian(tf, x0, xf, q);   // Nx x (Nouts * Nx)
    // derivatives of +
    update_joplus(e0, ef, dxlfval);
    update_hoplus(e0, ef, dxlfval);
    // update result
    set_zero(H_);
    d2r_fog(H_, Jf, Hf, Joplus_, Hoplus_);
    H_.makeCompressed();
    return H_;
  }
};

}  // namespace detail
// \endcond

/**
 * @brief Flatten a LieGroup OCP by defining it in the tangent space around a trajectory.
 *
 * @param ocp OCPType defined on a LieGroup
 * @param xl nominal state trajectory
 * @param ul nominal state trajectory
 *
 * @note The flattened problem defines analytical jacobians and hessians if \p ocp does.
 *
 * @warn The Hessian of the flattened dynamics is not implemented in an efficient manner.
 *
 * @return FlatOCPType in variables (xe, ue) obtained via variables change x = xl ⊕ xe, u = ul ⊕
 * ue,
 */
auto flatten_ocp(const OCPType auto & ocp, auto && xl, auto && ul)
{
  using ocp_t = std::decay_t<decltype(ocp)>;
  using X     = typename ocp_t::X;
  using U     = typename ocp_t::U;
  using Xl    = decltype(xl);
  using Ul    = decltype(ul);

  static constexpr auto Nq = ocp_t::Nq;

  return OCP<
    Tangent<X>,
    Tangent<U>,
    detail::FlatEndptFun<X, U, Nq, 1, decltype(ocp.theta), Xl>,
    detail::FlatDyn<X, U, decltype(ocp.f), Xl, Ul>,
    detail::FlatInnerFun<X, U, Nq, decltype(ocp.g), Xl, Ul>,
    detail::FlatInnerFun<X, U, ocp_t::Ncr, decltype(ocp.cr), Xl, Ul>,
    detail::FlatEndptFun<X, U, Nq, ocp_t::Nce, decltype(ocp.ce), Xl>>{
    .theta = detail::FlatEndptFun<X, U, Nq, 1, decltype(ocp.theta), Xl>{ocp.theta, xl},
    .f     = detail::FlatDyn<X, U, decltype(ocp.f), Xl, Ul>{ocp.f, xl, ul},
    .g     = detail::FlatInnerFun<X, U, Nq, decltype(ocp.g), Xl, Ul>{ocp.g, xl, ul},
    .cr    = detail::FlatInnerFun<X, U, ocp_t::Ncr, decltype(ocp.cr), Xl, Ul>{ocp.cr, xl, ul},
    .crl   = ocp.crl,
    .cru   = ocp.cru,
    .ce    = detail::FlatEndptFun<X, U, Nq, ocp_t::Nce, decltype(ocp.ce), Xl>{ocp.ce, xl},
    .cel   = ocp.cel,
    .ceu   = ocp.ceu,
  };
}

/**
 * @brief Unflatten a FlatOCPSolution
 *
 * If flat_sol is a solution to flat_ocp = flatten_ocp(ocp, xl_fun, ul_fun),
 * then unflatten_ocpsol(flat_sol, xl_fun, ul_fun) is a solution to ocp.
 */
template<LieGroup X, Manifold U>
auto unflatten_ocpsol(const auto & flatsol, auto && xl_fun, auto && ul_fun)
{
  using ocpsol_t = std::decay_t<decltype(flatsol)>;

  auto u_unflat = [ul_fun = std::forward<decltype(ul_fun)>(ul_fun),
                   usol   = flatsol.u](double t) -> U { return rplus(ul_fun(t), usol(t)); };

  auto x_unflat = [xl_fun = std::forward<decltype(xl_fun)>(xl_fun),
                   xsol   = flatsol.x](double t) -> X { return rplus(xl_fun(t), xsol(t)); };

  return OCPSolution<X, U, ocpsol_t::Nq, ocpsol_t::Ncr, ocpsol_t::Nce>{
    .t0         = flatsol.t0,
    .tf         = flatsol.tf,
    .Q          = flatsol.Q,
    .u          = std::move(u_unflat),
    .x          = std::move(x_unflat),
    .lambda_q   = flatsol.lambda_q,
    .lambda_ce  = flatsol.lambda_ce,
    .lambda_dyn = flatsol.lambda_dyn,
    .lambda_cr  = flatsol.lambda_cr,
  };
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__FLATTEN_OCP_HPP_
