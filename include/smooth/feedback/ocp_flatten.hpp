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
 * @brief Formulate optimal control problem as a nonlinear program
 */

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <smooth/algo/hessian.hpp>
#include <smooth/diff.hpp>
#include <smooth/feedback/utils/sparse.hpp>

#include "ocp.hpp"

namespace smooth::feedback {

namespace detail {
/**
 * @brief Flattening of dynamics function (t, x, u) -> Tangent, and its derivatives.
 *
 * @note Ignores derivative dependence on t in xl(t)
 * @note Derivatives very much WIP
 */
template<LieGroup X, Manifold U, typename F, typename Xl, typename Ul>
struct FlatDyn
{
  F f;
  Xl xl_fun;
  Ul ul_fun;

  static constexpr auto Nx    = Dof<X>;
  static constexpr auto Nu    = Dof<U>;
  static constexpr auto Nouts = Nx;
  static constexpr auto Nvars = 1 + Nx + Nu;

  template<typename T>
  Eigen::Vector<T, Nx>
  operator()(const T & t, const Eigen::Vector<T, Nx> & e, const Eigen::Vector<T, Nu> & v) const
  {
    // can not double-differentiate, so we hide derivative of f_new w.r.t. t
    const double tdbl    = static_cast<double>(t);
    const auto [xl, dxl] = diff::dr(xl_fun, wrt(tdbl));
    const auto ul        = ul_fun(tdbl);

    const CastT<T, X> x = rplus(xl.template cast<T>(), e);
    const CastT<T, U> u = rplus(ul.template cast<T>(), v);

    return dr_expinv<CastT<T, X>>(e) * f.template operator()<T>(t, x, u)
         - dl_expinv<CastT<T, X>>(e) * dxl.template cast<T>();
  }

  Eigen::SparseMatrix<double>
  jacobian(double t, const Eigen::Vector<double, Nx> & e, const Eigen::Vector<double, Nu> & v)
    // clang-format off
        requires(
          diff::detail::diffable_order1<F, std::tuple<double, X, U>>
        )
  // clang-format on
  {
    // TODO compute this one
    const auto x    = rplus(xl_fun(t), e);
    const auto u    = rplus(ul_fun(t), v);
    const auto & df = f.jacobian(t, x, u);

    Eigen::SparseMatrix<double> j(Nouts, Nvars);
    j = df;
    return j;
  }

  Eigen::SparseMatrix<double>
  hessian(double t, const Eigen::Vector<double, Nx> & e, const Eigen::Vector<double, Nu> & v)
    // clang-format off
      requires(
        diff::detail::diffable_order1<F, std::tuple<double, X, U>>
        && diff::detail::diffable_order2<F, std::tuple<double, X, U>>
      )
  // clang-format on
  {
    // TODO compute this one
    const auto x     = rplus(xl_fun(t), e);
    const auto u     = rplus(ul_fun(t), v);
    const auto & d2f = f.hessian(t, x, u);

    Eigen::SparseMatrix<double> h(Nvars, Nouts * Nvars);
    h = d2f;
    return h;
  }
};

/**
 * @brief Flattening of inner function (t, x, u) -> Vector, and its derivatives.
 *
 * @note Ignores derivative dependence on t in xl(t)
 * @note Derivatives very much WIP
 */
template<LieGroup X, Manifold U, std::size_t Nouts, typename F, typename Xl, typename Ul>
struct FlatInnerFun
{
  F f;
  Xl xl_fun;
  Ul ul_fun;

  static constexpr auto Nx    = Dof<X>;
  static constexpr auto Nu    = Dof<U>;
  static constexpr auto Nvars = 1 + Nx + Nu;

  static constexpr auto t_B = 0;
  static constexpr auto x_B = t_B + 1;
  static constexpr auto u_B = x_B + Nx;

  template<typename T>
  Eigen::Vector<T, Nouts>
  operator()(const T & t, const Eigen::Vector<T, Nx> & e, const Eigen::Vector<T, Nu> & v) const
  {
    const auto x = rplus(xl_fun(t), e);
    const auto u = rplus(ul_fun(t), v);
    return f.template operator()<T>(t, x, u);
  }

  Eigen::SparseMatrix<double>
  jacobian(double t, const Eigen::Vector<double, Nx> & e, const Eigen::Vector<double, Nu> & v)
    // clang-format off
    requires(
      diff::detail::diffable_order1<F, std::tuple<double, X, U>>
    )
  // clang-format on
  {
    const auto x    = rplus(xl_fun(t), e);
    const auto u    = rplus(ul_fun(t), v);
    const auto & df = f.jacobian(t, x, u);

    Eigen::SparseMatrix<double> ret(Nouts, Nvars);
    block_add(ret, 0, t_B, df.middleCols(t_B, 1));
    block_add(ret, 0, x_B, df.middleCols(x_B, Nx) * dr_exp<X>(e));
    block_add(ret, 0, u_B, df.middleCols(u_B, Nu) * dr_exp<U>(v));
    return ret;
  }

  Eigen::SparseMatrix<double>
  hessian(double t, const Eigen::Vector<double, Nx> & e, const Eigen::Vector<double, Nu> & v)
    // clang-format off
    requires(
      diff::detail::diffable_order1<F, std::tuple<double, X, U>>
      && diff::detail::diffable_order2<F, std::tuple<double, X, U>>
    )
  // clang-format on
  {
    const auto x = rplus(xl_fun(t), e);
    const auto u = rplus(ul_fun(t), v);

    const auto & Jf = f.jacobian(t, x, u);
    const auto & Hf = f.hessian(t, x, u);

    const auto Je = dr_exp<X>(e);
    const auto Jv = dr_exp<U>(v);

    const auto He = hessian_rplus<X>(e);
    const auto Hv = hessian_rplus<U>(v);

    Eigen::SparseMatrix<double> ret(Nvars, Nouts * Nvars);

    for (auto no = 0u; no < Nouts; ++no) {  // for each output
      const auto b0 = Nvars * no;           // block index

      // clang-format off
      // tt
      block_add(ret, t_B, b0 + t_B, Hf.block(t_B, b0 + t_B, 1,  1));
      // te
      block_add(ret, t_B, b0 + x_B, Hf.block(t_B, b0 + x_B, 1, Nx) * Je);
      // tv
      block_add(ret, t_B, b0 + u_B, Hf.block(t_B, b0 + u_B, 1, Nu) * Jv);

      // If g = f(x + e), then
      // Hg_ijk = Hf_imk Jp_mj Jp_lk + Jf_il Hp_ljk

      // et
      block_add(ret, x_B, b0 + t_B, Je.transpose() * Hf.block(x_B, b0 + t_B, Nx, 1));
      // ee
      block_add(ret, x_B, b0 + x_B, Je.transpose() * Hf.block(x_B, b0 + x_B, Nx, Nx) * Je);
      // ev
      block_add(ret, x_B, b0 + u_B, Je.transpose() * Hf.block(x_B, b0 + u_B, Nx, Nu) * Jv);

      // vt
      block_add(ret, u_B, b0 + t_B, Jv.transpose() * Hf.block(u_B, b0 + t_B, Nu, 1));
      // ve
      block_add(ret, u_B, b0 + x_B, Jv.transpose() * Hf.block(u_B, b0 + x_B, Nu, Nx) * Je);
      // vv
      block_add(ret, u_B, b0 + u_B, Jv.transpose() * Hf.block(u_B, b0 + u_B, Nu, Nu) * Jv);

      // second derivatives w.r.t. x+e
      for (auto nx = 0u; nx < Nx; ++nx) {
        // TODO: skip if Jf isn't filled
        block_add(ret, x_B, b0 + x_B, Jf.coeff(no, x_B + nx) * He.block(0, nx * Nx, Nx, Nx));
      }

      // second derivatives w.r.t. u+v
      for (auto nu = 0u; nu < Nu; ++nu) {
        // TODO: skip if Jf isn't filled
        block_add(ret, u_B, b0 + u_B, Jf.coeff(no, u_B + nu) * Hv.block(0, nu * Nu, Nu, Nu));
      }
      // clang-format on
    }
    return ret;
  }
};

/**
 * @brief Flattening of endpoint function (tf, x0, xf, q) -> Vector, and its derivatives.
 *
 * @note Ignores dependence of tf in (xl(tf) + ef)
 * @note Currently does not support Hessians that are not block-diagonal...
 */
template<
  LieGroup X,
  Manifold U,
  std::size_t Nq,
  std::size_t Nouts,
  typename F,
  typename Xl,
  typename Ul>
struct FlatEndptFun
{
  F f;
  Xl xl_fun;
  Ul ul_fun;

  static constexpr auto Nx    = Dof<X>;
  static constexpr auto Nvars = 1 + 2 * Nx + Nq;

  static constexpr auto tf_B = 0;
  static constexpr auto x0_B = tf_B + 1;
  static constexpr auto xf_B = x0_B + Nx;
  static constexpr auto q_B  = xf_B + Nx;

  template<typename T>
  auto operator()(
    const T & tf,
    const Eigen::Vector<T, Nx> & e0,
    const Eigen::Vector<T, Nx> & ef,
    const Eigen::Vector<T, Nq> & q) const
  {
    return f.template operator()<T>(tf, rplus(xl_fun(T(0.)), e0), rplus(xl_fun(tf), ef), q);
  }

  Eigen::SparseMatrix<double> jacobian(
    double tf,
    const Eigen::Vector<double, Nx> & e0,
    const Eigen::Vector<double, Nx> & ef,
    const Eigen::Vector<double, Nq> & q)
    // clang-format off
        requires(
          diff::detail::diffable_order1<F, std::tuple<double, X, X, Eigen::Vector<double, Nq>>>
        )
  // clang-format on
  {
    const auto xl_0            = xl_fun(0.);
    const auto [xl_tf, dxl_tf] = diff::dr(xl_fun, wrt(tf));

    const auto dx0_e0 = dr_exp<X>(e0);
    const auto dxf_ef = dr_exp<X>(ef);
    // const auto dxf_tf = Ad(smooth::exp<X>(-ef)) * dxl_tf;

    const auto & dtheta = f.jacobian(tf, rplus(xl_0, e0), rplus(xl_tf, ef), q);

    Eigen::SparseMatrix<double> j(Nouts, Nvars);
    // clang-format off
    block_add(j, 0, tf_B, dtheta.middleCols(tf_B, 1));
    // block_add(j, 0, tf_B, dtheta.middleCols(xf_B, Nx) * dxf_tf);  // ignore this dependence for now..
    block_add(j, 0, x0_B, dtheta.middleCols(x0_B, Nx) * dx0_e0);
    block_add(j, 0, xf_B, dtheta.middleCols(xf_B, Nx) * dxf_ef);
    block_add(j, 0,  q_B, dtheta.middleCols( q_B, Nq));
    // clang-format on
    return j;
  }

  Eigen::SparseMatrix<double> hessian(
    double tf,
    const Eigen::Vector<double, Nx> & e0,
    const Eigen::Vector<double, Nx> & ef,
    const Eigen::Vector<double, Nq> & q)
    // clang-format off
      requires(
        diff::detail::diffable_order1<F, std::tuple<double, X, X, Eigen::Vector<double, Nq>>>
        && diff::detail::diffable_order2<F, std::tuple<double, X, X, Eigen::Vector<double, Nq>>>
      )
  // clang-format on
  {
    const auto x0 = rplus(xl_fun(0.), e0);
    const auto xf = rplus(xl_fun(tf), ef);

    // derivatives of f
    const auto & Jf = f.jacobian(tf, x0, xf, q);  // Nouts x Nx
    const auto & Hf = f.hessian(tf, x0, xf, q);   // Nx x (Nouts * Nx)

    // derivatives of (x + e) w.r.t. e
    const auto Jp0 = dr_exp<X>(e0);         // Nx x Nx
    const auto Hp0 = hessian_rplus<X>(e0);  // Nx x (Nx * Nx)
    const auto Jpf = dr_exp<X>(ef);         // Nx x Nx
    const auto Hpf = hessian_rplus<X>(ef);  // Nx x (Nx * Nx)

    Eigen::SparseMatrix<double> ret(Nvars, Nouts * Nvars);

    for (auto no = 0u; no < Nouts; ++no) {  // for each output
      const auto b0 = Nvars * no;           // block index

      // terms involving second derivatives w.r.t. f

      // clang-format off
      // wrt tf
      block_add(ret, tf_B, b0 + tf_B, Hf.block(tf_B, b0 + tf_B, 1,  1));
      block_add(ret, tf_B, b0 + x0_B, Hf.block(tf_B, b0 + x0_B, 1, Nx) * Jp0);
      block_add(ret, tf_B, b0 + xf_B, Hf.block(tf_B, b0 + xf_B, 1, Nx) * Jpf);
      block_add(ret, tf_B, b0 +  q_B, Hf.block(tf_B, b0 +  q_B, 1, Nq));

      // wrt x0
      block_add(ret, x0_B, b0 + tf_B, Jp0.transpose() * Hf.block(x0_B, b0 + tf_B, Nx,  1));
      block_add(ret, x0_B, b0 + x0_B, Jp0.transpose() * Hf.block(x0_B, b0 + x0_B, Nx, Nx) * Jp0);
      block_add(ret, x0_B, b0 + xf_B, Jp0.transpose() * Hf.block(x0_B, b0 + xf_B, Nx, Nx) * Jpf);
      block_add(ret, x0_B, b0 +  q_B, Jp0.transpose() * Hf.block(x0_B, b0 +  q_B, Nx, Nq));

      // wrt xf
      block_add(ret, xf_B, b0 + tf_B, Jpf.transpose() * Hf.block(xf_B, b0 + tf_B, Nx,  1));
      block_add(ret, xf_B, b0 + x0_B, Jpf.transpose() * Hf.block(xf_B, b0 + x0_B, Nx, Nx) * Jp0);
      block_add(ret, xf_B, b0 + xf_B, Jpf.transpose() * Hf.block(xf_B, b0 + xf_B, Nx, Nx) * Jpf);
      block_add(ret, xf_B, b0 +  q_B, Jpf.transpose() * Hf.block(xf_B, b0 +  q_B, Nx, Nq));

      // wrt q
      block_add(ret,  q_B, b0 + tf_B, Hf.block(q_B, b0 + tf_B, Nq,  1));
      block_add(ret,  q_B, b0 + x0_B, Hf.block(q_B, b0 + x0_B, Nq, Nx) * Jp0);
      block_add(ret,  q_B, b0 + xf_B, Hf.block(q_B, b0 + xf_B, Nq, Nx) * Jpf);
      block_add(ret,  q_B, b0 +  q_B, Hf.block(q_B, b0 +  q_B, Nq, Nq));
      // clang-format on

      // terms involving second derivatives w.r.t. e0 and ef

      for (auto nx = 0u; nx < Nx; ++nx) {
        // TODO: skip if Jf isn't filled
        block_add(ret, x0_B, b0 + x0_B, Jf.coeff(no, x0_B + nx) * Hp0.block(0, nx * Nx, Nx, Nx));
        // TODO: skip if Jf isn't filled
        block_add(ret, xf_B, b0 + xf_B, Jf.coeff(no, xf_B + nx) * Hpf.block(0, nx * Nx, Nx, Nx));
      }
    }

    return ret;
  }
};

}  // namespace detail

/**
 * @brief Flatten a LieGroup OCP by defining it in the tangent space around a trajectory.
 *
 * @param ocp OCPType defined on a LieGroup
 * @param xl nominal state trajectory
 * @param ul nominal state trajectory
 *
 * @return FlatOCPType in variables (xe, ue) obtained via variables change x = xl ⊕ xe, u = ul ⊕
 * ue,
 */
auto flatten_ocp2(const OCPType auto & ocp, auto && xl, auto && ul)
{
  using ocp_t = std::decay_t<decltype(ocp)>;
  using X     = typename ocp_t::X;
  using U     = typename ocp_t::U;
  using xl_t  = decltype(xl);
  using ul_t  = decltype(ul);

  static constexpr auto Nq = ocp_t::Nq;

  detail::FlatEndptFun<X, U, Nq, 1, decltype(ocp.theta), xl_t, ul_t> flat_theta{ocp.theta, xl, ul};
  detail::FlatDyn<X, U, decltype(ocp.f), xl_t, ul_t> flat_f{ocp.f, xl, ul};
  detail::FlatInnerFun<X, U, Nq, decltype(ocp.g), xl_t, ul_t> flat_g{ocp.g, xl, ul};
  detail::FlatInnerFun<X, U, ocp_t::Ncr, decltype(ocp.cr), xl_t, ul_t> flat_cr{ocp.cr, xl, ul};
  detail::FlatEndptFun<X, U, Nq, ocp_t::Nce, decltype(ocp.ce), xl_t, ul_t> flat_ce{ocp.ce, xl, ul};

  return OCP<
    Eigen::Vector<double, Dof<X>>,
    Eigen::Vector<double, Dof<U>>,
    decltype(flat_theta),
    decltype(flat_f),
    decltype(flat_g),
    decltype(flat_cr),
    decltype(flat_ce)>{
    .theta = std::move(flat_theta),
    .f     = std::move(flat_f),
    .g     = std::move(flat_g),
    .cr    = std::move(flat_cr),
    .crl   = ocp.crl,
    .cru   = ocp.cru,
    .ce    = std::move(flat_ce),
    .cel   = ocp.cel,
    .ceu   = ocp.ceu,
  };
}

}  // namespace smooth::feedback

#endif  // SMOOTH__FEEDBACK__FLATTEN_OCP_HPP_
