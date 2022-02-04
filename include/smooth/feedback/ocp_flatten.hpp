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
  Xl xl;
  Ul ul;

  static constexpr auto Nx    = Dof<X>;
  static constexpr auto Nu    = Dof<U>;
  static constexpr auto Nouts = Nx;
  static constexpr auto Nvars = 1 + Nx + Nu;

  using E = Tangent<X>;
  using V = Tangent<U>;

  template<typename T>
  CastT<T, E> operator()(const T & t, const CastT<T, E> & e, const CastT<T, V> & v) const
  {
    // can not double-differentiate, so we hide derivative of f_new w.r.t. t
    const double tdbl          = static_cast<double>(t);
    const auto [xlval, dxlval] = diff::dr(xl, wrt(tdbl));
    const auto ulval           = ul(tdbl);

    const CastT<T, X> x = rplus(xlval.template cast<T>(), e);
    const CastT<T, U> u = rplus(ulval.template cast<T>(), v);

    return dr_expinv<CastT<T, X>>(e) * f.template operator()<T>(t, x, u)
         - dl_expinv<CastT<T, X>>(e) * dxlval.template cast<T>();
  }

  // Function: g_i = A_il f_l
  //
  // First derivative
  // d_k g_i = d_k(A_il) f_l + A_il d_k f_l
  //
  // Second derivative
  // d_j d_k g_i = d_jk(A_il) f_l + d_k(A_il) d_j f_l + d_j A_il d_k f_l + A_il d_jk f_l
  //

  Eigen::SparseMatrix<double> jacobian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>>)
  {
    const auto x       = rplus(xl(t), e);
    const auto u       = rplus(ul(t), v);
    const auto fval    = f(t, x, u);
    const auto & dfval = f.jacobian(t, x, u);
    const auto K       = d2r_expinv<X>(e);

    Eigen::SparseMatrix<double> ret(Nouts, Nvars);
    block_add(ret, 0, 0, dr_expinv<X>(e) * dfval);  // A_il d_k f_l = A_il * [Jf]_lk
    for (auto i = 0u; i < Dof<X>; ++i) {            // d_k(A_il) f_l
      block_add(ret, i, 1, fval.transpose() * K.template block<Dof<X>, Dof<X>>(0, i * Dof<X>));
    }
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>> &&
      diff::detail::diffable_order2<F, std::tuple<double, X, U>>)
  {
    // TODO incorrect, needs a lot of work
    const auto x = rplus(xl(t), e);
    const auto u = rplus(ul(t), v);
    // const auto fval    = f(t, x, u);
    const auto & dfval = f.jacobian(t, x, u);  // nx * (1 + nx + nu)

    const auto K = d2r_expinv<X>(e);

    Eigen::SparseMatrix<double> hess(Nvars, Nouts * Nvars);
    for (auto i = 0u; i < Dof<X>; ++i) {
      block_add(hess, 0, 0, dfval.transpose() * K.template block<Dof<X>, Dof<X>>(0, i * Dof<X>));
    }
    return hess;
  }
};

/**
 * @brief Flattening of inner function (t, x, u) -> Vector, and its derivatives.
 *
 * @note Ignores derivative dependence on t in xl(t)
 */
template<LieGroup X, Manifold U, std::size_t Nouts, typename F, typename Xl, typename Ul>
struct FlatInnerFun
{
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

  template<typename T>
  Eigen::Vector<T, Nouts>
  operator()(const T & t, const CastT<T, E> & e, const CastT<T, V> & v) const
  {
    return f.template operator()<T>(t, rplus(xl(t), e), rplus(ul(t), v));
  }

  Eigen::SparseMatrix<double> jacobian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>>)
  {
    const auto & df = f.jacobian(t, rplus(xl(t), e), rplus(ul(t), v));

    Eigen::SparseMatrix<double> ret(Nouts, Nvars);
    block_add(ret, 0, t_B, df.middleCols(t_B, 1));
    block_add(ret, 0, x_B, df.middleCols(x_B, Nx) * dr_exp<X>(e));
    block_add(ret, 0, u_B, df.middleCols(u_B, Nu) * dr_exp<U>(v));
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>> &&
      diff::detail::diffable_order2<F, std::tuple<double, X, U>>)
  {
    const auto x = rplus(xl(t), e);
    const auto u = rplus(ul(t), v);

    const auto & Jf = f.jacobian(t, x, u);
    const auto & Hf = f.hessian(t, x, u);

    const auto Je = dr_exp<X>(e);
    const auto Jv = dr_exp<U>(v);

    const auto He = d2r_exp<X>(e);
    const auto Hv = d2r_exp<U>(v);

    Eigen::SparseMatrix<double> ret(Nvars, Nouts * Nvars);

    for (auto no = 0u; no < Nouts; ++no) {  // for each output
      const auto b0 = Nvars * no;           // block index

      // clang-format off
      block_add(ret, t_B, b0 + t_B, Hf.block(t_B, b0 + t_B, 1,  1));       // tt
      block_add(ret, t_B, b0 + x_B, Hf.block(t_B, b0 + x_B, 1, Nx) * Je);  // te
      block_add(ret, t_B, b0 + u_B, Hf.block(t_B, b0 + u_B, 1, Nu) * Jv);  // tv

      block_add(ret, x_B, b0 + t_B, Je.transpose() * Hf.block(x_B, b0 + t_B, Nx,  1));       // et
      block_add(ret, x_B, b0 + x_B, Je.transpose() * Hf.block(x_B, b0 + x_B, Nx, Nx) * Je);  // ee
      block_add(ret, x_B, b0 + u_B, Je.transpose() * Hf.block(x_B, b0 + u_B, Nx, Nu) * Jv);  // ev

      block_add(ret, u_B, b0 + t_B, Jv.transpose() * Hf.block(u_B, b0 + t_B, Nu,  1));       // vt
      block_add(ret, u_B, b0 + x_B, Jv.transpose() * Hf.block(u_B, b0 + x_B, Nu, Nx) * Je);  // ve
      block_add(ret, u_B, b0 + u_B, Jv.transpose() * Hf.block(u_B, b0 + u_B, Nu, Nu) * Jv);  // vv

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
  Xl xl;
  Ul ul;

  static constexpr auto Nx    = Dof<X>;
  static constexpr auto Nvars = 1 + 2 * Nx + Nq;

  using E = Tangent<X>;
  using Q = Eigen::Vector<Scalar<X>, Nq>;

  static constexpr auto tf_B = 0;
  static constexpr auto x0_B = tf_B + 1;
  static constexpr auto xf_B = x0_B + Nx;
  static constexpr auto q_B  = xf_B + Nx;

  template<typename T>
  auto operator()(
    const T & tf, const CastT<T, E> & e0, const CastT<T, E> & ef, const CastT<T, Q> & q) const
  {
    return f.template operator()<T>(tf, rplus(xl(T(0.)), e0), rplus(xl(tf), ef), q);
  }

  Eigen::SparseMatrix<double> jacobian(double tf, const E & e0, const E & ef, const Q & q) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, X, Q>>)
  {
    const auto xl_0            = xl(0.);
    const auto [xl_tf, dxl_tf] = diff::dr(xl, wrt(tf));

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

  Eigen::SparseMatrix<double> hessian(double tf, const E & e0, const E & ef, const Q & q) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, X, Q>> &&
      diff::detail::diffable_order2<F, std::tuple<double, X, X, Q>>)
  {
    const auto x0 = rplus(xl(0.), e0);
    const auto xf = rplus(xl(tf), ef);

    // derivatives of f
    const auto & Jf = f.jacobian(tf, x0, xf, q);  // Nouts x Nx
    const auto & Hf = f.hessian(tf, x0, xf, q);   // Nx x (Nouts * Nx)

    // derivatives of (x + e) w.r.t. e
    const auto Jp0 = dr_exp<X>(e0);   // Nx x Nx
    const auto Hp0 = d2r_exp<X>(e0);  // Nx x (Nx * Nx)
    const auto Jpf = dr_exp<X>(ef);   // Nx x Nx
    const auto Hpf = d2r_exp<X>(ef);  // Nx x (Nx * Nx)

    Eigen::SparseMatrix<double> ret(Nvars, Nouts * Nvars);

    for (auto no = 0u; no < Nouts; ++no) {  // for each output
      const auto b0 = Nvars * no;           // block index

      // terms involving second derivatives w.r.t. f

      // clang-format off
      block_add(ret, tf_B, b0 + tf_B, Hf.block(tf_B, b0 + tf_B, 1,  1));        // tftf
      block_add(ret, tf_B, b0 + x0_B, Hf.block(tf_B, b0 + x0_B, 1, Nx) * Jp0);  // tfx0
      block_add(ret, tf_B, b0 + xf_B, Hf.block(tf_B, b0 + xf_B, 1, Nx) * Jpf);  // tfxf
      block_add(ret, tf_B, b0 +  q_B, Hf.block(tf_B, b0 +  q_B, 1, Nq));        // tfq

      block_add(ret, x0_B, b0 + tf_B, Jp0.transpose() * Hf.block(x0_B, b0 + tf_B, Nx,  1));        // x0tf
      block_add(ret, x0_B, b0 + x0_B, Jp0.transpose() * Hf.block(x0_B, b0 + x0_B, Nx, Nx) * Jp0);  // x0x0
      block_add(ret, x0_B, b0 + xf_B, Jp0.transpose() * Hf.block(x0_B, b0 + xf_B, Nx, Nx) * Jpf);  // x0xf
      block_add(ret, x0_B, b0 +  q_B, Jp0.transpose() * Hf.block(x0_B, b0 +  q_B, Nx, Nq));        // x0q

      block_add(ret, xf_B, b0 + tf_B, Jpf.transpose() * Hf.block(xf_B, b0 + tf_B, Nx,  1));        // xftf
      block_add(ret, xf_B, b0 + x0_B, Jpf.transpose() * Hf.block(xf_B, b0 + x0_B, Nx, Nx) * Jp0);  // xfx0
      block_add(ret, xf_B, b0 + xf_B, Jpf.transpose() * Hf.block(xf_B, b0 + xf_B, Nx, Nx) * Jpf);  // xfxf
      block_add(ret, xf_B, b0 +  q_B, Jpf.transpose() * Hf.block(xf_B, b0 +  q_B, Nx, Nq));        // xfq

      block_add(ret,  q_B, b0 + tf_B, Hf.block(q_B, b0 + tf_B, Nq,  1));        // qtf
      block_add(ret,  q_B, b0 + x0_B, Hf.block(q_B, b0 + x0_B, Nq, Nx) * Jp0);  // qx0
      block_add(ret,  q_B, b0 + xf_B, Hf.block(q_B, b0 + xf_B, Nq, Nx) * Jpf);  // qxf
      block_add(ret,  q_B, b0 +  q_B, Hf.block(q_B, b0 +  q_B, Nq, Nq));        // qq
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
auto flatten_ocp(const OCPType auto & ocp, auto && xl, auto && ul)
{
  using ocp_t = std::decay_t<decltype(ocp)>;
  using X     = typename ocp_t::X;
  using U     = typename ocp_t::U;
  using Xl    = decltype(xl);
  using Ul    = decltype(ul);

  static constexpr auto Nq = ocp_t::Nq;

  detail::FlatEndptFun<X, U, Nq, 1, decltype(ocp.theta), Xl, Ul> flat_theta{ocp.theta, xl, ul};
  detail::FlatDyn<X, U, decltype(ocp.f), Xl, Ul> flat_f{ocp.f, xl, ul};
  detail::FlatInnerFun<X, U, Nq, decltype(ocp.g), Xl, Ul> flat_g{ocp.g, xl, ul};
  detail::FlatInnerFun<X, U, ocp_t::Ncr, decltype(ocp.cr), Xl, Ul> flat_cr{ocp.cr, xl, ul};
  detail::FlatEndptFun<X, U, Nq, ocp_t::Nce, decltype(ocp.ce), Xl, Ul> flat_ce{ocp.ce, xl, ul};

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
