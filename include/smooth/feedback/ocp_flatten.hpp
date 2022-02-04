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

  // Function: drexp_e f - dlexp_e dxl
  //   = \sum Bn (-1)^n / n!  ad_a^n f - \sum Bn / n!  ad_a^n dxl
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

  // First derivative
  //     \sum Bn (-1)^n / n! dr (ad_a^n f)_a  - \sum Bn / n! dr ( ad_a^n dxl )_a
  Eigen::SparseMatrix<double> jacobian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>>)
  {
    const double tdbl          = static_cast<double>(t);
    const auto [xlval, dxlval] = diff::dr(xl, wrt(tdbl));

    const auto x             = rplus(xlval, e);
    const auto u             = rplus(ul(t), v);
    const auto & dfval       = f.jacobian(t, x, u);  // nx * (1 + nx + nu)
    const TangentMap<X> ad_e = ad<X>(e);

    Eigen::SparseMatrix<double> ret(Nouts, Nvars);

    // know that d (ad_a b)_a = -ad_b + ad_a db_a,
    // so d (ad_a^n b)_a = -ad_{ad_a^{n-1} b} + ad_a d (ad_a^{n-1} b)_a

    double coeff1                            = 1;           // hold (-1)^i / i!
    Tangent<X> g1i                           = f(t, x, u);  // (ad_a)^i * f
    Eigen::Matrix<double, Nouts, Nvars> dg1i = dfval;       // derivative of g1i w.r.t. e
    dg1i.middleCols(1, Nx) *= dr_exp<X>(e);
    dg1i.middleCols(1 + Nx, Nu) *= dr_exp<U>(v);

    double coeff2  = 1;                        // hold 1 / i!
    Tangent<X> g2i = dxlval;                   // (ad_a)^i * dxl
    Eigen::Matrix<double, Nouts, Nvars> dg2i;  // derivative of g1i w.r.t. e
    dg2i.setZero();

    for (auto iter = 0u; iter < std::tuple_size_v<decltype(smooth::detail::kBn)>; ++iter) {
      if (smooth::detail::kBn[iter] != 0) {
        block_add(ret, 0, 0, (smooth::detail::kBn[iter] * coeff1) * dg1i);
        block_add(ret, 0, 0, (smooth::detail::kBn[iter] * coeff2) * dg2i, -1);
      }
      dg1i.applyOnTheLeft(ad_e);
      dg1i.middleCols(1, Nx) -= ad<X>(g1i);
      g1i.applyOnTheLeft(ad_e);

      dg2i.applyOnTheLeft(ad_e);
      dg2i.middleCols(1, Nx) -= ad<X>(g2i);
      g2i.applyOnTheLeft(ad_e);

      coeff1 *= (-1.) / (iter + 1);
      coeff2 *= 1. / (iter + 1);
    }

    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>> &&
      diff::detail::diffable_order2<F, std::tuple<double, X, U>>)
  {
    using smooth::detail::kBn;

    const double tdbl          = static_cast<double>(t);
    const auto [xlval, dxlval] = diff::dr(xl, wrt(tdbl));

    const auto x             = rplus(xlval, e);
    const auto u             = rplus(ul(t), v);
    const auto & dfval       = f.jacobian(t, x, u);  // nx x (1 + nx + nu)
    const auto & d2fval      = f.hessian(t, x, u);   // (1 + nx + nu) x (nx * (1 + nx + nu))
    const TangentMap<X> ad_e = ad<X>(e);

    Eigen::SparseMatrix<double> ret(Nvars, Nouts * Nvars);

    // w.r.t. t

    block_add(ret, 0, 0, dfval.block(0, 0, 1, 1));

    // w.r.t. x

    double coef      = 1;                                       // hold (-1)^i / i!
    Tangent<X> vi    = f(t, x, u);                              // (ad_a)^i * f
    TangentMap<X> ji = dfval.middleCols(1, Nx) * dr_exp<X>(e);  // first derivative of vi w.r.t. e
    smooth::detail::hess_t<X> hi;                               // second derivative of vi w.r.t. e
    for (auto i = 0u; i < Nx; ++i) {
      hi.middleCols(i * Nx, Nx) = d2fval.middleCols(i * (1 + Nx + Nu) + 1, Nx);
    }

    for (auto iter = 0u; iter < std::tuple_size_v<decltype(smooth::detail::kBn)>; ++iter) {
      // add to result
      if (kBn[iter] != 0) {
        for (auto i = 0u; i < Nx; ++i) {
          block_add(ret, 1, i * (1 + Nx + Nu) + 1, (kBn[iter] * coef) * hi.middleCols(i * Nx, Nx));
        }
      }

      // update hi
      smooth::detail::hess_t<X> hip = smooth::detail::hess_t<X>::Zero();
      for (auto k = 0u; k < Nx; ++k) {
        Eigen::Matrix<double, Nx, Nx> ek_gens;
        for (auto l = 0u; l < Nx; ++l) {
          ek_gens.row(l) = Eigen::Vector<double, Nx>::Unit(k).transpose()
                         * smooth::detail::algebra_generators<X>[l];
        }

        hip.middleCols(k * Nx, Nx) += ek_gens * ji;
        hip.middleCols(k * Nx, Nx) -= ji.transpose() * ek_gens;

        for (auto l = 0u; l < Nx; ++l) {
          hip.middleCols(k * Nx, Nx) += ad_e(k, l) * hi.middleCols(l * Nx, Nx);
        }
      }

      hi = hip;

      // update ji
      ji.applyOnTheLeft(ad_e);
      ji -= ad<X>(vi);

      // update vi
      vi.applyOnTheLeft(ad_e);

      coef *= (-1.) / (iter + 1);
    }

    // w.r.t. u
    block_add(ret, 1 + Nx, 1 + Nx, d2fval.block(1 + Nx, 1 + Nx, Nu, Nu));

    return ret;
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
    Tangent<X>,
    Tangent<U>,
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
