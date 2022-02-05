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
    using Jt = Eigen::Matrix<double, Nouts, Nvars>;

    const double tdbl          = static_cast<double>(t);
    const auto [xlval, dxlval] = diff::dr(xl, wrt(tdbl));

    const auto x             = rplus(xlval, e);
    const auto u             = rplus(ul(t), v);
    const auto & Jf          = f.jacobian(t, x, u);  // nx * (1 + nx + nu)
    const TangentMap<X> ad_e = ad<X>(e);

    Eigen::SparseMatrix<double> ret(Nouts, Nvars);

    double coef    = 1;           // hold (-1)^i / i!
    Tangent<X> vi1 = f(t, x, u);  // (ad_a)^i * f
    Jt ji1         = Jf;          // derivative of vi1 w.r.t. e
    ji1.middleCols(1, Nx) *= dr_exp<X>(e);
    ji1.middleCols(1 + Nx, Nu) *= dr_exp<U>(v);

    Tangent<X> vi2 = dxlval;  // (ad_a)^i * dxl
    Jt ji2;                   // derivative of vi1 w.r.t. e
    ji2.setZero();

    for (auto iter = 0u; iter < std::tuple_size_v<decltype(smooth::detail::kBn)>; ++iter) {
      if (smooth::detail::kBn[iter] != 0) {
        block_add(ret, 0, 0, ji1, smooth::detail::kBn[iter] * coef);
        block_add(ret, 0, 0, ji2, -1 * smooth::detail::kBn[iter] * std::abs(coef));
      }

      // update ji1, vi1
      ji1.applyOnTheLeft(ad_e);
      ji1.middleCols(1, Nx) -= ad<X>(vi1);
      vi1.applyOnTheLeft(ad_e);

      // update ji2, vi2
      ji2.applyOnTheLeft(ad_e);
      ji2.middleCols(1, Nx) -= ad<X>(vi2);
      vi2.applyOnTheLeft(ad_e);

      coef *= (-1.) / (iter + 1);
    }

    return ret;
  }

  // Second derivative
  //    \sum Bn (-1)^n / n! d2r (ad_a^n f)_aa - \sum Bn / n! d2r(ad_a^n dxl)_aa
  Eigen::SparseMatrix<double> hessian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>> &&
      diff::detail::diffable_order2<F, std::tuple<double, X, U>>)
  {
    using Jt = Eigen::Matrix<double, Nouts, Nvars>;
    using Ht = Eigen::Matrix<double, Nvars, Nouts * Nvars>;

    const double tdbl          = static_cast<double>(t);
    const auto [xlval, dxlval] = diff::dr(xl, wrt(tdbl));

    const auto x             = rplus(xlval, e);
    const auto u             = rplus(ul(t), v);
    const auto He            = d2r_exp<X>(e);
    const auto Hv            = d2r_exp<U>(v);
    const auto & Jf          = f.jacobian(t, x, u);  // nx x (1 + nx + nu)
    const auto & Hf          = f.hessian(t, x, u);   // (1 + nx + nu) x (nx * (1 + nx + nu))
    const TangentMap<X> ad_e = ad<X>(e);

    Eigen::SparseMatrix<double> Jblocks(Nvars, Nvars);
    block_add_identity(Jblocks, 0, 0, 1);
    block_add(Jblocks, 1, 1, dr_exp<X>(e));
    block_add(Jblocks, 1 + Nx, 1 + Nx, dr_exp<U>(v));

    Eigen::SparseMatrix<double> ret(Nvars, Nouts * Nvars);

    double coef    = 1;             // hold (-1)^i / i!
    Tangent<X> vi1 = f(t, x, u);    // (ad_a)^i * f
    Jt ji1         = Jf * Jblocks;  // dr (vi1)_{t, e, v}
    Ht hi1;                         // dr (ji1' e_k)_{t, e, v}
    for (auto no = 0u; no < Nx; ++no) {
      const auto b0 = no * Nvars;

      // second derivatives w.r.t. f
      hi1.middleCols(b0, Nvars) = Jblocks.transpose() * Hf.middleCols(b0, Nvars) * Jblocks;

      // second derivatives w.r.t. e
      for (auto nx = 0u; nx < Nx; ++nx) {
        hi1.middleCols(b0, Nvars).block(1, 1, Nx, Nx) +=
          Jf.coeff(no, 1 + nx) * He.block(0, nx * Nx, Nx, Nx);
      }
      // second derivatives w.r.t. v
      for (auto nu = 0u; nu < Nu; ++nu) {
        hi1.middleCols(b0, Nvars).block(1 + Nx, 1 + Nx, Nu, Nu) +=
          Jf.coeff(no, 1 + Nx + nu) * Hv.block(0, nu * Nu, Nu, Nu);
      }
    };

    Tangent<X> vi2 = dxlval;      // (ad_a)^i * dxl
    Jt ji2         = Jt::Zero();  // dr (vi2)_{t, e, v}
    Ht hi2         = Ht::Zero();  // dr (ji2' e_k)_{t, e, v}

    for (auto iter = 0u; iter < std::tuple_size_v<decltype(smooth::detail::kBn)>; ++iter) {
      // add to result
      if (smooth::detail::kBn[iter] != 0) {
        block_add(ret, 0, 0, hi1, smooth::detail::kBn[iter] * coef);
        block_add(ret, 0, 0, hi2, -1 * smooth::detail::kBn[iter] * std::abs(coef));
      }

      // update hi1 and hi2
      Ht hi1p = Ht::Zero(), hi2p = Ht::Zero();
      for (auto k = 0u; k < Nx; ++k) {
        Eigen::Matrix<double, Nx, Nx> ek_gens;
        for (auto l = 0u; l < Nx; ++l) {
          // ek_gens[l][j] = generator_l[k][j]
          ek_gens.row(l) = smooth::detail::algebra_generators<X>[l].row(k);

          hi1p.middleCols(k * Nvars, Nvars) += ad_e(k, l) * hi1.middleCols(l * Nvars, Nvars);
          hi2p.middleCols(k * Nvars, Nvars) += ad_e(k, l) * hi2.middleCols(l * Nvars, Nvars);
        }

        hi1p.middleCols(k * Nvars, Nvars).middleRows(1, Nx) += ek_gens * ji1;
        hi1p.middleCols(k * Nvars, Nvars).middleCols(1, Nx) -= ji1.transpose() * ek_gens;

        hi2p.middleCols(k * Nvars, Nvars).middleRows(1, Nx) += ek_gens * ji2;
        hi2p.middleCols(k * Nvars, Nvars).middleCols(1, Nx) -= ji2.transpose() * ek_gens;
      }
      hi1 = hi1p;
      hi2 = hi2p;

      // update ji1 and ji2
      ji1.applyOnTheLeft(ad_e);
      ji1.middleCols(1, Nx) -= ad<X>(vi1);
      ji2.applyOnTheLeft(ad_e);
      ji2.middleCols(1, Nx) -= ad<X>(vi2);

      // update vi1 and vi2
      vi1.applyOnTheLeft(ad_e);
      vi2.applyOnTheLeft(ad_e);

      coef *= (-1.) / (iter + 1);
    }

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

    Eigen::SparseMatrix<double> Jblocks(Nvars, Nvars);
    block_add_identity(Jblocks, 0, 0, 1);
    block_add(Jblocks, 1, 1, dr_exp<X>(e));
    block_add(Jblocks, 1 + Nx, 1 + Nx, dr_exp<U>(v));

    return df * Jblocks;
  }

  Eigen::SparseMatrix<double> hessian(double t, const E & e, const V & v) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, U>> &&
      diff::detail::diffable_order2<F, std::tuple<double, X, U>>)
  {
    const auto x = rplus(xl(t), e);
    const auto u = rplus(ul(t), v);

    const auto & Jf = f.jacobian(t, x, u);
    const auto & Hf = f.hessian(t, x, u);

    const auto He = d2r_exp<X>(e);
    const auto Hv = d2r_exp<U>(v);

    Eigen::SparseMatrix<double> Jblocks(Nvars, Nvars);
    block_add_identity(Jblocks, 0, 0, 1);
    block_add(Jblocks, 1, 1, dr_exp<X>(e));
    block_add(Jblocks, 1 + Nx, 1 + Nx, dr_exp<U>(v));

    Eigen::SparseMatrix<double> ret(Nvars, Nouts * Nvars);

    for (auto no = 0u; no < Nouts; ++no) {  // for each output
      const auto b0 = Nvars * no;           // block index

      // second derivatives w.r.t. f
      block_add(ret, 0, b0, Jblocks.transpose() * Hf.block(0, b0, Nvars, Nvars) * Jblocks);

      // second derivatives w.r.t. e
      for (auto nx = 0u; nx < Nx; ++nx) {
        // TODO: skip if Jf isn't filled
        block_add(ret, x_B, b0 + x_B, Jf.coeff(no, x_B + nx) * He.block(0, nx * Nx, Nx, Nx));
      }

      // second derivatives w.r.t. v
      for (auto nu = 0u; nu < Nu; ++nu) {
        // TODO: skip if Jf isn't filled
        block_add(ret, u_B, b0 + u_B, Jf.coeff(no, u_B + nu) * Hv.block(0, nu * Nu, Nu, Nu));
      }
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
    const auto x0 = xl(0.);
    const auto xf = xl(tf);

    Eigen::SparseMatrix<double> Jblocks(Nvars, Nvars);
    block_add_identity(Jblocks, 0, 0, 1);
    block_add(Jblocks, 1, 1, dr_exp<X>(e0));
    block_add(Jblocks, 1 + Nx, 1 + Nx, dr_exp<X>(ef));
    block_add_identity(Jblocks, 1 + 2 * Nx, 1 + 2 * Nx, Nq);
    // const auto dxf_tf = Ad(smooth::exp<X>(-ef)) * dxl_tf;  // skipping this for now

    const auto & dtheta = f.jacobian(tf, rplus(x0, e0), rplus(xf, ef), q);
    return dtheta * Jblocks;
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
    const auto Hp0 = d2r_exp<X>(e0);  // Nx x (Nx * Nx)
    const auto Hpf = d2r_exp<X>(ef);  // Nx x (Nx * Nx)

    Eigen::SparseMatrix<double> Jblocks(Nvars, Nvars);
    block_add_identity(Jblocks, 0, 0, 1);
    block_add(Jblocks, 1, 1, dr_exp<X>(e0));
    block_add(Jblocks, 1 + Nx, 1 + Nx, dr_exp<X>(ef));
    block_add_identity(Jblocks, 1 + 2 * Nx, 1 + 2 * Nx, Nq);

    Eigen::SparseMatrix<double> ret(Nvars, Nouts * Nvars);
    for (auto no = 0u; no < Nouts; ++no) {  // for each output
      const auto b0 = Nvars * no;           // block index

      // terms involving second derivatives w.r.t. f
      block_add(ret, 0, b0, Jblocks.transpose() * Hf.block(0, b0, Nvars, Nvars) * Jblocks);

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
