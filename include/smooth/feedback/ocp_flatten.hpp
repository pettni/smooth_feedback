// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

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
#include <smooth/bundle.hpp>
#include <smooth/diff.hpp>

#include "ocp.hpp"
#include "smooth/lie_sparse.hpp"
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
 * @brief Sparse matrices containing reordered rows of algebra generators.
 */
template<LieGroup G>
inline auto generators_sparse_reordered = []() -> std::array<Eigen::SparseMatrix<double, Eigen::RowMajor>, Dof<G>> {
  std::array<Eigen::SparseMatrix<double, Eigen::RowMajor>, Dof<G>> ret;
  for (auto k = 0u; k < Dof<G>; ++k) {
    ret[k].resize(Dof<G>, Dof<G>);
    for (auto i = 0u; i < Dof<G>; ++i) { ret[k].row(i) = ::smooth::generators_sparse<G>[i].row(k); }
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
    for (auto j = 0u; j < Dof<G>; ++j) { ret.col(j * Dof<G> + i) = smooth::generators_sparse<G>[i].row(j).transpose(); }
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
  using BundleT = smooth::Bundle<Eigen::Vector<double, 1>, X, U>;

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

  Eigen::SparseMatrix<double> ad_e_ = smooth::ad_sparse_pattern<X>;
  Eigen::SparseMatrix<double> ad_vi = smooth::ad_sparse_pattern<X>;

  Eigen::SparseMatrix<double> Joplus_ = smooth::d_exp_sparse_pattern<BundleT>;
  Eigen::SparseMatrix<double> Hoplus_ = smooth::d2_exp_sparse_pattern<BundleT>;

  Eigen::SparseMatrix<double> dexpinv_e_  = smooth::d_exp_sparse_pattern<X>;
  Eigen::SparseMatrix<double> d2expinv_e_ = smooth::d2_exp_sparse_pattern<X>;

  Eigen::SparseMatrix<double> J_{Nouts, Nvars};
  Eigen::SparseMatrix<double> H_{Nvars, Nouts * Nvars};

  // Would ideally like to remove these temporaries...
  Eigen::SparseMatrix<double> ji_{Nouts, Nvars}, ji_tmp_{Nouts, Nvars}, hi_{Nvars, Nouts *Nvars},
    hi_tmp_{Nvars, Nouts *Nvars};

  /// @brief Calculate jacobian of (t, x(t)+e, u(t)+v) w.r.t. (t, e, v)
  void update_joplus(const E & e, const V & v, const E & dxl, const V & dul)
  {
    dr_exp_sparse<BundleT>(Joplus_, (Tangent<BundleT>() << 1, e, v).finished());

    block_write(Joplus_, x_B, t_B, Ad<X>(smooth::exp<X>(-e)) * dxl);
    block_write(Joplus_, u_B, t_B, Ad<U>(smooth::exp<U>(-v)) * dul);

    Joplus_.makeCompressed();
  }

  /// @brief Calculate hessian of (t, x(t)+e, u(t)+v) w.r.t. (t, e, v)
  void update_hoplus(const E & e, const V & v, const E & dxl, const V & dul)
  {
    d2r_exp_sparse<BundleT>(Hoplus_, (Tangent<BundleT>() << 1, e, v).finished());

    // d (Ad_X b) = -ad_(Ad_X b) * Ad_X
    const TangentMap<X> Adexp_X  = Ad<X>(smooth::exp<X>(-e));
    const TangentMap<X> dAdexp_X = ad<X>(Adexp_X * dxl) * Adexp_X * dr_exp<X>(-e);
    const TangentMap<U> Adexp_U  = Ad<U>(smooth::exp<U>(-v));
    const TangentMap<U> dAdexp_U = ad<U>(Adexp_U * dul) * Adexp_U * dr_exp<U>(-v);

    for (auto nx = 0u; nx < Nx; ++nx) {
      const auto b0 = Nvars * (x_B + nx);
      block_write(Hoplus_, t_B, b0 + x_B, dAdexp_X.middleRows(nx, 1));
    }
    for (auto nu = 0u; nu < Nu; ++nu) {
      const auto b0 = Nvars * (u_B + nu);
      block_write(Hoplus_, t_B, b0 + u_B, dAdexp_U.middleRows(nu, 1));
    }

    Hoplus_.makeCompressed();
  }

public:
  template<typename A1, typename A2, typename A3>
  FlatDyn(A1 && a1, A2 && a2, A3 && a3) : f(std::forward<A1>(a1)), xl(std::forward<A2>(a2)), ul(std::forward<A3>(a3))
  {}

  template<typename T>
  CastT<T, E> operator()(const T & t, const CastT<T, E> & e, const CastT<T, V> & v) const
  {
    using XT = CastT<T, X>;

    // can not double-differentiate, so we hide derivative of xl w.r.t. t
    const double tdbl           = static_cast<double>(t);
    const auto [unused, dxlval] = diff::dr<1>(xl, wrt(tdbl));

    return dr_expinv<XT>(e) * (f(t, rplus(xl(t), e), rplus(ul(t), v)) - dxlval.template cast<T>())
         + ad<XT>(e) * dxlval.template cast<T>();
  }

  // First derivative
  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  jacobian(double t, const E & e, const V & v) requires(diff::detail::diffable_order1<F, std::tuple<double, X, U>>)
  {
    const double tdbl          = static_cast<double>(t);
    const auto [xlval, dxlval] = diff::dr<1>(xl, wrt(tdbl));
    const auto [ulval, dulval] = diff::dr<1>(ul, wrt(tdbl));
    const auto x               = rplus(xlval, e);
    const auto u               = rplus(ulval, v);

    dr_expinv_sparse<X>(dexpinv_e_, e);
    d2r_expinv_sparse<X>(d2expinv_e_, e);
    update_joplus(e, v, dxlval, dulval);

    // value and derivative of f
    const auto fval = f(t, x, u);
    const auto & Jf = f.jacobian(t, x, u);

    // Want to differentiate  drexpinv * (f o plus - dxl) + ad dxl

    // Start with drexpinv * d (f \circ (+))
    J_ = dexpinv_e_ * Jf * Joplus_;
    // Add d ( drexpinv ) * (f \circ (+) - dxl)
    for (auto i = 0u; i < d2expinv_e_.outerSize(); ++i) {
      for (Eigen::InnerIterator it(d2expinv_e_, i); it; ++it) {
        J_.coeffRef(it.col() / Nx, 1 + (it.col() % Nx)) += (fval(it.row()) - dxlval(it.row())) * it.value();
      }
    }
    // Add d ( ad ) * dxl
    for (auto i = 0u; i < d_ad<X>.outerSize(); ++i) {
      for (Eigen::InnerIterator it(d_ad<X>, i); it; ++it) {
        J_.coeffRef(it.col() / Nx, 1 + (it.col() % Nx)) += dxlval(it.row()) * it.value();
      }
    }

    J_.makeCompressed();
    return J_;
  }

  // Second derivative
  //    \sum Bn (-1)^n / n! d2r (ad_a^n f)_aa - \sum Bn / n! d2r(ad_a^n dxl)_aa
  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  hessian(double t, const E & e, const V & v) requires(diff::detail::diffable_order1<F, std::tuple<double, X, U>> &&
                                                         diff::detail::diffable_order2<F, std::tuple<double, X, U>>)
  {
    const double tdbl          = static_cast<double>(t);
    const auto [xlval, dxlval] = diff::dr<1>(xl, wrt(tdbl));
    const auto [ulval, dulval] = diff::dr<1>(ul, wrt(tdbl));

    const auto x    = rplus(xlval, e);
    const auto u    = rplus(ul(t), v);
    const auto & Jf = f.jacobian(t, x, u);  // nx x (1 + nx + nu)
    const auto & Hf = f.hessian(t, x, u);   // (1 + nx + nu) x (nx * (1 + nx + nu))

    ad_sparse<X>(ad_e_, e);
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
      for (auto i = 0u; i < ad_e_.outerSize(); ++i) {
        for (Eigen::InnerIterator it(ad_e_, i); it; ++it) {
          const auto b0 = it.row() * Nvars;
          block_add(hi_tmp_, 0, b0, hi_.middleCols(it.col() * Nvars, Nvars), it.value());
        }
      }
      for (auto k = 0u; k < Nx; ++k) {
        const auto b0 = k * Nvars;
        block_add(hi_tmp_, 1, b0, generators_sparse_reordered<X>[k] * ji_);
        block_add(hi_tmp_, 0, b0 + 1, ji_.transpose() * generators_sparse_reordered<X>[k], -1);
      }
      std::swap(hi_, hi_tmp_);

      // update ji
      ji_tmp_.setZero();
      ji_tmp_ = ad_e_ * ji_;
      ad_sparse<X>(ad_vi, vi);
      block_add(ji_tmp_, 0, 1, ad_vi, -1);
      std::swap(ji_, ji_tmp_);

      // update vi
      vi.applyOnTheLeft(ad_e_);

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
  using BundleT = smooth::Bundle<Eigen::Vector<double, 1>, X, U>;

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

  Eigen::SparseMatrix<double> Joplus_ = smooth::d_exp_sparse_pattern<BundleT>;
  Eigen::SparseMatrix<double> Hoplus_ = smooth::d2_exp_sparse_pattern<BundleT>;

  Eigen::SparseMatrix<double> J_{Nouts, Nvars};
  Eigen::SparseMatrix<double> H_{Nvars, Nouts * Nvars};

  /// @brief Calculate jacobian of (t, x(t)+e, u(t)+v) w.r.t. (t, e, v)
  void update_joplus(const E & e, const V & v, const E & dxl, const V & dul)
  {
    dr_exp_sparse<BundleT>(Joplus_, (Tangent<BundleT>() << 1, e, v).finished());

    block_write(Joplus_, x_B, t_B, Ad<X>(smooth::exp<X>(-e)) * dxl);
    block_write(Joplus_, u_B, t_B, Ad<U>(smooth::exp<U>(-v)) * dul);

    Joplus_.makeCompressed();
  }

  /// @brief Calculate hessian of (t, x(t)+e, u(t)+v) w.r.t. (t, e, v)
  void update_hoplus(const E & e, const V & v, const E & dxl, const V & dul)
  {
    d2r_exp_sparse<BundleT>(Hoplus_, (Tangent<BundleT>() << 1, e, v).finished());

    // d (Ad_X b) = -ad_(Ad_X b) * Ad_X
    const TangentMap<X> Adexp_X  = Ad<X>(smooth::exp<X>(-e));
    const TangentMap<X> dAdexp_X = ad<X>(Adexp_X * dxl) * Adexp_X * dr_exp<X>(-e);
    const TangentMap<U> Adexp_U  = Ad<U>(smooth::exp<U>(-v));
    const TangentMap<U> dAdexp_U = ad<U>(Adexp_U * dul) * Adexp_U * dr_exp<U>(-v);

    for (auto nx = 0u; nx < Nx; ++nx) {
      const auto b0 = Nvars * (x_B + nx);
      block_write(Hoplus_, t_B, b0 + x_B, dAdexp_X.middleRows(nx, 1));
    }
    for (auto nu = 0u; nu < Nu; ++nu) {
      const auto b0 = Nvars * (u_B + nu);
      block_write(Hoplus_, t_B, b0 + u_B, dAdexp_U.middleRows(nu, 1));
    }

    Hoplus_.makeCompressed();
  }

public:
  template<typename A1, typename A2, typename A3>
  FlatInnerFun(A1 && a1, A2 && a2, A3 && a3)
      : f(std::forward<A1>(a1)), xl(std::forward<A2>(a2)), ul(std::forward<A3>(a3))
  {}

  template<typename T>
  Eigen::Vector<T, Nouts> operator()(const T & t, const CastT<T, E> & e, const CastT<T, V> & v) const
  {
    return f.template operator()<T>(t, rplus(xl(t), e), rplus(ul(t), v));
  }

  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  jacobian(double t, const E & e, const V & v) requires(diff::detail::diffable_order1<F, std::tuple<double, X, U>>)
  {
    const auto & [xlval, dxlval] = diff::dr<1>(xl, wrt(t));
    const auto & [ulval, dulval] = diff::dr<1>(ul, wrt(t));
    const auto & Jf              = f.jacobian(t, rplus(xlval, e), rplus(ulval, v));

    update_joplus(e, v, dxlval, dulval);

    J_ = Jf * Joplus_;
    J_.makeCompressed();
    return J_;
  }

  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  hessian(double t, const E & e, const V & v) requires(diff::detail::diffable_order1<F, std::tuple<double, X, U>> &&
                                                         diff::detail::diffable_order2<F, std::tuple<double, X, U>>)
  {
    const auto & [xlval, dxlval] = diff::dr<1>(xl, wrt(t));
    const auto & [ulval, dulval] = diff::dr<1>(ul, wrt(t));
    const auto x                 = rplus(xlval, e);
    const auto u                 = rplus(ulval, v);
    const auto & Jf              = f.jacobian(t, x, u);
    const auto & Hf              = f.hessian(t, x, u);

    update_joplus(e, v, dxlval, dulval);
    update_hoplus(e, v, dxlval, dulval);

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
  using E       = Tangent<X>;
  using Q       = Eigen::Vector<Scalar<X>, Nq>;
  using BundleT = smooth::Bundle<Eigen::Vector<double, 1>, X, X, Q>;

  F f;
  Xl xl;

  static constexpr auto Nx    = Dof<X>;
  static constexpr auto Nvars = 1 + 2 * Nx + Nq;

  static constexpr auto tf_B = 0;
  static constexpr auto x0_B = tf_B + 1;
  static constexpr auto xf_B = x0_B + Nx;
  static constexpr auto q_B  = xf_B + Nx;

  Eigen::SparseMatrix<double> Joplus_ = d_exp_sparse_pattern<BundleT>;
  Eigen::SparseMatrix<double> Hoplus_ = d2_exp_sparse_pattern<BundleT>;

  Eigen::SparseMatrix<double> J_{Nouts, Nvars};
  Eigen::SparseMatrix<double> H_{Nvars, Nouts * Nvars};

  /// @brief Calculate jacobian of (tf, xl(0.)+e0, xl(tf)+ef, q) w.r.t. (tf, e0, ef, q)
  void update_joplus(const E & e0, const E & ef, const E & dxlf)
  {
    dr_exp_sparse<BundleT>(Joplus_, (Tangent<BundleT>() << 1, e0, ef, Q::Ones()).finished());

    block_write(Joplus_, xf_B, tf_B, Ad<X>(smooth::exp<X>(-ef)) * dxlf);

    Joplus_.makeCompressed();
  }

  /// @brief Calculate hessian of (tf, xl(0.)+e0, xl(tf)+ef, q) w.r.t. (tf, e0, ef, q)
  void update_hoplus(const E & e0, const E & ef, [[maybe_unused]] const E & dxlf)
  {
    d2r_exp_sparse<BundleT>(Hoplus_, (Tangent<BundleT>() << 1, e0, ef, Q::Ones()).finished());

    // dr (Ad_X b)_X = -ad_{Ad_X b} Ad_X
    const TangentMap<X> Adexp_f  = Ad<X>(smooth::exp<X>(-ef));
    const TangentMap<X> dAdexp_f = ad<X>(Adexp_f * dxlf) * Adexp_f * dr_exp<X>(-ef);

    for (auto nx = 0u; nx < Nx; ++nx) {
      const auto b0 = Nvars * (xf_B + nx);
      block_write(Hoplus_, tf_B, b0 + xf_B, dAdexp_f.middleRows(nx, 1));
    }

    Hoplus_.makeCompressed();
  }

public:
  template<typename A1, typename A2>
  FlatEndptFun(A1 && a1, A2 && a2) : f(std::forward<A1>(a1)), xl(std::forward<A2>(a2))
  {}

  template<typename T>
  auto operator()(const T & tf, const CastT<T, E> & e0, const CastT<T, E> & ef, const CastT<T, Q> & q) const
  {
    return f.template operator()<T>(tf, rplus(xl(T(0.)), e0), rplus(xl(tf), ef), q);
  }

  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  jacobian(double tf, const E & e0, const E & ef, const Q & q) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, X, Q>>)
  {
    const auto & [xlfval, dxlfval] = diff::dr<1>(xl, wrt(tf));
    const auto & Jf                = f.jacobian(tf, rplus(xl(0.), e0), rplus(xlfval, ef), q);

    update_joplus(e0, ef, dxlfval);

    J_ = Jf * Joplus_;
    J_.makeCompressed();
    return J_;
  }

  std::reference_wrapper<const Eigen::SparseMatrix<double>>
  hessian(double tf, const E & e0, const E & ef, const Q & q) requires(
    diff::detail::diffable_order1<F, std::tuple<double, X, X, Q>> &&
      diff::detail::diffable_order2<F, std::tuple<double, X, X, Q>>)
  {
    const auto & [xlfval, dxlfval] = diff::dr<1>(xl, wrt(tf));
    const auto x0                  = rplus(xl(0.), e0);
    const auto xf                  = rplus(xlfval, ef);
    const auto & Jf                = f.jacobian(tf, x0, xf, q);  // Nouts x Nx
    const auto & Hf                = f.hessian(tf, x0, xf, q);   // Nx x (Nouts * Nx)

    update_joplus(e0, ef, dxlfval);
    update_hoplus(e0, ef, dxlfval);

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

  auto u_unflat = [ul_fun = std::forward<decltype(ul_fun)>(ul_fun), usol = flatsol.u](double t) -> U {
    return rplus(ul_fun(t), usol(t));
  };

  auto x_unflat = [xl_fun = std::forward<decltype(xl_fun)>(xl_fun), xsol = flatsol.x](double t) -> X {
    return rplus(xl_fun(t), xsol(t));
  };

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
