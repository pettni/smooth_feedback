// Copyright (C) 2022 Petter Nilsson. MIT License.

/**
 * @file Optimal control problem on SE2.
 */

#ifndef OCP_SE2_HPP_
#define OCP_SE2_HPP_

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <smooth/bundle.hpp>
#include <smooth/derivatives.hpp>
#include <smooth/feedback/ocp.hpp>
#include <smooth/feedback/utils/sparse.hpp>
#include <smooth/se2.hpp>

template<typename T>
using X = smooth::Bundle<smooth::SE2<T>, Eigen::Vector2<T>>;

template<typename T>
using U = Eigen::Vector2<T>;

template<typename T, std::size_t N>
using Vec = Eigen::Vector<T, N>;

/// @brief Objective function
struct SE2Theta
{
  template<typename T>
  T operator()(T tf, const X<T> &, const X<T> &, const Vec<T, 1> & q) const
  {
    return tf + q.x();
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> &, const X<double> &, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(1, 12);
    ret.coeffRef(0, 0)  = 1;
    ret.coeffRef(0, 11) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> &, const X<double> &, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(12, 12);
    return ret;
  }
};

/// @brief Dynamics
struct SE2Dyn
{
  template<typename T>
  smooth::Tangent<X<T>> operator()(T, const X<T> & x, const U<T> & u) const
  {
    smooth::Tangent<X<T>> ret;
    ret(0) = x.template part<1>().x();
    ret(1) = T(0);
    ret(2) = x.template part<1>().y();
    ret(3) = u.x();
    ret(4) = u.y();
    return ret;
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(5, 8);
    ret.coeffRef(0, 4) = 1;
    ret.coeffRef(2, 5) = 1;
    ret.coeffRef(3, 6) = 1;
    ret.coeffRef(4, 7) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(8, 5 * 8);
    return ret;
  }
};

/// @brief Target trajectory
const auto xdes = []<typename T>(T t) -> X<T> {
  const Eigen::Vector3<T> vel{1., 0., 0.5};

  X<T> ret;
  ret.template part<0>()     = smooth::SE2<T>::exp(t * vel);
  ret.template part<1>().x() = vel.x();
  ret.template part<1>().y() = vel.z();
  return ret;
};

/// @brief Integrals
struct SE2Integral
{
  template<typename T>
  Vec<T, 1> operator()(T t, const X<T> & x, const U<T> & u) const
  {
    return 0.5 * Vec<T, 1>{(x - xdes(t)).squaredNorm() + u.squaredNorm()};
  }

  Eigen::SparseMatrix<double> jacobian(double t, const X<double> & x, const U<double> & u) const
  {
    const auto a = x - xdes(t);

    Eigen::SparseMatrix<double> ret(1, 8);
    ret.coeffRef(0, 0) =
      -(a.transpose() * smooth::dl_expinv<X<double>>(a)).dot(Eigen::Vector<double, 5>{1., 0., 0.5, 0, 0});
    smooth::feedback::block_add(ret, 0, 1, smooth::dr_rminus_squarednorm<X<double>>(a));
    ret.coeffRef(0, 6) = u.x();
    ret.coeffRef(0, 7) = u.y();
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double t, const X<double> & x, const U<double> &) const
  {
    const auto H = smooth::d2r_rminus_squarednorm<X<double>>(x - xdes(t));

    Eigen::SparseMatrix<double> ret(8, 8);
    /// @todo don't have derivatives w.r.t. t
    smooth::feedback::block_add(ret, 1, 1, H);
    ret.coeffRef(6, 6) = 1;
    ret.coeffRef(7, 7) = 1;
    return ret;
  }
};

/// @brief Running constraints
struct SE2Cr
{
  template<typename T>
  Vec<T, 2> operator()(T, const X<T> &, const U<T> & u) const
  {
    return u;
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(2, 8);
    ret.coeffRef(0, 6) = 1;
    ret.coeffRef(1, 7) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(8, 2 * 8);
    return ret;
  }
};

/// @brief End constraints
struct SE2Ce
{
  template<typename T>
  Vec<T, 6> operator()(T tf, const X<T> & x0, const X<T> &, const Vec<T, 1> &) const
  {
    Vec<T, 6> ret;
    ret << tf, x0.log();
    return ret;
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> & x0, const X<double> &, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(6, 12);
    ret.coeffRef(0, 0) = 1;
    smooth::feedback::block_add(ret, 1, 1, smooth::dr_expinv<X<double>>(x0.log()));
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> & x0, const X<double> &, const Vec<double, 1> &) const
  {
    const auto d2_logx0 = smooth::d2r_rminus<X<double>>(x0.log());

    Eigen::SparseMatrix<double> ret(12, 6 * 12);
    for (auto i = 0u; i < 5; ++i) {
      smooth::feedback::block_add(ret, 1, 12 * (1 + i) + 1, d2_logx0.block(0, i * 5, 5, 5));
    }

    return ret;
  }
};

using OcpSE2 = smooth::feedback::OCP<X<double>, U<double>, SE2Theta, SE2Dyn, SE2Integral, SE2Cr, SE2Ce>;

inline const OcpSE2 ocp_se2{
  .theta = SE2Theta{},
  .f     = SE2Dyn{},
  .g     = SE2Integral{},
  .cr    = SE2Cr{},
  .crl   = Vec<double, 2>{{-1, -1}},
  .cru   = Vec<double, 2>{{1, 1}},
  .ce    = SE2Ce{},
  .cel   = Vec<double, 6>{{5, 0, 0, 0, 1, 0}},
  .ceu   = Vec<double, 6>{{5, 0, 0, 0, 1, 0}},
};

#endif  // OCP_SE2_HPP_
