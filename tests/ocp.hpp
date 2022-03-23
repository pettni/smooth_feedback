// smooth: Lie Theory for Robotics
// https://github.com/pettni/smooth
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

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <smooth/derivatives.hpp>
#include <smooth/feedback/ocp.hpp>
#include <smooth/feedback/utils/sparse.hpp>
#include <smooth/se2.hpp>

template<typename T>
using X = smooth::SE2<T>;

template<typename T>
using U = Eigen::Vector<T, 2>;

template<typename T>
using Q = Eigen::Vector<T, 1>;

template<typename T, std::size_t N>
using Vec = Eigen::Vector<T, N>;

static constexpr auto Nx     = smooth::Dof<X<double>>;
static constexpr auto Nq     = smooth::Dof<Q<double>>;
static constexpr auto Nu     = smooth::Dof<U<double>>;
static constexpr auto Ninner = 1 + Nx + smooth::Dof<U<double>>;
static constexpr auto Nouter = 1 + 2 * Nx + smooth::Dof<Q<double>>;

static constexpr auto t_B_inner = 0;
static constexpr auto x_B_inner = t_B_inner + 1;
static constexpr auto u_B_inner = x_B_inner + Nx;

static constexpr auto tf_B_outer = 0;
static constexpr auto x0_B_outer = tf_B_outer + 1;
static constexpr auto xf_B_outer = x0_B_outer + Nx;
static constexpr auto q_B_outer  = xf_B_outer + Nx;

/// @brief Objective function
struct TestOcpObjective
{
  template<typename T>
  T operator()(T, const X<T> &, const X<T> &, const Vec<T, 1> & q) const
  {
    return q.x();
  }

  Eigen::SparseMatrix<double>
  jacobian(double, const X<double> &, const X<double> &, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(1, Nouter);
    ret.coeffRef(0, q_B_outer) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double>
  hessian(double, const X<double> &, const X<double> &, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(Nouter, Nouter);
    return ret;
  }
};

struct TestOcpDyn
{
  template<typename T>
  smooth::Tangent<X<T>> operator()(T, const X<T> & x, const U<T> & u) const
  {
    return {
      u.x() - 0.1 * x.r2().x(),
      0,
      u.y(),
    };
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> & x, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(Nx, Ninner);
    ret.coeffRef(0, x_B_inner)     = -0.1 * std::cos(x.so2().angle());  // df1 / dx
    ret.coeffRef(0, x_B_inner + 1) = 0.1 * std::sin(x.so2().angle());   // df1 / dy
    ret.coeffRef(0, u_B_inner)     = 1;
    ret.coeffRef(2, u_B_inner + 1) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> & x, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(Ninner, Nx * Ninner);
    ret.coeffRef(x_B_inner + 0, 0 * Nx + x_B_inner + 2) =
      0.1 * std::sin(x.so2().angle());  // d2f1 / dx dth
    ret.coeffRef(x_B_inner + 1, 0 * Nx + x_B_inner + 2) =
      0.1 * std::cos(x.so2().angle());  // d2f1 / dy dth
    return ret;
  }
};

struct TestOcpIntegrand
{
  template<typename T>
  Vec<T, 1> operator()(T, const X<T> & x, const U<T> & u) const
  {
    return 0.5 * Vec<T, 1>{(x - X<T>::Identity()).squaredNorm() + u.squaredNorm()};
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> & x, const U<double> & u) const
  {
    const auto a = x - X<double>::Identity();
    Eigen::SparseMatrix<double> ret(1, Ninner);
    smooth::feedback::block_add(ret, 0, x_B_inner, smooth::dr_rminus_squarednorm<X<double>>(a));
    ret.coeffRef(0, u_B_inner)     = u.x();
    ret.coeffRef(0, u_B_inner + 1) = u.y();
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> & x, const U<double> &) const
  {
    const auto H = smooth::d2r_rminus_squarednorm<X<double>>(x.log());

    Eigen::SparseMatrix<double> ret(Ninner, 1 * Ninner);
    smooth::feedback::block_add(ret, x_B_inner, x_B_inner, H);
    ret.coeffRef(u_B_inner, u_B_inner)         = 1;
    ret.coeffRef(u_B_inner + 1, u_B_inner + 1) = 1;
    return ret;
  }
};

struct TestOcpCr
{
  template<typename T>
  Vec<T, 1> operator()(T, const X<T> &, const U<T> & u) const
  {
    return Vec<T, 1>{u.x()};
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(1, Ninner);
    ret.coeffRef(0, u_B_inner) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(Ninner, 1 * Ninner);
    return ret;
  }
};

struct TestOcpCe
{
  static constexpr auto Nce = 1 + 2 * Nx;

  template<typename T>
  Vec<T, Nce> operator()(T tf, const X<T> & x0, const X<T> & xf, const Vec<T, 1> &) const
  {
    Vec<T, Nce> ret;
    ret << tf, x0.log(), xf.log();
    return ret;
  }

  Eigen::SparseMatrix<double>
  jacobian(double, const X<double> & x0, const X<double> & xf, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(Nce, Nouter);
    ret.coeffRef(tf_B_outer, tf_B_outer) = 1;
    smooth::feedback::block_add(
      ret, x0_B_outer, x0_B_outer, smooth::dr_expinv<X<double>>(x0.log()));
    smooth::feedback::block_add(
      ret, xf_B_outer, xf_B_outer, smooth::dr_expinv<X<double>>(xf.log()));
    return ret;
  }

  Eigen::SparseMatrix<double>
  hessian(double, const X<double> & x0, const X<double> & xf, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(Nouter, Nce * Nouter);

    const auto d2_logx0 = smooth::d2r_rminus<X<double>>(x0.log());
    const auto d2_logxf = smooth::d2r_rminus<X<double>>(xf.log());

    for (auto i = 0u; i < Nx; ++i) {
      smooth::feedback::block_add(
        ret, x0_B_outer, Nouter * (1 + i) + x0_B_outer, d2_logx0.block(0, i * Nx, Nx, Nx));

      smooth::feedback::block_add(
        ret, xf_B_outer, Nouter * (1 + Nx + i) + xf_B_outer, d2_logxf.block(0, i * Nx, Nx, Nx));
    }

    return ret;
  }
};

using OcpTest = smooth::feedback::
  OCP<X<double>, U<double>, TestOcpObjective, TestOcpDyn, TestOcpIntegrand, TestOcpCr, TestOcpCe>;

inline const OcpTest ocp_test{
  .theta = TestOcpObjective{},
  .f     = TestOcpDyn{},
  .g     = TestOcpIntegrand{},
  .cr    = TestOcpCr{},
  .crl   = Vec<double, 1>{{-1}},
  .cru   = Vec<double, 1>{{1}},
  .ce    = TestOcpCe{},
  .cel   = Vec<double, 1 + 2 * Nx>::Random(),
  .ceu   = Vec<double, 1 + 2 * Nx>::Random(),
};
