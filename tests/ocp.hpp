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
#include <smooth/algo/hessian.hpp>
#include <smooth/feedback/ocp.hpp>
#include <smooth/feedback/utils/sparse.hpp>
#include <smooth/se2.hpp>

template<typename T>
using X = smooth::SE2<T>;

template<typename T>
using U = Eigen::Vector<T, 1>;

template<typename T>
using Q = Eigen::Vector<T, 1>;

template<typename T, std::size_t N>
using Vec = Eigen::Vector<T, N>;

static constexpr auto Nx     = smooth::Dof<X<double>>;
static constexpr auto Ninner = 1 + Nx + smooth::Dof<U<double>>;
static constexpr auto Nouter = 1 + 2 * Nx + smooth::Dof<Q<double>>;

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
    ret.coeffRef(0, 7) = 1;
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
  smooth::Tangent<X<T>> operator()(T, const X<T> &, const U<T> & u) const
  {
    return {T(0), u.x(), -u.x()};
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(Nx, Ninner);
    ret.coeffRef(1, 4) = 1;
    ret.coeffRef(2, 4) = -1;
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(Ninner, Nx * Ninner);
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
    ret.middleCols(1, 3) = (a.transpose() * smooth::dr_expinv<X<double>>(a)).sparseView();
    ret.coeffRef(0, 4)   = u.x();
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> & x, const U<double> &) const
  {
    const auto H = smooth::hessian_rminus_norm(x, X<double>::Identity());

    Eigen::SparseMatrix<double> ret(Ninner, 1 * Ninner);
    smooth::feedback::block_add(ret, 1, 1, H);
    ret.coeffRef(4, 4) = 1;
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
    ret.coeffRef(0, 4) = 1;
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
  static constexpr auto Nce = 7;

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
    ret.coeffRef(0, 0) = 1;
    smooth::feedback::block_add(ret, 1, 1, smooth::dr_expinv<X<double>>(x0.log()));
    smooth::feedback::block_add(ret, 4, 4, smooth::dr_expinv<X<double>>(xf.log()));
    return ret;
  }

  Eigen::SparseMatrix<double>
  hessian(double, const X<double> & x0, const X<double> & xf, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(Nouter, Nce * Nouter);

    const auto d2_logx0 = smooth::hessian_rminus<X<double>>(x0, X<double>::Identity());
    const auto d2_logxf = smooth::hessian_rminus<X<double>>(xf, X<double>::Identity());

    smooth::feedback::block_add(ret, 1, Nouter * 1 + 1, d2_logx0.block(0, 0, 3, 3));
    smooth::feedback::block_add(ret, 1, Nouter * 2 + 1, d2_logx0.block(0, 3, 3, 3));
    smooth::feedback::block_add(ret, 1, Nouter * 3 + 1, d2_logx0.block(0, 6, 3, 3));

    smooth::feedback::block_add(ret, 4, Nouter * 4 + 4, d2_logxf.block(0, 0, 3, 3));
    smooth::feedback::block_add(ret, 4, Nouter * 5 + 4, d2_logxf.block(0, 3, 3, 3));
    smooth::feedback::block_add(ret, 4, Nouter * 6 + 4, d2_logxf.block(0, 6, 3, 3));

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
  .cel   = Vec<double, 7>{{5, 1, 1, 0, 0.1, 0, 0}},
  .ceu   = Vec<double, 7>{{5, 1, 1, 0, 0.1, 0, 0}},
};
