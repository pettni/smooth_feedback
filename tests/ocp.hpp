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
#include <smooth/se2.hpp>

template<typename T>
using X = smooth::SE2<T>;

template<typename T>
using U = Eigen::Vector<T, 1>;

template<typename T, std::size_t N>
using Vec = Eigen::Vector<T, N>;

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
    Eigen::SparseMatrix<double> ret(1, 8);
    ret.coeffRef(0, 7) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double>
  hessian(double, const X<double> &, const X<double> &, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(8, 8);
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
    Eigen::SparseMatrix<double> ret(3, 5);
    ret.coeffRef(1, 4) = 1;
    ret.coeffRef(2, 4) = -1;
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(5, 3 * 5);
    return ret;
  }
};

struct TestOcpIntegrand
{
  template<typename T>
  Vec<T, 1> operator()(T, const X<T> & x, const U<T> & u) const
  {
    return Vec<T, 1>{x.r2().squaredNorm() + u.squaredNorm()};
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> & x, const U<double> & u) const
  {
    Eigen::SparseMatrix<double> ret(1, 5);
    ret.coeffRef(0, 1) = 2 * x.r2().x();
    ret.coeffRef(0, 2) = 2 * x.r2().y();
    ret.coeffRef(0, 4) = 2 * u.x();
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(5, 5);
    ret.coeffRef(1, 1) = 2;
    ret.coeffRef(2, 2) = 2;
    ret.coeffRef(4, 4) = 2;
    return ret;
  }
};

struct TestOcpCr
{
  template<typename T>
  Vec<T, 2> operator()(T, const X<T> & x, const U<T> & u) const
  {
    return Vec<T, 2>{x.r2().y(), u.x()};
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(2, 5);
    ret.coeffRef(0, 2) = 1;
    ret.coeffRef(1, 4) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(5, 2 * 5);
    return ret;
  }
};

struct TestOcpCe
{
  template<typename T>
  Vec<T, 7> operator()(T tf, const X<T> & x0, const X<T> & xf, const Vec<T, 1> &) const
  {
    Vec<T, 7> ret;
    ret << tf, x0.log(), xf.log();
    return ret;
  }

  Eigen::SparseMatrix<double>
  jacobian(double, const X<double> & x0, const X<double> & xf, const Vec<double, 1> &) const
  {
    const auto dlog_x0 = smooth::dr_expinv<X<double>>(x0.log());
    const auto dlog_xf = smooth::dr_expinv<X<double>>(xf.log());

    Eigen::SparseMatrix<double> ret(7, 8);
    ret.coeffRef(0, 0) = 1;
    for (auto i = 0u; i < smooth::Dof<X<double>>; ++i) {
      for (auto j = 0u; j < smooth::Dof<X<double>>; ++j) {
        ret.coeffRef(1 + i, 1 + j) = dlog_x0(i, j);
      }
    }
    for (auto i = 0u; i < smooth::Dof<X<double>>; ++i) {
      for (auto j = 0u; j < smooth::Dof<X<double>>; ++j) {
        ret.coeffRef(4 + i, 4 + j) = dlog_xf(i, j);
      }
    }
    return ret;
  }

  Eigen::SparseMatrix<double>
  hessian(double, const X<double> & x0, const X<double> & xf, const Vec<double, 1> &) const
  {
    Eigen::Matrix<double, -1, -1> ret(8, 7 * 8);
    ret.setZero();

    const auto d2_logx0 = smooth::hessian_rminus<X<double>>(x0, X<double>::Identity());  // 3 x 9
    const auto d2_logxf = smooth::hessian_rminus<X<double>>(xf, X<double>::Identity());  // 3 x 9

    ret.block(1, 8 * 1 + 1, 3, 3) = d2_logx0.block(0, 0, 3, 3);
    ret.block(1, 8 * 2 + 1, 3, 3) = d2_logx0.block(0, 3, 3, 3);
    ret.block(1, 8 * 3 + 1, 3, 3) = d2_logx0.block(0, 6, 3, 3);

    ret.block(4, 8 * 4 + 4, 3, 3) = d2_logxf.block(0, 0, 3, 3);
    ret.block(4, 8 * 5 + 4, 3, 3) = d2_logxf.block(0, 3, 3, 3);
    ret.block(4, 8 * 6 + 4, 3, 3) = d2_logxf.block(0, 6, 3, 3);

    return ret.sparseView();
  }
};

using OcpTest = smooth::feedback::
  OCP<X<double>, U<double>, TestOcpObjective, TestOcpDyn, TestOcpIntegrand, TestOcpCr, TestOcpCe>;

inline const OcpTest ocp_test{
  .theta = TestOcpObjective{},
  .f     = TestOcpDyn{},
  .g     = TestOcpIntegrand{},
  .cr    = TestOcpCr{},
  .crl   = Vec<double, 2>{{-0.5, -1}},
  .cru   = Vec<double, 2>{{1.5, 1}},
  .ce    = TestOcpCe{},
  .cel   = Vec<double, 7>{{5, 1, 1, 0, 0.1, 0, 0}},
  .ceu   = Vec<double, 7>{{5, 1, 1, 0, 0.1, 0, 0}},
};
