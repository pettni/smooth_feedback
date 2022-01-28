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
#include <smooth/feedback/ocp.hpp>

template<typename T>
using X = Eigen::Vector<T, 2>;

template<typename T>
using U = Eigen::Vector<T, 1>;

template<typename T, std::size_t N>
using Vec = Eigen::Vector<T, N>;

/// @brief Objective function
struct DITheta
{
  template<typename T>
  T operator()(T, const X<T> &, const X<T> &, const Vec<T, 1> & q) const
  {
    return q.x();
  }

  Eigen::SparseMatrix<double>
  jacobian(double, const X<double> &, const X<double> &, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(1, 6);
    ret.coeffRef(0, 5) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double>
  hessian(double, const X<double> &, const X<double> &, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(6, 6);
    return ret;
  }
};

struct DIDyn
{
  template<typename T>
  smooth::Tangent<X<T>> operator()(T, const X<T> & x, const U<T> & u) const
  {
    return {x.y(), u.x()};
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(2, 4);
    ret.coeffRef(0, 2) = 1;
    ret.coeffRef(1, 3) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(4, 8);
    return ret;
  }
};

struct DIIntegral
{
  template<typename T>
  Vec<T, 1> operator()(T, const X<T> & x, const U<T> & u) const
  {
    return Vec<T, 1>{x.squaredNorm() + u.squaredNorm()};
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> & x, const U<double> & u) const
  {
    Eigen::SparseMatrix<double> ret(1, 4);
    ret.coeffRef(0, 1) = 2 * x.x();
    ret.coeffRef(0, 2) = 2 * x.y();
    ret.coeffRef(0, 3) = 2 * u.x();
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(4, 4);
    ret.coeffRef(1, 1) = 2;
    ret.coeffRef(2, 2) = 2;
    ret.coeffRef(3, 3) = 2;
    return ret;
  }
};

struct DICr
{
  template<typename T>
  Vec<T, 2> operator()(T, const X<T> & x, const U<T> & u) const
  {
    return Vec<T, 2>{x.y(), u.x()};
  }

  Eigen::SparseMatrix<double> jacobian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(2, 4);
    ret.coeffRef(0, 2) = 1;
    ret.coeffRef(1, 3) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double> hessian(double, const X<double> &, const U<double> &) const
  {
    Eigen::SparseMatrix<double> ret(4, 8);
    return ret;
  }
};

struct DICe
{
  template<typename T>
  Vec<T, 5> operator()(T tf, const X<T> & x0, const X<T> & xf, const Vec<T, 1> &) const
  {
    Vec<T, 5> ret;
    ret << tf, x0, xf;
    return ret;
  }

  Eigen::SparseMatrix<double>
  jacobian(double, const X<double> &, const X<double> &, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(5, 6);
    ret.coeffRef(0, 0) = 1;
    ret.coeffRef(1, 1) = 1;
    ret.coeffRef(2, 2) = 1;
    ret.coeffRef(3, 3) = 1;
    ret.coeffRef(4, 4) = 1;
    return ret;
  }

  Eigen::SparseMatrix<double>
  hessian(double, const X<double> &, const X<double> &, const Vec<double, 1> &) const
  {
    Eigen::SparseMatrix<double> ret(6, 30);
    return ret;
  }
};

using OcpDI = smooth::feedback::OCP<X<double>, U<double>, DITheta, DIDyn, DIIntegral, DICr, DICe>;

inline const OcpDI ocp_di{
  .theta = DITheta{},
  .f     = DIDyn{},
  .g     = DIIntegral{},
  .cr    = DICr{},
  .crl   = Vec<double, 2>{{-0.5, -1}},
  .cru   = Vec<double, 2>{{1.5, 1}},
  .ce    = DICe{},
  .cel   = Vec<double, 5>{{5, 1, 1, 0.1, 0}},
  .ceu   = Vec<double, 5>{{5, 1, 1, 0.1, 0}},
};
