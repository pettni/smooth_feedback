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

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "smooth/feedback/collocation/meshfunction.hpp"

struct Functor
{
  template<typename T>
  Eigen::Vector3<T> operator()(T t, const Eigen::Vector3<T> & x, const Eigen::Vector3<T> & u)
  {
    return t * (x + 2 * u);
  }

  Eigen::SparseMatrix<double>
  jacobian(double t, const Eigen::Vector3d & x, const Eigen::Vector3d & u) const
  {
    Eigen::SparseMatrix<double> ret(3, 7);
    ret.coeffRef(0, 0) = (x(0) + 2 * u(0));
    ret.coeffRef(1, 0) = (x(1) + 2 * u(1));
    ret.coeffRef(2, 0) = (x(2) + 2 * u(2));

    ret.coeffRef(0, 1) = t;
    ret.coeffRef(1, 2) = t;
    ret.coeffRef(2, 3) = t;

    ret.coeffRef(0, 4) = 2 * t;
    ret.coeffRef(1, 5) = 2 * t;
    ret.coeffRef(2, 6) = 2 * t;

    ret.makeCompressed();

    return ret;
  }

  Eigen::SparseMatrix<double>
  hessian(double, const Eigen::Vector3d &, const Eigen::Vector3d &) const
  {
    Eigen::SparseMatrix<double> ret(7, 3 * 7);
    ret.coeffRef(0, 1) = 1;
    ret.coeffRef(0, 4) = 2;
    ret.coeffRef(1, 0) = 1;
    ret.coeffRef(4, 0) = 2;

    ret.coeffRef(0, 7 + 2) = 1;
    ret.coeffRef(0, 7 + 5) = 2;
    ret.coeffRef(2, 7 + 0) = 1;
    ret.coeffRef(5, 7 + 0) = 2;

    ret.coeffRef(0, 14 + 3) = 1;
    ret.coeffRef(0, 14 + 6) = 2;
    ret.coeffRef(3, 14 + 0) = 1;
    ret.coeffRef(6, 14 + 0) = 2;

    ret.makeCompressed();

    return ret;
  }
};

TEST(MeshEval, EvalDerivative)
{
  // Mesh function
  Functor f;

  // Mesh
  smooth::feedback::Mesh<5, 5> mesh;
  mesh.refine_ph(0, 10);
  const auto N = mesh.N_colloc();

  // Mesh variables
  const double t0                = 1;
  const double tf                = 6;
  Eigen::Matrix<double, 3, -1> X = Eigen::Matrix<double, 3, -1>::Random(3, N + 1);
  Eigen::Matrix<double, 3, -1> U = Eigen::Matrix<double, 3, -1>::Random(3, N);

  // Compute numerical derivative
  const auto f_deriv =
    [&](double t0, double tf, Eigen::VectorXd xvar, Eigen::VectorXd uvar) -> Eigen::VectorXd {
    const Eigen::Matrix<double, 3, -1> Xvar = xvar.reshaped(3, N + 1);
    const Eigen::Matrix<double, 3, -1> Uvar = uvar.reshaped(3, N);

    smooth::feedback::MeshValue<0> o;
    smooth::feedback::mesh_eval(o, mesh, f, t0, tf, Xvar.colwise(), Uvar.colwise());
    return o.F;
  };

  Eigen::VectorXd x_flat = X.reshaped();
  Eigen::VectorXd u_flat = U.reshaped();

  const auto [fval, df_num] = smooth::diff::dr<1, smooth::diff::Type::Numerical>(
    f_deriv, smooth::wrt(t0, tf, x_flat, u_flat));

  // Compute analytical derivative (autodiff)
  smooth::feedback::MeshValue<1> out1;
  smooth::feedback::mesh_eval<1, smooth::diff::Type::Numerical>(
    out1, mesh, f, t0, tf, X.colwise(), U.colwise());

  // Compute analytical derivative (analytic)
  smooth::feedback::MeshValue<1> out2;
  smooth::feedback::mesh_eval<1, smooth::diff::Type::Analytic>(
    out2, mesh, f, t0, tf, X.colwise(), U.colwise());

  // Compare derivatives
  ASSERT_TRUE(fval.isApprox(Eigen::MatrixXd(out1.F)));
  ASSERT_TRUE(fval.isApprox(Eigen::MatrixXd(out2.F)));
  ASSERT_TRUE(df_num.isApprox(Eigen::MatrixXd(out1.dF), 1e-3));
  ASSERT_TRUE(df_num.isApprox(Eigen::MatrixXd(out2.dF), 1e-3));

  // REDUCE
  std::vector<double> lambdas(N, 0.5);
  smooth::feedback::MeshValue<0> out3;
  smooth::feedback::mesh_eval_reduce<0>(out3, mesh, f, t0, tf, X.colwise(), U.colwise(), lambdas);

  ASSERT_TRUE(out3.F.isApprox(0.5 * out1.F.reshaped(3, N).rowwise().sum()));
}

TEST(MeshEval, EvalReduceDerivative)
{
  // Mesh function
  Functor f;

  // Mesh
  smooth::feedback::Mesh<5, 5> mesh;
  mesh.refine_ph(0, 10);
  const auto N = mesh.N_colloc();

  // Mesh variables
  const double t0                      = 1;
  const double tf                      = 6;
  const Eigen::Matrix<double, 3, -1> X = Eigen::Matrix<double, 3, -1>::Random(3, N + 1);
  const Eigen::Matrix<double, 3, -1> U = Eigen::Matrix<double, 3, -1>::Random(3, N);
  const Eigen::VectorXd lambdas        = Eigen::VectorXd::Random(N);

  // Compute numerical outer derivative
  const auto f_deriv =
    [&](double t0, double tf, Eigen::VectorXd xvar, Eigen::VectorXd uvar) -> Eigen::VectorXd {
    const Eigen::Matrix<double, 3, -1> Xvar = xvar.reshaped(3, N + 1);
    const Eigen::Matrix<double, 3, -1> Uvar = uvar.reshaped(3, N);

    smooth::feedback::MeshValue<0> o;
    smooth::feedback::mesh_eval_reduce(o, mesh, f, t0, tf, Xvar.colwise(), Uvar.colwise(), lambdas);
    return o.F;
  };

  Eigen::VectorXd x_flat = X.reshaped();
  Eigen::VectorXd u_flat = U.reshaped();

  const auto [f_num, df_num, d2f_num] = smooth::diff::dr<2, smooth::diff::Type::Numerical>(
    f_deriv, smooth::wrt(t0, tf, x_flat, u_flat));

  // Compute analytical derivative (numerical inner derivatives)
  smooth::feedback::MeshValue<2> out1;
  smooth::feedback::mesh_eval_reduce<2, smooth::diff::Type::Numerical>(
    out1, mesh, f, t0, tf, X.colwise(), U.colwise(), lambdas);

  // Compute analytical derivative (analytic inner derivatives)
  smooth::feedback::MeshValue<2> out2;
  smooth::feedback::mesh_eval_reduce<2, smooth::diff::Type::Analytic>(
    out2, mesh, f, t0, tf, X.colwise(), U.colwise(), lambdas);

  // Compare values
  ASSERT_TRUE(f_num.isApprox(out1.F));
  ASSERT_TRUE(f_num.isApprox(out2.F));

  // Compare first derivatives
  ASSERT_TRUE(df_num.isApprox(Eigen::MatrixXd(out1.dF), 1e-3));
  ASSERT_TRUE(df_num.isApprox(Eigen::MatrixXd(out2.dF), 1e-3));

  // Compare second derivatives
  ASSERT_TRUE(d2f_num.isApprox(Eigen::MatrixXd(out1.d2F), 1e-3));
  ASSERT_TRUE(d2f_num.isApprox(Eigen::MatrixXd(out2.d2F), 1e-3));
}
