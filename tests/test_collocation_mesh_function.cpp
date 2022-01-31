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
#include <smooth/so3.hpp>

#include "smooth/feedback/collocation/mesh_function.hpp"

using namespace Eigen;

/// Reduce a hstacked Hessian (as returned by smooth::diff) with multipliers
MatrixXd blocksum(const VectorXd & lambda, const MatrixXd & in)
{
  MatrixXd ret = MatrixXd::Zero(in.rows(), in.rows());
  for (auto i = 0u; i < lambda.size(); ++i) {
    ret += lambda(i) * in.block(0, i * in.rows(), in.rows(), in.rows());
  }
  return ret;
}

/// Sample analytic mesh function (does not allocate)
struct Functor
{
  SparseMatrix<double> jac, hess;

  Functor()
  {
    jac.resize(3, 6);

    jac.coeffRef(0, 0) = 0.;
    jac.coeffRef(0, 1) = 0.;
    jac.coeffRef(0, 5) = 0.;

    jac.coeffRef(1, 3) = 0.;

    jac.coeffRef(2, 0) = 0.;
    jac.coeffRef(2, 2) = 0.;
    jac.coeffRef(2, 4) = 0.;

    jac.makeCompressed();

    hess.resize(6, 3 * 6);

    hess.coeffRef(0, 0 * 6 + 1) = 0;
    hess.coeffRef(0, 0 * 6 + 5) = 0;

    hess.coeffRef(1, 0 * 6 + 0) = 0;
    hess.coeffRef(1, 0 * 6 + 1) = 0;
    hess.coeffRef(1, 0 * 6 + 5) = 0;

    hess.coeffRef(5, 0 * 6 + 0) = 0;
    hess.coeffRef(5, 0 * 6 + 1) = 0;

    // second component

    hess.coeffRef(3, 1 * 6 + 3) = 0;

    // third component

    hess.coeffRef(0, 2 * 6 + 0) = 0;
    hess.coeffRef(0, 2 * 6 + 2) = 0;
    hess.coeffRef(0, 2 * 6 + 4) = 0;

    hess.coeffRef(2, 2 * 6 + 0) = 0;
    hess.coeffRef(2, 2 * 6 + 4) = 0;

    hess.coeffRef(4, 2 * 6 + 0) = 0;
    hess.coeffRef(4, 2 * 6 + 2) = 0;

    hess.makeCompressed();
  }

  template<typename T>
  Vector3<T> operator()(T t, const Vector3<T> & x, const Vector2<T> & u)
  {
    return Vector3<T>{
      t * x.x() * x.x() * u.y(),
      x.z() * x.z(),
      t * t * x.y() * u.x(),
    };
  }

  const SparseMatrix<double> & jacobian(double t, const Vector3d & x, const Vector2d & u)
  {
    jac.coeffRef(0, 0) = x.x() * x.x() * u.y();  // df0 / dt
    jac.coeffRef(0, 1) = t * 2 * x.x() * u.y();  // df0 / dx0
    jac.coeffRef(0, 5) = t * x.x() * x.x();      // df0 / du1

    jac.coeffRef(1, 3) = 2 * x.z();  // df1 / dx2

    jac.coeffRef(2, 0) = 2 * t * x.y() * u.x();  // df2 / dt
    jac.coeffRef(2, 2) = t * t * u.x();          // df2 / dx1
    jac.coeffRef(2, 4) = t * t * x.y();          // df2 / du1

    assert(jac.isCompressed());

    return jac;
  }

  const SparseMatrix<double> & hessian(double t, const Vector3d & x, const Vector2d & u)
  {
    // first component

    hess.coeffRef(0, 0 * 6 + 1) = 2 * x.x() * u.y();  // d2f0 / dtx0
    hess.coeffRef(0, 0 * 6 + 5) = x.x() * x.x();      // d2f0 / dtu1

    hess.coeffRef(1, 0 * 6 + 0) = 2 * x.x() * u.y();  // d2f0 / dx0t
    hess.coeffRef(1, 0 * 6 + 1) = 2 * t * u.y();      // d2f0 / dx0x0
    hess.coeffRef(1, 0 * 6 + 5) = 2 * t * x.x();      // d2f0 / dx0u1

    hess.coeffRef(5, 0 * 6 + 0) = x.x() * x.x();  // d2f0 / du1t
    hess.coeffRef(5, 0 * 6 + 1) = 2 * t * x.x();  // d2f0 / du1x0

    // second component

    hess.coeffRef(3, 1 * 6 + 3) = 2;  // d2f1 / dx2x2

    // third component

    hess.coeffRef(0, 2 * 6 + 0) = 2 * x.y() * u.x();  // d2f2 / dtt
    hess.coeffRef(0, 2 * 6 + 2) = 2 * t * u.x();      // d2f2 / dtx1
    hess.coeffRef(0, 2 * 6 + 4) = 2 * t * x.y();      // d2f2 / dtu0

    hess.coeffRef(2, 2 * 6 + 0) = 2 * t * u.x();  // d2f2 / dx1t
    hess.coeffRef(2, 2 * 6 + 4) = t * t;          // d2f2 / dx1u0

    hess.coeffRef(4, 2 * 6 + 0) = 2 * t * x.y();  // d2f2 / du0t
    hess.coeffRef(4, 2 * 6 + 2) = t * t;          // d2f2 / du0x1

    assert(hess.isCompressed());

    return hess;
  }
};

class MeshFunction_Random : public testing::Test
{
protected:
  Functor f{};

  static constexpr int nx = 3, nu = 2;

  // mesh
  std::size_t N;
  smooth::feedback::Mesh<4, 6> mesh;

  // variables
  double t0 = 1;
  double tf = 6;

  Matrix<double, nx, -1> X;
  Matrix<double, nu, -1> U;

  VectorXd xflat;
  VectorXd uflat;

  void SetUp() override
  {
    mesh.refine_ph(0, 8);
    mesh.refine_ph(0, 5);

    N = mesh.N_colloc();

    X.setRandom(nx, N + 1);
    U.setRandom(nu, N);

    xflat = X.reshaped();
    uflat = U.reshaped();
  }
};

TEST_F(MeshFunction_Random, TestTheTest)
{
  const double t                    = 1;
  const Eigen::Vector<double, nx> x = Eigen::Vector<double, nx>::Random();
  const Eigen::Vector<double, nu> u = Eigen::Vector<double, nu>::Random();

  const auto [fn, dfn, d2fn] = smooth::diff::dr<2>(f, smooth::wrt(t, x, u));
  const auto [fa, dfa, d2fa] =
    smooth::diff::dr<2, smooth::diff::Type::Analytic>(f, smooth::wrt(t, x, u));

  ASSERT_TRUE(fn.isApprox(fa));
  ASSERT_TRUE(dfn.isApprox(Eigen::MatrixXd(dfa), 1e-3));
  ASSERT_TRUE(d2fn.isApprox(Eigen::MatrixXd(d2fa), 1e-3));
}

TEST_F(MeshFunction_Random, EvalDerivative)
{
  using namespace smooth;

  // numerical derivatives of function
  auto fd = [&](double t0, double tf, const VectorXd & xvar, const VectorXd & uvar) -> VectorXd {
    Eigen::Matrix<double, nx, -1> X = xvar.reshaped(nx, N + 1);
    Eigen::Matrix<double, nu, -1> U = uvar.reshaped(nu, N);
    feedback::MeshValue<0> o;
    feedback::mesh_eval(o, mesh, f, t0, tf, X.colwise(), U.colwise());
    return o.F;
  };
  auto [fval, dfval, d2fval] = diff::dr<2, diff::Type::Numerical>(fd, wrt(t0, tf, xflat, uflat));

  // numerical derivative of jacobian
  auto fdj = [&](double t0, double tf, const VectorXd & xvar, const VectorXd & uvar) -> VectorXd {
    Eigen::Matrix<double, nx, -1> X = xvar.reshaped(nx, N + 1);
    Eigen::Matrix<double, nu, -1> U = uvar.reshaped(nu, N);
    feedback::MeshValue<1> o;
    // NOTE: can not double-differentiate!
    feedback::mesh_eval<1, diff::Type::Analytic>(o, mesh, f, t0, tf, X.colwise(), U.colwise());
    return Eigen::MatrixXd(o.dF).transpose().reshaped();
  };
  auto [u0_, d2fval_j] = diff::dr<1, diff::Type::Numerical>(fdj, wrt(t0, tf, xflat, uflat));

  // analytic (analytic inner)
  {
    feedback::MeshValue<2> out;
    out.lambda.setRandom(nx * N);
    feedback::mesh_eval<2, diff::Type::Analytic>(out, mesh, f, t0, tf, X.colwise(), U.colwise());

    // compare values
    ASSERT_TRUE(MatrixXd(out.dF).isApprox(dfval, 1e-3));

    MatrixXd hess = MatrixXd(out.d2F).selfadjointView<Upper>();
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval), 1e-3));
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval_j.transpose()), 1e-3));
  }

  // analytic (numerical inner)
  {
    feedback::MeshValue<2> out;
    out.lambda.setRandom(nx * N);
    feedback::mesh_eval<2, diff::Type::Numerical>(out, mesh, f, t0, tf, X.colwise(), U.colwise());

    // compare values
    ASSERT_TRUE(MatrixXd(out.dF).isApprox(dfval, 1e-3));

    MatrixXd hess = MatrixXd(out.d2F).selfadjointView<Upper>();
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval), 1e-3));
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval_j.transpose()), 1e-3));
  }
}

TEST_F(MeshFunction_Random, IntegralDerivative)
{
  using namespace smooth;

  // numerical derivative of function
  auto fd = [&](double t0, double tf, const VectorXd & xvar, const VectorXd & uvar) -> VectorXd {
    Eigen::Matrix<double, nx, -1> X = xvar.reshaped(nx, N + 1);
    Eigen::Matrix<double, nu, -1> U = uvar.reshaped(nu, N);
    feedback::MeshValue<0> o;
    feedback::mesh_integrate(o, mesh, f, t0, tf, X.colwise(), U.colwise());
    return o.F;
  };
  auto [fval, dfval, d2fval] = diff::dr<2>(fd, wrt(t0, tf, xflat, uflat));

  // numerical derivative of jacobian
  auto fdj = [&](double t0, double tf, const VectorXd & xvar, const VectorXd & uvar) -> VectorXd {
    Eigen::Matrix<double, nx, -1> X = xvar.reshaped(nx, N + 1);
    Eigen::Matrix<double, nu, -1> U = uvar.reshaped(nu, N);
    feedback::MeshValue<1> o;
    // NOTE: can not double-differentiate!
    feedback::mesh_integrate<1, diff::Type::Analytic>(o, mesh, f, t0, tf, X.colwise(), U.colwise());
    return Eigen::MatrixXd(o.dF).transpose().reshaped();
  };
  auto [u0_, d2fval_j] = diff::dr<1, diff::Type::Numerical>(fdj, wrt(t0, tf, xflat, uflat));

  // analytic (analytic inner)
  {
    feedback::MeshValue<2> out;
    out.lambda.setRandom(nx);
    feedback::mesh_integrate<2, diff::Type::Analytic>(
      out, mesh, f, t0, tf, X.colwise(), U.colwise());

    // compare values
    ASSERT_TRUE(MatrixXd(out.dF).isApprox(dfval, 1e-3));
    MatrixXd hess = MatrixXd(out.d2F).selfadjointView<Upper>();
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval), 1e-3));
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval_j.transpose()), 1e-3));
  }

  // analytic (numerical inner)
  {
    feedback::MeshValue<2> out;
    out.lambda.setRandom(nx);
    feedback::mesh_integrate<2, diff::Type::Numerical>(
      out, mesh, f, t0, tf, X.colwise(), U.colwise());

    // compare values
    ASSERT_TRUE(MatrixXd(out.dF).isApprox(dfval, 1e-3));
    MatrixXd hess = MatrixXd(out.d2F).selfadjointView<Upper>();
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval), 1e-3));
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval_j.transpose()), 1e-3));
  }
}

TEST_F(MeshFunction_Random, DynDerivative)
{
  using namespace smooth;

  // numerical derivative of function
  auto fd = [&](double t0, double tf, const VectorXd & xvar, const VectorXd & uvar) -> VectorXd {
    Eigen::Matrix<double, nx, -1> X = xvar.reshaped(nx, N + 1);
    Eigen::Matrix<double, nu, -1> U = uvar.reshaped(nu, N);
    feedback::MeshValue<0> o;
    feedback::mesh_dyn(o, mesh, f, t0, tf, X.colwise(), U.colwise());
    return o.F;
  };
  auto [fval, dfval, d2fval] = diff::dr<2, diff::Type::Numerical>(fd, wrt(t0, tf, xflat, uflat));

  // numerical derivative of jacobian
  auto fdj = [&](double t0, double tf, const VectorXd & xvar, const VectorXd & uvar) -> VectorXd {
    Eigen::Matrix<double, nx, -1> X = xvar.reshaped(nx, N + 1);
    Eigen::Matrix<double, nu, -1> U = uvar.reshaped(nu, N);
    feedback::MeshValue<1> o;
    // NOTE: can not double-differentiate with Numerical!
    feedback::mesh_dyn<1, diff::Type::Analytic>(o, mesh, f, t0, tf, X.colwise(), U.colwise());
    return Eigen::MatrixXd(o.dF).transpose().reshaped();
  };
  auto [u0_, d2fval_j] = diff::dr<1, diff::Type::Numerical>(fdj, wrt(t0, tf, xflat, uflat));

  // analytic (analytic inner)
  {
    feedback::MeshValue<2> out;
    out.lambda.setRandom(nx * N);
    feedback::mesh_dyn<2, diff::Type::Analytic>(out, mesh, f, t0, tf, X.colwise(), U.colwise());

    // compare values
    ASSERT_TRUE(MatrixXd(out.dF).isApprox(dfval, 1e-3));
    MatrixXd hess = MatrixXd(out.d2F).selfadjointView<Upper>();
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval), 1e-3));
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval_j.transpose()), 1e-3));
  }

  // analytic (numerical inner)
  {
    feedback::MeshValue<2> out;
    out.lambda.setRandom(nx * N);
    feedback::mesh_dyn<2, diff::Type::Numerical>(out, mesh, f, t0, tf, X.colwise(), U.colwise());

    // compare values
    ASSERT_TRUE(MatrixXd(out.dF).isApprox(dfval, 1e-3));
    MatrixXd hess = MatrixXd(out.d2F).selfadjointView<Upper>();
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval), 1e-3));
    ASSERT_TRUE(hess.isApprox(blocksum(out.lambda, d2fval_j.transpose()), 1e-3));
  }
}

TEST_F(MeshFunction_Random, EvalAllocationAnalytic)
{
  smooth::feedback::MeshValue<2> out;
  out.lambda.setRandom(nx * N);

  // first call: not allocated
  ASSERT_FALSE(out.allocated);
  smooth::feedback::mesh_eval<2, smooth::diff::Type::Analytic>(
    out, mesh, f, t0, tf, X.colwise(), U.colwise());

  ASSERT_FALSE(out.dF.isCompressed());
  ASSERT_FALSE(out.d2F.isCompressed());

  out.dF.makeCompressed();
  out.d2F.makeCompressed();

  // second call: allocated
  ASSERT_TRUE(out.allocated);
  smooth::feedback::mesh_eval<2, smooth::diff::Type::Analytic>(
    out, mesh, f, t0, tf, X.colwise(), U.colwise());

  // expect to still be compressed (means we only touched existing coeffs)
  ASSERT_TRUE(out.dF.isCompressed());
  ASSERT_TRUE(out.d2F.isCompressed());
}

TEST_F(MeshFunction_Random, EvalAllocationNumerical)
{
  smooth::feedback::MeshValue<2> out;
  out.lambda.setRandom(nx * N);

  // first call: not allocated
  ASSERT_FALSE(out.allocated);
  smooth::feedback::mesh_eval<2>(out, mesh, f, t0, tf, X.colwise(), U.colwise());

  ASSERT_FALSE(out.dF.isCompressed());
  ASSERT_FALSE(out.d2F.isCompressed());

  out.dF.makeCompressed();
  out.d2F.makeCompressed();

  // numerical eval gives non-sparse derivatives, so we expect all allocated memory to be used
  ASSERT_EQ(out.dF.nonZeros(), out.dF.data().size());
  ASSERT_EQ(out.d2F.nonZeros(), out.d2F.data().size());

  // second call: allocated
  ASSERT_TRUE(out.allocated);
  smooth::feedback::mesh_eval<2>(out, mesh, f, t0, tf, X.colwise(), U.colwise());

  // expect to still be compressed (means we only touched existing coeffs)
  ASSERT_TRUE(out.dF.isCompressed());
  ASSERT_TRUE(out.d2F.isCompressed());
}

TEST_F(MeshFunction_Random, IntegrateAllocationAnalytic)
{
  smooth::feedback::MeshValue<2> out;
  out.lambda.setRandom(nx);

  // first call: not allocated
  ASSERT_FALSE(out.allocated);
  smooth::feedback::mesh_integrate<2, smooth::diff::Type::Analytic>(
    out, mesh, f, t0, tf, X.colwise(), U.colwise());

  ASSERT_FALSE(out.dF.isCompressed());
  ASSERT_FALSE(out.d2F.isCompressed());

  out.dF.makeCompressed();
  out.d2F.makeCompressed();

  // second call: allocated
  ASSERT_TRUE(out.allocated);
  smooth::feedback::mesh_integrate<2, smooth::diff::Type::Analytic>(
    out, mesh, f, t0, tf, X.colwise(), U.colwise());

  // expect to still be compressed (means we only touched existing coeffs)
  ASSERT_TRUE(out.dF.isCompressed());
  ASSERT_TRUE(out.d2F.isCompressed());
}

TEST_F(MeshFunction_Random, IntegrateAllocationNumerical)
{
  smooth::feedback::MeshValue<2> out;
  out.lambda.setRandom(nx);

  // first call: not allocated
  ASSERT_FALSE(out.allocated);
  smooth::feedback::mesh_integrate<2>(out, mesh, f, t0, tf, X.colwise(), U.colwise());

  ASSERT_FALSE(out.dF.isCompressed());
  ASSERT_FALSE(out.d2F.isCompressed());

  // numerical eval gives non-sparse derivatives, so we expect all allocated memory to be used
  ASSERT_EQ(out.dF.nonZeros(), out.dF.data().size());
  ASSERT_EQ(out.d2F.nonZeros(), out.d2F.data().size());

  out.dF.makeCompressed();
  out.d2F.makeCompressed();

  // second call: allocated
  ASSERT_TRUE(out.allocated);
  smooth::feedback::mesh_integrate<2>(out, mesh, f, t0, tf, X.colwise(), U.colwise());

  // expect to still be compressed (means we only touched existing coeffs)
  ASSERT_TRUE(out.dF.isCompressed());
  ASSERT_TRUE(out.d2F.isCompressed());
}

TEST_F(MeshFunction_Random, DynAllocationAnalytic)
{
  smooth::feedback::MeshValue<2> out;
  out.lambda.setRandom(nx * N);

  // first call: not allocated
  ASSERT_FALSE(out.allocated);
  smooth::feedback::mesh_dyn<2, smooth::diff::Type::Analytic>(
    out, mesh, f, t0, tf, X.colwise(), U.colwise());

  ASSERT_FALSE(out.dF.isCompressed());
  ASSERT_FALSE(out.d2F.isCompressed());

  out.dF.makeCompressed();
  out.d2F.makeCompressed();

  // second call: allocated
  ASSERT_TRUE(out.allocated);
  smooth::feedback::mesh_dyn<2, smooth::diff::Type::Analytic>(
    out, mesh, f, t0, tf, X.colwise(), U.colwise());

  // expect to still be compressed (means we only touched existing coeffs)
  ASSERT_TRUE(out.dF.isCompressed());
  ASSERT_TRUE(out.d2F.isCompressed());
}

TEST_F(MeshFunction_Random, DynAllocationNumerical)
{
  smooth::feedback::MeshValue<2> out;
  out.lambda.setRandom(nx * N);

  // first call: not allocated
  ASSERT_FALSE(out.allocated);
  smooth::feedback::mesh_dyn<2>(out, mesh, f, t0, tf, X.colwise(), U.colwise());

  ASSERT_FALSE(out.dF.isCompressed());
  ASSERT_FALSE(out.d2F.isCompressed());

  // numerical eval gives non-sparse derivatives, so we expect all allocated memory to be used
  ASSERT_EQ(out.dF.nonZeros(), out.dF.data().size());
  ASSERT_EQ(out.d2F.nonZeros(), out.d2F.data().size());

  out.dF.makeCompressed();
  out.d2F.makeCompressed();

  // second call: allocated
  ASSERT_TRUE(out.allocated);
  smooth::feedback::mesh_dyn<2>(out, mesh, f, t0, tf, X.colwise(), U.colwise());

  // expect to still be compressed (means we only touched existing coeffs)
  ASSERT_TRUE(out.dF.isCompressed());
  ASSERT_TRUE(out.d2F.isCompressed());
}

/**
 * @brief Test using time-defined trajectory x(t) = 0.1 t^2 - 0.4 t + 0.2
 */
class MeshFunction_Traj1 : public testing::Test
{
protected:
  static constexpr std::size_t nx = 1, nu = 0, nq = 1;

  static constexpr auto df_dt =
    []<typename T>(const T & t, const Vector<T, nx> &, const Vector<T, nu> &) -> Vector<T, nx> {
    return Vector<T, nx>{{0.2 * t - 0.4}};
  };

  // variables
  const double t0 = 3, tf = 5;
  Matrix<double, nx, -1> X;
  Matrix<double, nu, -1> U;

  // mesh
  smooth::feedback::Mesh<5, 5> m;

  void SetUp() override
  {
    m.refine_ph(0, 40);
    const auto N = m.N_colloc();

    X.setRandom(nx, N + 1);
    U.setRandom(nu, N);

    for (const auto & [i, tau] : smooth::utils::zip(std::views::iota(0u), m.all_nodes())) {
      const auto s = t0 + (tf - t0) * tau;
      X(i)         = 0.1 * s * s - 0.4 * s + 0.2;
    }
  }
};

TEST_F(MeshFunction_Traj1, Integral)
{
  const auto g =
    []<typename T>(const T &, const Vector<T, nx> & x, const Vector<T, nu> &) -> Vector<T, nq> {
    return Vector<T, nq>{{0.1 + x.squaredNorm()}};
  };

  smooth::feedback::MeshValue<0> out;
  smooth::feedback::mesh_integrate(out, m, g, t0, tf, X.colwise(), U.colwise());
  ASSERT_NEAR(out.F.x(), 0.217333 + 0.1 * (tf - t0), 1e-4);
}

TEST_F(MeshFunction_Traj1, Dynamics)
{
  smooth::feedback::MeshValue<1> out;
  smooth::feedback::mesh_dyn<1>(out, m, df_dt, t0, tf, X.colwise(), U.colwise());

  ASSERT_LE(out.F.cwiseAbs().maxCoeff(), 1e-8);
}

/**
 * @brief Test using time-defined trajectory x(t) = 1.5 exp(-t)
 */
class MeshFunction_Traj2 : public testing::Test
{
protected:
  static constexpr std::size_t nx = 1, nu = 0, nq = 1;

  static constexpr auto df_dt =
    []<typename T>(const T &, const Vector<T, nx> & x, const Vector<T, nu> &) -> Vector<T, nx> {
    return Vector<T, nx>{{-x.x()}};
  };

  // variables
  const double t0 = 3, tf = 5;
  Matrix<double, nx, -1> X;
  Matrix<double, nu, -1> U;

  // mesh
  smooth::feedback::Mesh<5, 5> m;

  void SetUp() override
  {
    m.refine_ph(0, 40);
    const auto N = m.N_colloc();

    X.setRandom(nx, N + 1);
    U.setRandom(nu, N);

    for (const auto & [i, tau] : smooth::utils::zip(std::views::iota(0u), m.all_nodes())) {
      const auto s = t0 + (tf - t0) * tau;
      X(i)         = 1.5 * exp(-s);
    }
  }
};

TEST_F(MeshFunction_Traj2, Integral)
{
  const auto g =
    []<typename T>(const T &, const Vector<T, nx> & x, const Vector<T, nu> &) -> Vector<T, nq> {
    return Vector<T, nq>{{x.squaredNorm()}};
  };

  smooth::feedback::MeshValue<0> out;
  smooth::feedback::mesh_integrate(out, m, g, t0, tf, X.colwise(), U.colwise());
  ASSERT_NEAR(out.F.x(), 0.00273752, 1e-4);
}

TEST_F(MeshFunction_Traj2, Dynamics)
{
  smooth::feedback::MeshValue<1> out;
  smooth::feedback::mesh_dyn<1>(out, m, df_dt, t0, tf, X.colwise(), U.colwise());
  ASSERT_LE(out.F.cwiseAbs().maxCoeff(), 1e-8);
}
