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

#include <smooth/diff.hpp>
#include <smooth/se2.hpp>
#include <smooth/se3.hpp>
#include <smooth/so3.hpp>

using G = smooth::SE3d;

struct Func
{
  G xd;

  double operator()(const G & x) const { return (x - xd).squaredNorm() / 2; }

  Eigen::RowVector<double, smooth::Dof<G>> jacobian(const G & x) const
  {
    const smooth::Tangent<G> e = x - xd;
    return e.transpose() * G::dr_expinv(e);
  }

  Eigen::Matrix<double, smooth::Dof<G>, smooth::Dof<G>> hessian(const G & x) const
  {
    const smooth::Tangent<G> e = x - xd;

    const smooth::TangentMap<G> dexpinv = G::dr_expinv(e);

    using At = Eigen::Matrix<double, 2 * smooth::Dof<G>, 2 * smooth::Dof<G>>;

    At A;
    A.topLeftCorner<smooth::Dof<G>, smooth::Dof<G>>()  = G::ad(e).transpose();
    A.topRightCorner<smooth::Dof<G>, smooth::Dof<G>>() = G::ad(e).transpose();
    A.bottomLeftCorner<smooth::Dof<G>, smooth::Dof<G>>().setZero();
    A.bottomRightCorner<smooth::Dof<G>, smooth::Dof<G>>() = G::ad(e).transpose();

    const double B0 = 1.;
    const double B1 = 1. / 2;
    const double B2 = 1. / 6;
    const double B4 = -1. / 30;
    const double B6 = 1. / 42;
    const double B8 = -1. / 32;

    At res = B0 * At::Identity() + B1 * A + B2 * A * A / 2 + B4 * A * A * A * A / 24
           + B6 * A * A * A * A * A * A / 720 + B8 * A * A * A * A * A * A * A * A / 40320;

    std::cout << "CHECK A\n" << dexpinv.transpose() << '\n';
    std::cout << "CHECK B\n" << res.topLeftCorner<smooth::Dof<G>, smooth::Dof<G>>() << '\n';

    std::cout << "topright\n";
    std::cout << res.topRightCorner<smooth::Dof<G>, smooth::Dof<G>>() << '\n';

    return dexpinv.transpose() * dexpinv;
  }
};

TEST(Exp, Basic)
{
  for (auto i = 0u; i < 10; ++i) {
    const auto f = Func{.xd = G::Random()};
    const auto x = G::Random();

    const auto [f_n, drf_n, d2rf_n] =
      smooth::diff::dr<2, smooth::diff::Type::Numerical>(f, smooth::wrt(x));
    const auto [f_a, drf_a, d2rf_a] =
      smooth::diff::dr<2, smooth::diff::Type::Analytic>(f, smooth::wrt(x));

    std::cout << "num 1: \n" << drf_n << '\n';
    std::cout << "ana 1: \n" << drf_a << '\n';
    std::cout << "num 2: \n" << d2rf_n << '\n';
    std::cout << "ana 2: \n" << d2rf_a << '\n';
  }
}

TEST(Exp, Hypothesis)
{
  const smooth::Tangent<G> a = smooth::Tangent<G>::Random();
  const smooth::Tangent<G> c = smooth::Tangent<G>::Random();

  auto exp_v = [&a, &c](double t) -> Eigen::VectorXd { return G::dr_exp(a + t * c).reshaped(); };

  // calculate (d/da) b' drexp_a numerically
  const double ww = 0;
  const auto [f_n, drf_n] =
    smooth::diff::dr<1, smooth::diff::Type::Numerical>(exp_v, smooth::wrt(ww));

  // calculate (d/da) drexp_a in direction c
  using At = Eigen::Matrix<double, 2 * smooth::Dof<G>, 2 * smooth::Dof<G>>;

  At A;
  A.topLeftCorner<smooth::Dof<G>, smooth::Dof<G>>()  = G::ad(a);
  A.topRightCorner<smooth::Dof<G>, smooth::Dof<G>>() = G::ad(c);
  A.bottomLeftCorner<smooth::Dof<G>, smooth::Dof<G>>().setZero();
  A.bottomRightCorner<smooth::Dof<G>, smooth::Dof<G>>() = G::ad(a);

  At test = At::Identity() - A / 2 + A * A / 6 - A * A * A / 24 + A * A * A * A / 120
          - A * A * A * A * A / 720;

  std::cout << "Numerical\n";
  std::cout << drf_n.reshaped(smooth::Dof<G>, smooth::Dof<G>) << std::endl;
  std::cout << "Formula\n";
  std::cout << test.topRightCorner<smooth::Dof<G>, smooth::Dof<G>>() << std::endl;
}
