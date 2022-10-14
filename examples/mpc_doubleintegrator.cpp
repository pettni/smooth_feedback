// Copyright (C) 2022 Petter Nilsson. MIT License.

#include <chrono>

#include <boost/numeric/odeint.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/mpc.hpp>

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#endif

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;

using Time = std::chrono::duration<double>;

template<typename T>
using X = Eigen::Vector2<T>;
template<typename T>
using U = Eigen::Matrix<T, 1, 1>;

using Gd = X<double>;
using Ud = U<double>;

int main()
{
  using std::sin;
  std::srand(5);

  // system variables
  Gd g = Gd::Random();
  Ud u;

  // dynamics
  auto f = []<typename S>(const X<S> & x, const U<S> u) -> smooth::Tangent<X<S>> { return {x(1), u(0)}; };

  // running constraints
  auto cr = []<typename S>(const X<S> &, const U<S> & u) -> Eigen::Vector<S, 1> { return u; };
  Eigen::Vector<double, 1> crl{-0.5}, cru{0.5};

  // create MPC object and set input bounds, and desired trajectories
  smooth::feedback::MPC<Time, Gd, Ud, decltype(f), decltype(cr)> mpc{
    f,
    cr,
    crl,
    cru,
    {
      .K  = 20,
      .tf = 5,
      .qp = {.scaling = false, .polish = false},
    },
  };

  mpc.set_weights({
    .Q   = Eigen::Matrix2d::Identity(),
    .Qtf = 0.1 * Eigen::Matrix2d::Identity(),
    .R   = 0.1 * Eigen::Matrix<double, 1, 1>::Identity(),
  });
  mpc.set_xdes_rel([]<typename T>(T t) -> X<T> { return X<T>{-0.5 * sin(0.3 * t), 0}; });
  mpc.set_udes_rel([]<typename T>(T) -> U<T> { return U<T>::Zero(); });

  // prepare for integrating the closed-loop system
  runge_kutta4<Gd, double, smooth::Tangent<Gd>, double, vector_space_algebra> stepper{};
  const auto ode = [&f, &u](const Gd & x, smooth::Tangent<Gd> & d, double) { d = f(x, u); };
  std::vector<double> tvec, xvec, vvec, uvec;

  // integrate closed-loop system
  const auto t0 = std::chrono::high_resolution_clock::now();

  for (std::chrono::milliseconds t = 0s; t < 60s; t += 50ms) {
    // compute MPC input
    auto [u_mpc, code] = mpc(t, g);
    u                  = u_mpc;
    if (code != smooth::feedback::QPSolutionStatus::Optimal) {
      std::cerr << "Solver failed with code " << static_cast<int>(code) << std::endl;
    }

    // store data
    tvec.push_back(duration_cast<Time>(t).count());
    xvec.push_back(g.x());
    vvec.push_back(g.y());
    uvec.push_back(u(0));

    // step dynamics
    stepper.do_step(ode, g, 0, 0.05);
  }

  const auto tf = std::chrono::high_resolution_clock::now();

  std::cout << "MPC loop time: " << std::chrono::duration_cast<std::chrono::microseconds>(tf - t0).count() << "us\n";

#if ENABLE_PLOTTING
  matplot::figure();
  matplot::hold(matplot::on);

  matplot::plot(tvec, xvec)->line_width(2);
  matplot::plot(tvec, matplot::transform(tvec, [](auto t) { return -0.5 * sin(0.3 * t); }), "k--")->line_width(2);
  matplot::plot(tvec, vvec)->line_width(2);
  matplot::plot(tvec, uvec)->line_width(2);
  matplot::legend({"x", "x_{des}", "v", "u"});

  matplot::show();
#else
  std::cout << "TRAJECTORY:" << std::endl;
  for (auto i = 0u; i != tvec.size(); ++i) {
    std::cout << "t=" << tvec[i] << ": x=" << xvec[i] << ", v=" << vvec[i] << std::endl;
  }
#endif
}
