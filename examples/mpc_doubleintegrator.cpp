#include <boost/numeric/odeint.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/mpc.hpp>

#include <chrono>

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#endif

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;

using Time = std::chrono::duration<double>;

template<typename T>
using G = Eigen::Vector2<T>;
template<typename T>
using U = Eigen::Matrix<T, 1, 1>;

using Gd = G<double>;
using Ud = U<double>;

int main()
{
  using std::sin;
  std::srand(5);

  // system variables
  Gd g = Gd::Random();
  Ud u;

  // dynamics
  auto f = []<typename T>(Time, const G<T> & x, const U<T> u) -> smooth::Tangent<G<T>> {
    return {x(1), u(0)};
  };

  // parameters
  smooth::feedback::MPCParams<Gd, Ud> prm{
    .T = 5,
    .K = 20,
    .weights =
      {
        .Q  = Eigen::Matrix2d::Identity(),
        .QT = 0.1 * Eigen::Matrix2d::Identity(),
        .R  = Eigen::Matrix<double, 1, 1>::Constant(0.1),
      },
    .ulim =
      smooth::feedback::ManifoldBounds<Ud>{
        .A = Eigen::Matrix<double, 1, 1>(1),
        .c = Ud::Zero(),
        .l = Eigen::Matrix<double, 1, 1>(-0.5),
        .u = Eigen::Matrix<double, 1, 1>(0.5),
      },
  };

  // create MPC object and set input bounds, and desired trajectories
  smooth::feedback::MPC<Time, Gd, Ud, decltype(f)> mpc(f, prm);
  mpc.set_xdes([](Time t) -> std::pair<Gd, smooth::Tangent<Gd>> {
    return {
      Gd{-0.5 * sin(0.3 * t.count()), 0},
      smooth::Tangent<Gd>{-0.15 * cos(0.3 * t.count()), 0},
    };
  });
  mpc.set_udes([](Time) -> Ud { return Ud::Zero(); });

  // prepare for integrating the closed-loop system
  runge_kutta4<Gd, double, smooth::Tangent<Gd>, double, vector_space_algebra> stepper{};
  const auto ode = [&f, &u](const Gd & x, smooth::Tangent<Gd> & d, double t) -> void {
    d = f(Time(t), x, u);
  };
  std::vector<double> tvec, xvec, vvec, uvec;

  // integrate closed-loop system
  for (std::chrono::milliseconds t = 0s; t < 60s; t += 50ms) {
    // compute MPC input
    auto code = mpc(u, t, g);
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

#ifdef ENABLE_PLOTTING
  matplot::figure();
  matplot::hold(matplot::on);

  matplot::plot(tvec, xvec)->line_width(2);
  matplot::plot(tvec, matplot::transform(tvec, [](auto t) { return -0.5 * sin(0.3 * t); }), "k--")
    ->line_width(2);
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
