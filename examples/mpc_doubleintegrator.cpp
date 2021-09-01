#include <boost/numeric/odeint.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/mpc.hpp>
#include <smooth/tn.hpp>

#include <chrono>

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#endif

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;

using Time = std::chrono::duration<double>;

template<typename T>
using G = smooth::T2<T>;
template<typename T>
using U = Eigen::Matrix<T, 1, 1>;

using Gd = G<double>;
using Ud = U<double>;

using Tangentd = typename Gd::Tangent;

int main()
{
  using std::sin;
  std::srand(5);

  // number of MPC discretization points steps
  static constexpr int nMpc = 20;

  // system variables
  Gd g = Gd::Random();
  Ud u;

  // dynamics
  auto f = []<typename T>(Time, const G<T> & x, const U<T> u) ->
    typename G<T>::Tangent { return typename G<T>::Tangent(x.rn()(1), u(0)); };

  // parameters
  smooth::feedback::MPCParams prm{.T = 5};

  // create MPC object and set input bounds, and desired trajectories
  smooth::feedback::MPC<nMpc, Time, Gd, Ud, decltype(f)> mpc(f, prm);
  mpc.set_ulim(smooth::feedback::OptimalControlBounds<Ud>{
    .l = Eigen::Matrix<double, 1, 1>(-0.5),
    .u = Eigen::Matrix<double, 1, 1>(0.5),
  });
  mpc.set_input_cost(Eigen::Matrix<double, 1, 1>::Constant(0.1));
  mpc.set_running_state_cost(Eigen::Matrix<double, 2, 2>::Identity());
  mpc.set_final_state_cost(0.1 * Eigen::Matrix<double, 2, 2>::Identity());
  mpc.set_xudes([](Time t) -> Gd { return Gd(Eigen::Vector2d(-0.5 * sin(0.3 * t.count()), 0)); },
    [](Time) -> Ud { return Ud::Zero(); });

  // prepare for integrating the closed-loop system
  runge_kutta4<Gd, double, Tangentd, double, vector_space_algebra> stepper{};
  const auto ode = [&f, &u](const Gd & x, Tangentd & d, double t) { d = f(Time(t), x, u); };
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
    xvec.push_back(g.rn().x());
    vvec.push_back(g.rn().y());
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
