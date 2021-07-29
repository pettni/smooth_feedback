#include <boost/numeric/odeint.hpp>
#include <smooth/compat/autodiff.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/mpc.hpp>
#include <smooth/se2.hpp>

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
  auto f = []<typename T>(const G<T> & x, const U<T> & u) ->
    typename G<T>::Tangent { return typename G<T>::Tangent(x.rn()(1), u(0)); };

  // create MPC object and set input bounds, and desired trajectories
  smooth::feedback::MPC<nMpc, Time, Gd, Ud, decltype(f)> mpc(f, 5s);
  mpc.set_ulim(Eigen::Matrix<double, 1, 1>(-0.5), Eigen::Matrix<double, 1, 1>(0.5));
  mpc.set_xudes([](Time t) -> Gd { return Gd(Eigen::Vector2d(sin(0.5 * t.count()), 0)); },
    [](Time) -> Ud { return Ud::Zero(); });

  // prepare for integrating the closed-loop system
  runge_kutta4<Gd, double, typename Gd::Tangent, double, vector_space_algebra> stepper{};
  const auto ode = [&f, &u](const Gd & x, typename Gd::Tangent & d, double) { d = f(x, u); };
  std::vector<double> tvec, xvec, vvec, uvec;

  // integrate closed-loop system
  for (std::chrono::milliseconds t = 20s; t < 40s; t += 50ms) {
    // compute MPC input
    u = mpc(t, g);

    // store data
    tvec.push_back(t.count());
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
  matplot::plot(tvec, vvec)->line_width(2);
  matplot::plot(tvec, uvec)->line_width(2);
  matplot::legend({"x", "v", "u"});

  matplot::show();
#else
  std::cout << "TRAJECTORY:" << std::endl;
  for (auto i = 0u; i != tvec.size(); ++i) {
    std::cout << "t=" << tvec[i] << ": x=" << xvec[i] << ", v=" << vvec[i] << std::endl;
  }
#endif
}
