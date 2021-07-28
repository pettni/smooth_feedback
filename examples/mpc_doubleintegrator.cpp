#include <boost/numeric/odeint.hpp>
#include <smooth/compat/autodiff.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/mpc.hpp>
#include <smooth/se2.hpp>

#include <chrono>
#include <matplot/matplot.h>

template<typename T>
using G = smooth::T2<T>;
template<typename T>
using U = smooth::T1<T>;

using Gd = G<double>;
using Ud = U<double>;

using namespace boost::numeric::odeint;

int main()
{
  std::srand(5);
  static constexpr int nMpc = 20;
  Gd g;
  Ud u;
  g.setRandom();
  std::chrono::nanoseconds t(1234);
  std::chrono::nanoseconds dt = std::chrono::milliseconds(50);
  double dt_dbl               = std::chrono::duration<double>(dt).count();

  auto f = []<typename T>(const G<T> & x, const U<T> & u) {
    return typename G<T>::Tangent(x.rn()(1), u.rn()(0));
  };

  smooth::feedback::MPC<nMpc, Gd, Ud, decltype(f)> mpc(f);

  mpc.set_xudes([](auto) -> smooth::T2d { return smooth::T2d(Eigen::Vector2d(0.2, 0)); },
    [](auto) -> smooth::T1d { return smooth::T1d::Identity(); });

  runge_kutta4<Gd, double, typename Gd::Tangent, double, vector_space_algebra> stepper{};

  const auto ode = [&f, &u](const Gd & x, typename Gd::Tangent & d, double) { d = f(x, u); };

  std::vector<double> tvec, xvec, yvec, uvec;

  for (auto i = 0u; i != 200; ++i) {
    // compute MPC input
    u = mpc(t, g);

    // store data
    tvec.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(t).count());
    xvec.push_back(g.rn().x());
    yvec.push_back(g.rn().y());
    uvec.push_back(u.rn()(0));

    // step dynamics
    stepper.do_step(ode, g, double(0.), dt_dbl);
    t += dt;
  }

  matplot::figure();
  matplot::hold(matplot::on);

  matplot::plot(tvec, xvec)->line_width(2);
  matplot::plot(tvec, yvec)->line_width(2);
  matplot::plot(tvec, uvec)->line_width(2);
  matplot::legend({"x", "v", "u"});

  matplot::show();
}
