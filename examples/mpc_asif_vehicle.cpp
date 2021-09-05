#include <boost/numeric/odeint.hpp>
#include <smooth/bundle.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/asif.hpp>
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
using G = smooth::Bundle<smooth::SE2<T>, Eigen::Matrix<T, 3, 1>>;
template<typename T>
using U = Eigen::Matrix<T, 2, 1>;

using Gd = G<double>;
using Ud = U<double>;

using Tangentd = typename Gd::Tangent;

int main()
{
  // number of MPC discretization points steps
  static constexpr int nMpc = 50;
  double T                  = 5;

  // system variables
  Gd g = Gd::Identity();
  Ud u;

  // body velocity of desired trajectory
  Eigen::Vector3d vdes{1, 0, 0.4};

  // dynamics
  auto f = []<typename T>(Time, const G<T> & x, const U<T> & u) -> smooth::Tangent<G<T>> {
    return {
      x.template part<1>().x(),
      x.template part<1>().y(),
      x.template part<1>().z(),
      -T(0.2) * x.template part<1>().x() + u.x(),
      T(0),
      -T(0.4) * x.template part<1>().z() + u.y(),
    };
  };

  // asif dynamics
  auto f_asif = []<typename T>(T, const G<T> & x, const U<T> & u) -> smooth::Tangent<G<T>> {
    return {
      x.template part<1>().x(),
      x.template part<1>().y(),
      x.template part<1>().z(),
      -T(0.2) * x.template part<1>().x() + u.x(),
      T(0),
      -T(0.4) * x.template part<1>().z() + u.y(),
    };
  };

  // Input bounds
  const smooth::feedback::OptimalControlBounds<Ud> ulim{
    .l = Eigen::Vector2d(-0.2, -0.5),
    .u = Eigen::Vector2d(0.5, 0.5),
  };

  // SET UP MPC

  smooth::feedback::MPC<nMpc, Time, Gd, Ud, decltype(f)> mpc(f,
    smooth::feedback::MPCParams{
      .T         = T,
      .warmstart = true,
    });

  // set input bounds
  mpc.set_ulim(ulim);

  // set weights
  Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
  mpc.set_input_cost(R);
  Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Identity();
  mpc.set_running_state_cost(Q);
  mpc.set_final_state_cost(0.1 * Q);

  // set desired trajectory
  auto xdes = [&vdes](Time t) -> Gd {
    return Gd(
      smooth::SE2d(smooth::SO2d(M_PI_2), Eigen::Vector2d(2.5, 0)) + (t.count() * vdes), vdes);
  };
  auto udes = [](Time t) -> Ud { return Ud::Zero(); };
  mpc.set_xudes(xdes, udes);

  // SET UP ASIF

  // safe set
  auto h = []<typename T>(T, const G<T> & g) -> Eigen::Matrix<T, 1, 1> {
    const Eigen::Vector2<T> dir = g.template part<0>().r2() - Eigen::Vector2<T>{-2.5, 0};
    const Eigen::Vector2d e_dir = dir.template cast<double>().normalized();
    return Eigen::Matrix<T, 1, 1>(dir.dot(e_dir) - 0.5);
  };

  // backup controller
  auto bu = []<typename T>(T, const G<T> & g) -> U<T> {
    return {0.2 * g.template part<1>().x(), -0.5};
  };

  smooth::feedback::ASIFParams<Ud> asif_prm{
    .T        = T,
    .u_weight = Eigen::Vector2d{100, 1},
    .u_lim    = ulim,
    .asif =
      {
        .alpha      = 10,
        .dt         = 0.05,
        .relax_cost = 10,
      },
  };

  smooth::feedback::ASIF<nMpc, Gd, Ud, decltype(f_asif), decltype(h), decltype(bu)> asif(
    f_asif, h, bu, asif_prm);

  // prepare for integrating the closed-loop system
  runge_kutta4<Gd, double, Tangentd, double, vector_space_algebra> stepper{};
  const auto ode = [&f, &u](const Gd & x, Tangentd & d, double t) { d = f(Time(t), x, u); };
  std::vector<double> tvec, xvec, yvec, u1vec, u2vec;

  // integrate closed-loop system
  for (std::chrono::milliseconds t = 0s; t < 30s; t += 50ms) {
    // compute MPC input
    auto mpc_code = mpc(u, t, g);
    if (mpc_code != smooth::feedback::QPSolutionStatus::Optimal) {
      std::cerr << "MPC failed with mpc_code " << static_cast<int>(mpc_code) << std::endl;
    }

    // filter input with ASIF
    auto asif_code = asif(u, 0, g);
    if (asif_code != smooth::feedback::QPSolutionStatus::Optimal) {
      std::cerr << "ASIF solver failed with asif_code " << static_cast<int>(asif_code) << std::endl;
    }

    // store data
    tvec.push_back(duration_cast<Time>(t).count());
    xvec.push_back(g.template part<0>().r2().x());
    yvec.push_back(g.template part<0>().r2().y());

    u1vec.push_back(u(0));
    u2vec.push_back(u(1));

    // step dynamics
    stepper.do_step(ode, g, 0, 0.05);
  }

#ifdef ENABLE_PLOTTING
  matplot::figure();
  matplot::hold(matplot::on);
  matplot::title("Path");

  matplot::plot(xvec, yvec)->line_width(2);
  matplot::plot(
    matplot::transform(tvec, [&](auto t) { return xdes(Time(t)).template part<0>().r2().x(); }),
    matplot::transform(tvec, [&](auto t) { return xdes(Time(t)).template part<0>().r2().y(); }),
    "k--")
    ->line_width(2);
  matplot::legend({"actual", "desired"});

  matplot::figure();
  matplot::hold(matplot::on);
  matplot::title("Inputs");
  matplot::plot(tvec, u1vec)->line_width(2);
  matplot::plot(tvec, u2vec)->line_width(2);
  matplot::legend({"u1", "u2"});

  matplot::show();
#else
  std::cout << "TRAJECTORY:" << std::endl;
  for (auto i = 0u; i != tvec.size(); ++i) {
    std::cout << "t=" << tvec[i] << ": x=" << xvec[i] << ", y=" << yvec[i] << std::endl;
  }
#endif
}
