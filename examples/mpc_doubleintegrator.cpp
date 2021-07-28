#include <boost/numeric/odeint.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/mpc.hpp>
#include <smooth/se2.hpp>

#include <matplot/matplot.h>
#include <chrono>

template<typename T>
using G = smooth::T2<T>;
template<typename T>
using U = smooth::T1<T>;

using Gd = G<double>;
using Ud = U<double>;

int main()
{
  using namespace boost::numeric::odeint;

  std::srand(5);

  smooth::feedback::SolverParams prm{};

  smooth::feedback::OptimalControlProblem<Gd, Ud> ocp{};

  ocp.gdes = []<typename T>(const T &) -> smooth::T2<T> {
    return smooth::T2<T>(Eigen::Matrix<T, 2, 1>(-0.2, 0));
  };

  ocp.udes = []<typename T>(const T &) -> smooth::T1<T> { return smooth::T1<T>::Identity(); };

  ocp.R.setIdentity();
  ocp.Q.diagonal().setConstant(2);
  ocp.QT.setIdentity();
  ocp.T                     = 5;
  static constexpr int nMpc = 20;

  const auto f = []<typename T>(const smooth::T2<T> & x, const smooth::T1<T> & u) {
    return Eigen::Matrix<T, 2, 1>(x.rn()(1), u.rn()(0));
  };

  const auto glin = []<typename T>(const T &) { return smooth::T2<T>::Identity(); };
  const auto ulin = []<typename T>(const T &) { return smooth::T1<T>::Identity(); };

  Gd g;
  Ud u;
  g.setRandom();
  double t                   = 0;
  static constexpr double dt = 0.05;

  using state_t = Gd;
  using deriv_t = Eigen::Vector2d;

  runge_kutta4<state_t, double, deriv_t, double, vector_space_algebra> stepper{};

  const auto ode = [&f, &u](const state_t & x, deriv_t & d, double) { d = f(x, u); };

  std::vector<double> tvec;
  std::vector<double> xvec;
  std::vector<double> yvec;
  std::vector<double> uvec;

  for (auto i = 0u; i != 200; ++i) {
    // solve MPC
    ocp.x0 = g;
    auto qp = smooth::feedback::ocp_to_qp<nMpc>(ocp, f, glin, ulin);
    auto sol = smooth::feedback::solve_qp(qp, prm);

    Eigen::Matrix<double, -1, 1> sol_u = sol.primal.head<nMpc>();
    Eigen::Matrix<double, -1, 1> sol_x = sol.primal.tail<2 * nMpc>();

    // store data
    tvec.push_back(t);
    xvec.push_back(g.rn().x());
    yvec.push_back(g.rn().y());
    uvec.push_back(sol_u(0));

    // step dynamics
    u.rn() = sol_u.head<1>();
    stepper.do_step(ode, g, t, dt);
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
