#include <boost/numeric/odeint.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/asif.hpp>

#include <chrono>

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#endif

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;

using Time = std::chrono::duration<double>;

template<typename T>
using G = Eigen::Matrix<T, 2, 1>;
template<typename T>
using U = Eigen::Matrix<T, 1, 1>;

using Gd = G<double>;
using Ud = U<double>;

int main()
{
  using std::sin;
  std::srand(5);

  // number of ASIF constraints
  static constexpr int nAsif = 50;

  // system variables
  Gd g(-5, 1);
  Ud udes = Ud(1), u;

  smooth::feedback::AsifParams prm{.alpha = 1, .tau = 0.01, .dt = 0.01, .relax_cost = 500};

  smooth::feedback::QPSolverParams qpprm{
    // .verbose = true,
    .scaling = true,
    // .max_time = std::chrono::microseconds(100),
    .polish = true,
  };

  // dynamics
  auto f = []<typename T>(T, const G<T> & x, const U<T> & u) -> smooth::Tangent<G<T>> {
    return {x(1), u(0)};
  };

  // safety set
  auto h = []<typename T>(T, const G<T> & g) -> Eigen::Matrix<T, 2, 1> {
    return {T(3) - g(0), T(1.5) - g(1)};
  };

  // backup controller
  auto bu = []<typename T>(T, const G<T> & g) -> U<T> { return U<T>(-0.6); };

  // prepare for integrating the closed-loop system
  runge_kutta4<Gd, double, smooth::Tangent<Gd>, double, vector_space_algebra> stepper{};
  const auto ode = [&f, &u](const Gd & x, smooth::Tangent<Gd> & d, double t) { d = f(t, x, u); };
  std::vector<double> tvec, xvec, vvec, uvec;

  std::optional<smooth::feedback::QPSolution<-1, -1, double>> sol;

  // integrate closed-loop system
  for (std::chrono::milliseconds t = 0s; t < 10s; t += 50ms) {
    auto qp = smooth::feedback::asif_to_qp<nAsif>(g, udes, f, h, bu, prm);
    sol     = smooth::feedback::solve_qp(qp, qpprm, sol);

    u = sol.value().primal.head<1>();

    if (sol.value().code != smooth::feedback::QPSolutionStatus::Optimal) {
      std::cerr << "Solver failed with code " << static_cast<int>(sol.value().code) << std::endl;
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
  using namespace matplot;

  figure();
  hold(on);

  plot(tvec, xvec)->line_width(2);
  plot(tvec, vvec)->line_width(2);
  plot(tvec, transform(tvec, [&](auto t) { return 3; }), "--")->line_width(2);
  plot(tvec, transform(tvec, [&](auto t) { return 1.5; }), "--")->line_width(2);
  legend({"x", "v", "x_{max}", "v_{max}"});

  figure();
  hold(on);
  title("Input");
  plot(tvec, uvec)->line_width(2);
  plot(tvec, transform(tvec, [](auto t) { return 1; }), "--")->line_width(2);

  show();
#else
  std::cout << "TRAJECTORY:" << std::endl;
  for (auto i = 0u; i != tvec.size(); ++i) {
    std::cout << "t=" << tvec[i] << ": x=" << xvec[i] << ", v=" << vvec[i] << std::endl;
  }
#endif
}
