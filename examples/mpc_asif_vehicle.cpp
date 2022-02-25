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

#include <boost/numeric/odeint.hpp>
#include <smooth/bundle.hpp>
#include <smooth/compat/odeint.hpp>
#include <smooth/feedback/asif.hpp>
#include <smooth/feedback/mpc.hpp>
#include <smooth/se2.hpp>
#include <smooth/se3.hpp>

#include <chrono>

#ifdef ENABLE_PLOTTING
#include <matplot/matplot.h>
#endif

#ifdef ENABLE_ROS
#include <gazebo_msgs/srv/set_entity_state.hpp>
#include <geometry_msgs/msg/accel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <smooth/compat/ros.hpp>
#endif

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;

using Time = std::chrono::duration<double>;

template<typename S>
using X = smooth::Bundle<smooth::SE2<S>, Eigen::Matrix<S, 3, 1>>;
template<typename S>
using U = Eigen::Matrix<S, 2, 1>;

using Xd = X<double>;
using Ud = U<double>;

using Tangentd = smooth::Tangent<Xd>;

int main()
{
  // dynamics
  auto f = []<typename S>(const X<S> & x, const U<S> & u) -> smooth::Tangent<X<S>> {
    return {
      x.template part<1>().x(),
      x.template part<1>().y(),
      x.template part<1>().z(),
      -S(0.2) * x.template part<1>().x() + u.x(),
      S(0),
      -S(0.4) * x.template part<1>().z() + u.y(),
    };
  };

  auto cr = []<typename S>(const X<S> &, const U<S> & u) -> Eigen::Vector<S, 2> { return u; };
  Eigen::Vector2d crl{-0.2, -0.5};
  Eigen::Vector2d cru{0.2, 0.5};

  // simulation time step
  const auto dt = 25ms;

  ////////////////////
  //// SET UP MPC ////
  ////////////////////

  smooth::feedback::MPC<30, Time, Xd, Ud, decltype(f), decltype(cr)> mpc{
    f,
    cr,
    crl,
    cru,
    {.tf = 5},
  };

  // define desired trajectory
  auto xdes = []<typename S>(S t) -> X<S> {
    const Eigen::Vector3d vdes{1, 0, 0.4};
    return X<S>{
      smooth::SE2<S>(smooth::SO2<S>(M_PI_2), Eigen::Vector2<S>(2.5, 0)) + (t * vdes),
      vdes,
    };
  };

  // set desired trajectory in MPC
  mpc.set_xdes_rel(xdes);
  mpc.set_udes_rel([]<typename S>(S) -> U<S> { return U<S>::Zero(); });

  /////////////////////
  //// SET UP ASIF ////
  /////////////////////

  // safe set
  auto h = []<typename S>(S, const X<S> & x) -> Eigen::Matrix<S, 1, 1> {
    const Eigen::Vector2<S> dir = x.template part<0>().r2() - Eigen::Vector2<S>{0, -2.3};
    const Eigen::Vector2d e_dir = dir.template cast<double>().normalized();
    return Eigen::Matrix<S, 1, 1>(dir.dot(e_dir) - 0.7);
  };

  // backup controller
  auto bu = []<typename S>(S, const X<S> & x) -> U<S> {
    return {0.2 * x.template part<1>().x(), -0.5};
  };

  const smooth::feedback::ManifoldBounds<Ud> ulim{
    .A = Eigen::Matrix2d{{1, 0}, {0, 1}},
    .c = Ud::Zero(),
    .l = Eigen::Vector2d(-0.2, -0.5),
    .u = Eigen::Vector2d(0.5, 0.5),
  };

  // parameters
  smooth::feedback::ASIFilterParams<Ud> asif_prm{
    .T        = 2.5,
    .nh       = 1,
    .u_weight = Eigen::Vector2d{20, 1},
    .ulim     = ulim,
    .asif =
      {
        .K          = 200,
        .alpha      = 5,
        .dt         = 0.01,
        .relax_cost = 100,
      },
    .qp =
      {
        .polish = false,
      },
  };

  smooth::feedback::ASIFilter<Xd, Ud, decltype(f)> asif(f, asif_prm);

  /////////////////////////
  //// CREATE ROS NODE ////
  /////////////////////////

#ifdef ENABLE_ROS
  rclcpp::init(0, nullptr);
  auto node       = std::make_shared<rclcpp::Node>("se2_example");
  auto ses_client = node->create_client<gazebo_msgs::srv::SetEntityState>("/set_entity_state");
  auto u_mpc_pub =
    node->create_publisher<geometry_msgs::msg::Accel>("u_mpc", rclcpp::SystemDefaultsQoS{});
  auto u_asif_pub =
    node->create_publisher<geometry_msgs::msg::Accel>("u_asif", rclcpp::SystemDefaultsQoS{});

  rclcpp::Rate rate(25ms);
#endif

  /////////////////////////////////////
  //// SIMULATE CLOSED-LOOP SYSTEM ////
  /////////////////////////////////////

  // system variables
  Xd x = Xd::Identity();
  Ud u;

  // prepare for integrating the closed-loop system
  runge_kutta4<Xd, double, Tangentd, double, vector_space_algebra> stepper{};
  const auto ode = [&f, &u](const Xd & x, Tangentd & d, double) { d = f(x, u); };
  std::vector<double> tvec, xvec, yvec, u1vec, u2vec, u1mpcvec, u2mpcvec;

  // integrate closed-loop system
  for (std::chrono::milliseconds t = 0s; t < 30s; t += dt) {
    // compute MPC input
    const auto [u_mpc, mpc_code] = mpc(t, x);
    if (mpc_code != smooth::feedback::QPSolutionStatus::Optimal) {
      std::cerr << "MPC failed with mpc_code " << static_cast<int>(mpc_code) << std::endl;
    }

    // filter input with ASIF
    const auto [u_asif, asif_code] = asif(x, u_mpc, h, bu);
    if (asif_code != smooth::feedback::QPSolutionStatus::Optimal) {
      std::cerr << "ASIF solver failed with asif_code " << static_cast<int>(asif_code) << std::endl;
    }

    // select input
    u = u_asif;

    // store data
    tvec.push_back(duration_cast<Time>(t).count());
    xvec.push_back(x.template part<0>().r2().x());
    yvec.push_back(x.template part<0>().r2().y());

    u1mpcvec.push_back(u_mpc(0));
    u2mpcvec.push_back(u_mpc(1));
    u1vec.push_back(u_asif(0));
    u2vec.push_back(u_asif(1));

#ifdef ENABLE_ROS
    auto req        = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
    req->state.name = "bus::link";
    smooth::Map<geometry_msgs::msg::Pose>(req->state.pose) = x.template part<0>().lift_se3();
    req->state.pose.position.x                             = 8 * req->state.pose.position.x;
    req->state.pose.position.y                             = 8 * req->state.pose.position.y;
    ses_client->async_send_request(req);

    geometry_msgs::msg::Accel u_mpc_msg;
    u_mpc_msg.linear.x  = u_mpc.x();
    u_mpc_msg.angular.z = u_mpc.y();
    u_mpc_pub->publish(u_mpc_msg);

    geometry_msgs::msg::Accel u_asif_msg;
    u_asif_msg.linear.x  = u_asif.x();
    u_asif_msg.angular.z = u_asif.y();
    u_asif_pub->publish(u_asif_msg);

    rate.sleep();
#endif
    // step dynamics
    stepper.do_step(
      ode, x, 0, std::chrono::duration_cast<std::chrono::duration<double>>(dt).count());
  }

#ifdef ENABLE_PLOTTING
  matplot::figure();
  matplot::hold(matplot::on);
  matplot::title("Path");

  matplot::plot(xvec, yvec)->line_width(2);
  matplot::plot(
    matplot::transform(tvec, [&](auto t) { return xdes(t).template part<0>().r2().x(); }),
    matplot::transform(tvec, [&](auto t) { return xdes(t).template part<0>().r2().y(); }),
    "k--")
    ->line_width(2);
  matplot::legend({"actual", "desired"});

  matplot::figure();
  matplot::hold(matplot::on);
  matplot::title("Inputs");
  matplot::plot(tvec, u1vec, "r")->line_width(2);
  matplot::plot(tvec, u2vec, "b")->line_width(2);
  matplot::plot(tvec, u1mpcvec, "--r")->line_width(2);
  matplot::plot(tvec, u2mpcvec, "--b")->line_width(2);
  matplot::legend({"u1", "u2", "u1des", "u2des"});

  matplot::show();
#else
  std::cout << "TRAJECTORY:" << std::endl;
  for (auto i = 0u; i != tvec.size(); ++i) {
    std::cout << "t=" << tvec[i] << ": x=" << xvec[i] << ", y=" << yvec[i] << std::endl;
  }
#endif

#ifdef ENABLE_ROS
  rclcpp::shutdown();
#endif

  return EXIT_SUCCESS;
}
