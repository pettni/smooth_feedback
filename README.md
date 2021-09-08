# smooth_feedback: Control and estimation on Lie groups

[![CI Build and Test][ci-shield]][ci-link]
[![Code coverage][cov-shield]][cov-link]
[![License][license-shield]][license-link]

<img src="media/se2_example.gif" width="100%">

Tool collection for control and estimation on Lie groups leveraging the [smooth][smooth-link] library.

* Requirements: C++20, Eigen 3.4, boost::numeric::odeint, LAPACK, [smooth][smooth-link]
* [Documentation][doc-link]


## Control on Lie groups

These controllers are implemented for systems with dynamics on the form
![](https://latex.codecogs.com/png.image?\dpi{110}&space;\mathrm{d}^r&space;\mathbf{x}_t&space;=&space;f(t,&space;\mathbf{x},&space;\mathbf{u}),&space;\quad&space;\mathbf{x}&space;\in&space;\mathbb{X},&space;\mathbf{u}&space;\in&space;\mathbb{U}) where `X` is a `smooth::LieGroup` and `U` is a `smooth::Manifold`. 

Nonlinearities are handled via linearization around a reference point or trajectory.
For group-linear dynamics this automatically results in a linear system in the tangent space, 
in which case these algorithms are expected to work very well. Linearization is done via
[automatic differentiation][ad-link]. For this to work with the most performant methods (e.g. [autodiff][autodiff-link]) the functions must be templated on the scalar type. 
The dynamical system

![](https://latex.codecogs.com/png.latex?%5CSigma%20%3A%20%5Cleft%5C%7B%20%5Cbegin%7Baligned%7D%20%5Cmathrm%7Bd%7D%5Er%20%5Cmathbf%7Bx%7D_t%20%26%3D%20v%28t%29%2C%20%5Cquad%20%5Cmathbf%7Bx%7D%28t%29%20%5Cin%20%5Cmathbb%7BSE%7D%282%29%29%2C%20v%20%5Cin%20%5Cmathbb%7BR%7D%5E3%20%5C%5C%20%5Cfrac%7B%5Cmathrm%7Bd%7D%7D%7B%5Cmathrm%7Bd%7Dt%7D%20v%28t%29%20%26%3D%20A%20v%20&plus;%20B%20u%2C%20%5Cquad%20u%20%5Cin%20%5Cmathbb%7BR%7D%5E2%20%5Cend%7Baligned%7D%20%5Cright.)

can be defined via a lambda function that supports automatic differentiation as follows:
```cpp
#include <Eigen/Core>

#include <smooth/bundle.hpp>
#include <smooth/se2.hpp>

#include <chrono>

// Time type
using Time = std::chrono::duration<double>;
// State space (must be a smooth::LieGroup)
template<typename T>
using X = smooth::Bundle<smooth::SE2<T>, Eigen::Matrix<T, 3, 1>>;
// Input space (must be a smooth::Manifold)
template<typename T>
using U = Eigen::Matrix<T, 2, 1>;

const Eigen::Matrix3d A{
  {-0.2, 0, 0},
  {0, 0, 0},
  {0, 0, -0.4},
};
const Eigen::Matrix<double, 3, 2> B{
  {1, 0},
  {0, 0},
  {0, 1},
};

auto Sigma = []<typename T>(Time, const X<T> & x, const U<T> & u) -> smooth::Tangent<X<T>> {
  smooth::Tangent<X<T>> dx_dt;
  dx_dt.head(3) = x.template part<1>();
  dx_dt.tail(3) = A * x.template part<1>() + B * u;
  return dx_dt;
};
```

### PID Control

* Model-free
* Assumes that inputs control body acceleration. See `examples/pid_se2.cpp` for an example of allocating PID inputs to actuators.

**Example** PID controller on SE(2)

```cpp
#include <smooth/feedpack/pid.hpp>

smooth::feedback::PID<Time, smooth::SE2d> pid;

// set desired motion
pid.set_xdes([](Time Time) -> std::tuple<smooth::SE2d, Eigen::Vector3d, Eigen::Vector3d> {
  return {
    smooth::SE2d::Identity(),  // position
    Eigen::Vector3d::Zero(),   // velocity (right derivative of position w.r.t. t)
    Eigen::Vector3d::Zero(),   // acceleration (second right derivative of position w.r.t. t)
  };
});

Time t            = Time(1);                    // current time
smooth::SE2d x    = smooth::SE2d::Random();     // current state
Eigen::Vector3d v = Eigen::Vector3d::Random();  // current body velocity

Eigen::Vector3d u = pid(t, x, v);
```

### Model-Predictive Control

* Automatic linearization and time discretization of nonlinear continuous dynamics
* Define state and input reference trajectories via arbitrary functions `T -> X` and `T -> U` for a time type T. The bus in the video above uses MPC to track the boundary of the circle.

**Example**: Model-predictive control for the system `Sigma` (see also `examples/mpc_asif_vehicle.cpp`)

```cpp
#include <smooth/feedback/mpc.hpp>

smooth::feedback::MPC<Time, X<double>, U<double>, decltype(Sigma)> mpc(Sigma, {.T = 5, .K = 50});

// set desired input and state trajectories
mpc.set_udes([]<typename T>(T t) -> U<T> { return U<T>::Zero(); });
mpc.set_xdes([]<typename T>(T t) -> X<T> { return X<T>::Identity(); });

Time t(0);
X<double> x = X<double>::Identity();

// get control input for time t and state x
auto [u, code] = mpc(t, x);
```

### Active Set Invariance Filtering (ASIF)

* Minimally invasive filtering of a control input in order to enforce state constraints. The bus in the video above is using an ASIF that avoids the red cylinder.
* Automatic differentiation of nonlinear continuous dynamics and constraints
* Theory (non-Lie group case) is described in e.g. [Thomas Gurriet's Ph.D. thesis](https://thesis.library.caltech.edu/13771/1/My_Thesis.pdf)

**Example**: Safety filtering for the system `Sigma`

```cpp
#include <smooth/feedback/asif.hpp>

smooth::feedback::ASIFilter<Time, X<double>, U<double>, decltype(Sigma)> asif(Sigma);

// safety set S(t) = { x : h(t, x) >= 0 }
auto h = []<typename T>(T, const X<T> & x) -> Eigen::Matrix<T, 1, 1> {
  return Eigen::Matrix<T, 1, 1>(x.template part<0>().r2().x() - T(0.2));
};

// backup controller
auto bu = []<typename T>(T, const X<T> &) -> U<T> { return U<T>(1, 1); };

Time t          = Time(1);
X<double> x     = X<double>::Random();
U<double> u_des = U<double>::Zero();

// get control input for time t, state x, and reference input u_des
auto [u_asif, code] = asif(t, x, u_des, h, bu);
```


## Estimation on Lie groups

Estimators take system models on the form
![](https://latex.codecogs.com/png.image?\dpi{110}&space;\mathrm{d}^r&space;\mathbf{x}_t&space;=&space;f(t,&space;\mathbf{x}),&space;\quad&space;\mathbf{x}&space;\in&space;\mathbb{X}) where `X` is a `smooth::LieGroup`, and measurements on the form ![](https://latex.codecogs.com/png.image?\dpi{110}&space;\mathbf{y}&space;=&space;h(\mathbf{x})&space;\oplus_r&space;w,&space;\;&space;w&space;\in&space;\mathcal&space;N(0,&space;R)).


To use in a feedback loop for a controlled system use partial application:
```cpp
// variable that holds current input
U<double> u = U<double>::Random();
// closed-loop dynamics
auto SigmaCL = [&u]<typename T>(T t, const X<T> & x) { return Sigma(Time(t), x, u.template cast<T>()); };
```

### Extended Kalman Filter

* Templated over dynamics and measurement models
* Automatic differentiation
* Reduces to [standard Kalman filter (KF)](https://en.wikipedia.org/wiki/Kalman_filter) for linear models on Rn
* Reduces to [Invariant Extended Kalman Filter (IEKF)](https://en.wikipedia.org/wiki/Invariant_extended_Kalman_filter) for group-linear models on Lie groups 

**Example: localization with a known 2D landmark** for the system `SigmaCL`

```cpp
#include <smooth/feedback/ekf.hpp>

// create filter
smooth::feedback::EKF<X<double>> ekf;

// measurement model
Eigen::Vector2d landmark(1, 1);
auto h = [&landmark]<typename T>(const X<T> & x) -> Eigen::Matrix<T, 2, 1> {
  return x.template part<0>().inverse() * landmark;
};

// PREDICT STEP: propagate filter over time
ekf.predict(SigmaCL,
  Eigen::Matrix<double, 6, 6>::Identity(),  // motion covariance Q
  1.                                        // time step length
);

// UPDATE STEP: register a measurement of the known landmark
ekf.update(h,
  Eigen::Vector2d(0.3, 0.6),   // measurement result y
  Eigen::Matrix2d::Identity()  // measurement covariance R
);

// access estimate
auto x_hat = ekf.estimate();
auto P_hat = ekf.covariance();
```


## Optimization

MPC and ASIF relies on online quadratic program optimization.

### Fast QP Solver

* Eigen port of the [operator splitting QP solver](https://osqp.org/). 
* For both dense and sparse problems.
* Eigen lazy evaluations enable fast SIMD in the compiled assembly.

Preliminary results suggest that the performance compares favorably to the original OSQP implementation.
The plot below compares solution times (lower is better) for square QPs over three different levels of sparsity. Although the `smooth` sparse (smooth-s) solver is consistently faster than the `smooth` dense (smooth-d) and osqp solvers, it currently appears to be slightly less robust than the other methods on ill-conditioned problems. 

<img src="media/qp_benchmarks.png" width="100%">

The results are generated from the benchmarking program in `benchmark/`. 

**Example**: Define and solve a dynamically sized dense quadratic program.

```cpp
#include <smooth/feedback/qp.hpp>

// define the QP
//  min 0.5 x' * P * x + q' * x
//  s.t l <= A * x <= u
smooth::feedback::QuadraticProgram<-1, -1> qp{
  .P = P,  // n x n matrix
  .q = q,  // n x 1 vector
  .A = A,  // m x n matrix
  .l = l,  // m x 1 vector
  .u = u,  // m x 1 vector
};

smooth::feedback::QPSolverParams prm{};
auto sol = smooth::feedback::solve_qp(qp, prm);
```

<!-- MARKDOWN LINKS AND IMAGES -->
[doc-link]: https://pettni.github.io/smooth_feedback

[ci-shield]: https://img.shields.io/github/workflow/status/pettni/smooth_feedback/build_and_test/master?style=flat-square
[ci-link]: https://github.com/pettni/smooth_feedback/actions/workflows/build_and_test.yml

[cov-shield]: https://img.shields.io/codecov/c/gh/pettni/smooth_feedback/master?style=flat-square
[cov-link]: https://codecov.io/gh/pettni/smooth_feedback

[license-shield]: https://img.shields.io/github/license/pettni/smooth_feedback.svg?style=flat-square
[license-link]: https://github.com/pettni/smooth_feedback/blob/master/LICENSE

[autodiff-link]: https://github.com/autodiff/autodiff/
[smooth-link]: https://github.com/pettni/smooth/
[ad-link]: https://en.wikipedia.org/wiki/Automatic_differentiation
