# smooth_feedback: Control and estimation on Lie groups

[![CI Build and Test][ci-shield]][ci-link]
[![Code coverage][cov-shield]][cov-link]
[![License][license-shield]][license-link]

Tool collection for control and estimation on Lie groups leveraging the
[smooth][smooth-link] library.


## Control on Lie groups

These controllers are implemented for systems with dynamics on the form
![](https://latex.codecogs.com/png.latex?\mathrm{d}^r&space;f_\mathbf{x}&space;=&space;f(\mathbf{x},&space;\mathbf{u}),&space;\quad&space;\mathbf{x}&space;\in&space;\mathbb{X},&space;\mathbf{u}&space;\in&space;\mathbb{U}.) 

Nonlinearities are handled via linearization around a reference point or trajectory. For group-linear dynamics
this automatically results in a linear system in the tangent space, in which case these algorithms are expected
to work very well.

### Proportional-Derivative Control: A classic, now on Lie groups

*STATUS*: not implemented

### Model-Predictive Control: When more look-ahead is needed

*STATUS*: not tested

* Linearization and time discretization of nonlinear dynamics
* Templated on problem size

**Example**:

```cpp
#include <smooth/feedback/mpc.hpp>
```

#### TODOs

- Possibility to bound deviation from linearization trajectory

### Active Set Invariance: Don't collide with stuff

*STATUS*: not tested

**Example**:

```cpp
#include <smooth/feedback/asif.hpp>
```


## Filtering on Lie groups

Filters take system models on the form
![](https://latex.codecogs.com/png.latex?\mathrm{d}^r&space;f_\mathbf{x}&space;=&space;f(\mathbf{x}),&space;\quad&space;\mathbf{x}&space;\in&space;\mathbb{X},&space;\mathbf{u}&space;\in&space;\mathbb{U},) 
to use it in a feedback loop for a controlled system use partial application:
```cpp
// controlled system dynamics dr x_t = f(x, u)
const auto f = [] (const auto & x, const auto & g) -> Tangent { ... };

// variable that holds current input
U u;

// closed-loop dynamics dr x_t = f(x) to use in the filters
const auto f_cl = [&f, &u] (const auto & x) -> Tangent { return f(x, u); };
```

### smooth::feedback::EKF: The only Kalman filter you need in 63 lines of code

*STATUS*: not tested

* Templated over dynamics and measurement models
* Automatic differentiation
* Reduces to [standard Kalman filter (KF)](https://en.wikipedia.org/wiki/Kalman_filter) for linear models on Rn
* Reduces to [Invariant Extended Kalman Filter (IEKF)](https://en.wikipedia.org/wiki/Invariant_extended_Kalman_filter) for group-linear models on Lie groups 

**Example: localization with a known 2D landmark**

```cpp
#include <smooth/feedback/ekf.hpp>

smooth::feedback::EKF<smooth::SE2d> ekf;

// motion model
const auto f = [](double, const auto &) { return smooth::SE2d::Tangent(0.4, 0.01, 0.1); };

// measurement model
Eigen::Vector2d landmark(1, 1);
const auto h = [&landmark](const auto & x) { return x.inverse() * landmark; };

// PREDICT STEP: propagate filter over time
ekf.predict(
  f,
  Eigen::Matrix3d::Identity(),  // motion covariance
  1.                            // time step length
);

// UPDATE STEP: register a measurement of a landmark at [1, 1]
ekf.update(
  h,
  Eigen::Vector2d(0.3, 0.6),   // measurement result
  Eigen::Matrix2d::Identity()  // measurement covariance
);

// access estimate
std::cout << ekf.estimate() << std::endl;
```


## Optimization

To make it easy to solve the quadratic programs generated by the MPC and ASIF functions
a solver is included.

### smooth::feedback::solve_qp: Fast QP Solver in pure Eigen

*STATUS*: moderately tested

* Eigen version of the [operator splitting algorithm](https://osqp.org/)
* For both dense and sparse problems
* Eigen lazy evaluations enable fast SIMD code

**Example**:

```cpp
#include <smooth/feedback/qp.hpp>
```

#### TODOs

- AMD reordering before sparse factorization
- Pivoting in sparse factorization


<!-- MARKDOWN LINKS AND IMAGES -->
[doc-link]: https://pettni.github.io/smooth_feedback

[ci-shield]: https://img.shields.io/github/workflow/status/pettni/smooth_feedback/build_and_test/master?style=flat-square
[ci-link]: https://github.com/pettni/lie/actions/workflows/build_and_test.yml

[cov-shield]: https://img.shields.io/codecov/c/gh/pettni/smooth_feedback/master?style=flat-square
[cov-link]: https://codecov.io/gh/pettni/smooth_feedback

[license-shield]: https://img.shields.io/github/license/pettni/smooth_feedback.svg?style=flat-square
[license-link]: https://github.com/pettni/smooth_feedback/blob/master/LICENSE

[smooth-link]: https://github.com/pettni/smooth/

