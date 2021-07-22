# `smooth_feedback`: Control and estimation on Lie groups

Planned content:

- Algo
  - [/] QP solver

- Control
  - [ ] PD
  - [x] MPC -> QP
  - [/] asif++ -> QP

- Estimation:
  - [x] EKF
  - [ ] UKF?
  - [ ] EKF with horizon (fixed-lag/fixed-interval smoother?)?
  - [ ] Pose-graph optimizer (NLE)

## Control

### smooth::feedback::MPC: Model-Predictive Control on Lie groups

* Linearization and time discretization of nonlinear dynamics
* Templated on problem size

## Filtering

### smooth::feedback::EKF: The only Kalman filter you need in 63 lines of code

* Templated over dynamics and measurement models
* Automatic differentiation
* Reduces to [standard Kalman filter (KF)](https://en.wikipedia.org/wiki/Kalman_filter) for linear models on Rn
* Reduces to [Invariant Extended Kalman Filter (IEKF)](https://en.wikipedia.org/wiki/Invariant_extended_Kalman_filter) for group-linear models on Lie groups 

**Example: localization with a known 2D landmark**

```cpp
smooth::feedback::EKF<smooth::SE2d> ekf;

// PREDICT STEP: propagate filter over time
ekf.predict(
  [](double, const auto &) { return smooth::SE2d::Tangent(0.4, 0.01, 0.1); },  // motion model
  Eigen::Matrix3d::Identity(),  // motion covariance
  1.                            // time step length
);

// UPDATE STEP: register a measurement of a landmark at [1, 1]
Eigen::Vector2d landmark(1, 1);
ekf.update([&landmark](const auto & x) { return x.inverse() * landmark; },  // measurement model
  Eigen::Vector2d(0.3, 0.6),                                                // measurement result
  Eigen::Matrix2d::Identity()  // measurement covariance
);

// access estimate
std::cout << ekf.estimate() << std::endl;
```
## Optimization

### smooth::feedback::solveQP: Eigen implementation of the operator splitting approach 

* [x] Dense *and* sparse problems
* [ ] Written in Eigen for vectorization [should verify assembly]
* [ ] Faster than OSQP [need at least two benchmarks]
* [x] Templated on Scalar type

## TODOs

- [ ] Sparse QP solver
- [ ] QP conditioning
- [ ] QP benchmarking
- [ ] Sparse MPC
- [ ] Bound distance from linearization in MPC
