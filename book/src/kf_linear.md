## Model formulation

We consider the following general state space model made of a state equation and an observation equation:

$$
x_k = \Theta x_{k-1} + W_k, \quad W_k \sim \mathcal{N}(0, R)
$$
$$
y_k = A_k x_k + V_k, \quad V_k \sim \mathcal{N}(0, S)
$$

where:

- $x_k \in \mathbb{R}^n$ is the state vector at time step $k$,
- $y_k \in \mathbb{R}^p$ is the observation/measurement vector.

We also have the following hyperparameters assumed to be known:
- $\Theta \in \mathbb{R}^{n \times n}$ is the state transition matrix,
- $A_k \in \mathbb{R}^{p \times n}$ is the observation matrix,
- $Q \in \mathbb{R}^{n \times n}$ is the process noise covariance,
- $R \in \mathbb{R}^{p \times p}$ is the observation noise covariance.

The aim is to estimate the state $x_k$ given the noisy observations $y_k, y_{k-1}, \dots, y_1$.

## Kalman filtering

We start with an initial state $x_0 \sim \mathcal{N}(x_0^0, P_0^0)$ and we want to determine the conditional distribution $p(x_1 | y_1)$ of the next state. Using Baye's rule, one has
$$
p(x_1 | y_1) \propto p(y_1 | x_1) p(x_1)\,.
$$
The likelihood $p(y_1 | x_1)$ can easily be determined thanks to the chosen observation equation. The marginal distribution $p(x_1)$ can be obtained by marginalizing $p(x_1, x_0)$.
This marginal distribution can be seen as our prior knowledge before seeing $y_1$. It can easily be shown that
$$
p(x_1) \propto \mathcal{N}(x_1^0, P_1^0)\,,
$$
with
$$
x_1^0 = \Theta x_0^0\,, \quad P_1^0 = \Theta P_0^0 \Theta^T + R\,.
$$
This first estimate is what we will call the **prediction step**. Using this result, we can determine $p(x_1 | y_1)$:
$$
p(x_1 | y_1) \propto \mathcal{N}(x_1^1, P_1^1)\,,
$$
with
$$
x_1^1 = x_1^0 + K_1(y_1 - A_1 x_1^0)\,, \quad P_1^1 = (I - K_1 A_1)P_1^0\,,
$$
which gives us our estimate the next state given the observation $y_1$. This is what we call the **update step**.

## General equations

For an arbitrary time step $k$, the **prediction step** yields:
$$
x_t^{t-1} = \Theta x_{t-1}^{t-1}\,, \quad P_t^{t-1} = \Theta P_{t-1}^{t-1} \Theta^T + R\,,
$$
and the **update step** is given by
$$
x_t^t = x_t^{t-1} + K_t(y_t - A_t x_t^{t-1})\,, \quad P_t^t = (I - K_t A_t)P_t^{t-1}\,,
$$
where the Kalman gain $K_t$ is defined as
$$
K_t = P_t^{t-1} A_t^T(A_t P_t^{t-1}A_t^T + S)^{-1}\,.
$$

Implementing the Kalman filter boils down to implement these few equations!

## C++ implementation

The following code provides a generic templated class `KFLinear` supporting both fixed-size and dynamic-size state and measurement vectors, using the [Eigen](https://eigen.tuxfamily.org/) linear algebra library.

<details>
<summary>Click here to view the full implementation: <b>include/kf_linear.hpp</b>. We break into down in the sequel of this section. </summary>

```cpp
{{#include ../../crates/kf_linear/include/kf_linear.hpp}}
```
</details>


Here's the header file without the inlined implementations.

```cpp
#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

/**
 * @brief Generic linear Kalman filter (templated, no control term).
 *
 * State-space model:
 *   x_k = A x_{k-1} + w_{k-1},   w ~ N(0, Q)
 *   z_k = H x_k     + v_k,       v ~ N(0, R)
 *
 * Template parameters:
 *   Nx = state dimension      (int or Eigen::Dynamic)
 *   Ny = measurement dimension(int or Eigen::Dynamic)
 */
template <int Nx, int Ny> class KFLinear {
  public:
    using StateVec = Eigen::Matrix<double, Nx, 1>;
    using StateMat = Eigen::Matrix<double, Nx, Nx>;
    using MeasVec  = Eigen::Matrix<double, Ny, 1>;
    using MeasMat  = Eigen::Matrix<double, Ny, Ny>;
    using ObsMat   = Eigen::Matrix<double, Ny, Nx>;

    KFLinear(const StateVec &initial_state, const StateMat &initial_covariance,
             const StateMat &transition_matrix, const ObsMat &observation_matrix,
             const StateMat &process_covariance, const MeasMat &measurement_covariance);

    void predict();
    void update(const MeasVec &z);
    void step(const std::optional<MeasVec> &measurement);
    std::vector<StateVec> filter(const std::vector<std::optional<MeasVec>> &measurements);

    [[nodiscard]] const StateVec &state() const { return x_; }
    [[nodiscard]] const StateMat &covariance() const { return P_; }

    void set_transition(const StateMat &A)      { A_ = A; }
    void set_observation(const ObsMat &H)       { H_ = H; }
    void set_process_noise(const StateMat &Q)   { Q_ = Q; }
    void set_measurement_noise(const MeasMat &R){ R_ = R; }

  private:
    StateMat A_, Q_, P_;
    ObsMat   H_;
    MeasMat  R_;
    StateVec x_;
};
```

A few comments:

- **Predict step**: The method `predict()` propagates the state and covariance using the transition matrix A and process noise covariance Q.

- **Update step**: The method `update(z)` corrects the prediction using a new measurement z. It computes the Kalman gain K efficiently by solving a linear system with LDLT factorization instead of forming the matrix inverse explicitly. The covariance update uses the Joseph form to ensure numerical stability and positive semi-definiteness.

- **Step and filter**: The `step()` method combines prediction with an optional update (useful when some measurements are missing). The `filter()` method processes an entire sequence of measurements, returning the sequence of estimated states.

- **Flexibility**:  
  - Works with both fixed-size and dynamic-size Eigen matrices.  
  - Provides setters to update system matrices online (e.g. if the model changes over time).  
  - Uses `std::optional` to naturally handle missing observations.
