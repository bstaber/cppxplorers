
# Autoregressive models

This chapter documents a header-only implementation of an Autoregressive model AR(*p*) in modern C++.  
It explains the class template, the OLS and Yule–Walker estimators, and small-but-important C++ details you asked about.

## AR(p) refresher

An AR(*p*) process is
$$
X_t = c + \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \varepsilon_t,
$$
with intercept $c$, coefficients $\phi_i$, and i.i.d. noise $\varepsilon_t \sim (0,\sigma^2)$.

## Header overview

```cpp
#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <numeric>

namespace cppx::ar_models {

template <int order> class ARModel {
  public:
    using Vector = Eigen::Matrix<double, order, 1>;

    ARModel() = default;
    ARModel(double intercept, double noise_variance) : c_(intercept), sigma2_(noise_variance){};

    [[nodiscard]] double intercept() const noexcept { return c_; }
    [[nodiscard]] double noise() const noexcept { return sigma2_; }
    [[nodiscard]] const Vector &coefficients() const noexcept { return phi_; }

    void set_coefficients(const Vector &phi) { phi_ = phi; }
    void set_intercept(double c) { c_ = c; }
    void set_noise(double noise) { sigma2_ = noise; }

    double forecast_one_step(const std::vector<double> &hist) const {
        if (static_cast<int>(hist.size()) < order) {
            throw std::invalid_argument("History shorter than model order");
        }
        double y = c_;
        for (int i = 0; i < order; ++i) {
            y += phi_(i) * hist[i];
        }
        return y;
    }

  private:
    Vector phi_;
    double c_ = 0.0;
    double sigma2_ = 1.0;
};
```
Notes
- `using Vector = Eigen::Matrix<double, order, 1>;` is the correct Eigen alias (there is no `Eigen::Vector<double, N>` type).
- Defaulted constructor + in-class member initializers (C++11) keep initialization simple.
- `[[nodiscard]]` marks return values that shouldn’t be ignored.
- `static_cast<int>` is used because `std::vector::size()` returns `size_t` (unsigned), while Eigen commonly uses `int` sizes.

## Forecasting (one-step)

```cpp
double forecast_one_step(const std::vector<double> &hist) const {
    if (static_cast<int>(hist.size()) < order) {
        throw std::invalid_argument("History shorter than model order");
    }
    double y = c_;
    for (int i = 0; i < order; ++i) {
        y += phi_(i) * hist[i];            // hist[0]=X_T, hist[1]=X_{T-1}, ...
    }
    return y;
}
```
The one-step-ahead plug‑in forecast $\hat X_{T+1|T}$ equals $c + \sum_{i=1}^p \phi_i X_{T+1-i}$.

## OLS estimator (header-only)

Mathematically, we fit

$$
X_t = c + \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \varepsilon_t.
$$

Define:

- $Y = (X_{p}, X_{p+1}, \dots, X_T)^\top \in \mathbb{R}^{n}$, where $n = T-p$.
- $X \in \mathbb{R}^{n \times (p+1)}$ the **design matrix**:

$$
X =
\begin{bmatrix}
1 & X_{p-1} & X_{p-2} & \cdots & X_{0} \\\\
1 & X_{p}   & X_{p-1} & \cdots & X_{1} \\\\
\vdots & \vdots & \vdots & & \vdots \\\\
1 & X_{T-1} & X_{T-2} & \cdots & X_{T-p}
\end{bmatrix}.
$$

The regression model is

$$
Y = X \beta + \varepsilon, \quad
\beta =
\begin{bmatrix}
c \\ \phi_1 \\ \vdots \\ \phi_p
\end{bmatrix}.
$$

The **OLS estimator** is

$$
\hat\beta = (X^\top X)^{-1} X^\top Y.
$$

Residual variance estimate:

$$
\hat\sigma^2 = \frac{1}{n-(p+1)} \|Y - X\hat\beta\|_2^2.
$$

In code, we solve this with Eigen’s QR decomposition:

```cpp
Eigen::VectorXd beta = X.colPivHouseholderQr().solve(Y);
```

```cpp
template <int order>
ARModel<order> fit_ar_ols(const std::vector<double> &x) {
    if (static_cast<int>(x.size()) <= order) {
        throw std::invalid_argument("Time series too short for AR(order)");
    }
    const int T = static_cast<int>(x.size());
    const int n = T - order;

    Eigen::MatrixXd X(n, order + 1);
    Eigen::VectorXd Y(n);

    for (int t = 0; t < n; ++t) {
        Y(t) = x[order + t];
        X(t, 0) = 1.0;                          // intercept column
        for (int j = 0; j < order; ++j) {
            X(t, j + 1) = x[order + t - 1 - j]; // lagged regressors (most-recent-first)
        }
    }

    Eigen::VectorXd beta = X.colPivHouseholderQr().solve(Y);
    Eigen::VectorXd resid = Y - X * beta;
    const double sigma2 = resid.squaredNorm() / static_cast<double>(n - (order + 1));

    typename ARModel<order>::Vector phi;
    for (int j = 0; j < order; ++j) phi(j) = beta(j + 1);

    ARModel<order> model;
    model.set_coefficients(phi);
    model.set_intercept(beta(0));   // beta(0) is the intercept
    model.set_noise(sigma2);
    return model;
}
```

## Yule–Walker (Levinson–Durbin)

The AR($p$) autocovariance equations are:

$$
\gamma_k = \sum_{i=1}^p \phi_i \gamma_{k-i}, \quad k = 1, \dots, p,
$$

where $\gamma_k = \text{Cov}(X_t, X_{t-k})$.

This leads to the **Yule–Walker system**:

$$
\begin{bmatrix}
\gamma_0 & \gamma_1 & \cdots & \gamma_{p-1} \\\\
\gamma_1 & \gamma_0 & \cdots & \gamma_{p-2} \\\\
\vdots   & \vdots   & \ddots & \vdots \\\\
\gamma_{p-1} & \gamma_{p-2} & \cdots & \gamma_0
\end{bmatrix}
\begin{bmatrix}
\phi_1 \\\\ \phi_2 \\\\ \vdots \\\\ \phi_p
\end{bmatrix}
=
\begin{bmatrix}
\gamma_1 \\\\ \gamma_2 \\\\ \vdots \\\\ \gamma_p
\end{bmatrix}.
$$

We estimate autocovariances by

$$
\hat\gamma_k = \frac{1}{T} \sum_{t=k}^{T-1} (X_t-\bar X)(X_{t-k}-\bar X).
$$

### Levinson–Durbin recursion

Efficiently solves the Toeplitz system in $O(p^2)$ time.  
At each step:

- Update reflection coefficient $\kappa_m$,
- Update AR coefficients $a_j$,
- Update innovation variance

$$
E_m = E_{m-1}(1 - \kappa_m^2).
$$

The final variance $E_p$ is the residual variance estimate.


```cpp
inline double _sample_mean(const std::vector<double> &x) {
    return std::accumulate(x.begin(), x.end(), 0.0) / x.size();
}
inline double _sample_autocov(const std::vector<double> &x, int k) {
    const int T = static_cast<int>(x.size());
    if (k >= T) throw std::invalid_argument("lag too large");
    const double mu = _sample_mean(x);
    double acc = 0.0;
    for (int t = k; t < T; ++t) acc += (x[t]-mu) * (x[t-k]-mu);
    return acc / static_cast<double>(T);
}
```
- `std::vector` has no `.mean()`; we compute it with `std::accumulate` (from `<numeric>`).  
- For compile-time sizes (since `order` is a template parameter) we can use `std::array<double, order+1>` to hold autocovariances.

Levinson–Durbin recursion:
```cpp
template <int order>
ARModel<order> fit_ar_yule_walkter(const std::vector<double> &x) {
    static_assert(order >= 1, "Yule–Walker needs order >= 1");
    if (static_cast<int>(x.size()) <= order) {
        throw std::invalid_argument("Time series too short for AR(order)");
    }

    std::array<double, order + 1> r{};
    for (int k = 0; k <= order; ++k) r[k] = _sample_autocov(x, k);

    typename ARModel<order>::Vector a; a.setZero();
    double E = r[0];
    if (std::abs(E) < 1e-15) throw std::runtime_error("Zero variance");

    for (int m = 1; m <= order; ++m) {
        double acc = r[m];
        for (int j = 1; j < m; ++j) acc -= a(j - 1) * r[m - j];
        const double kappa = acc / E;

        typename ARModel<order>::Vector a_new = a;
        a_new(m - 1) = kappa;
        for (int j = 1; j < m; ++j) a_new(j - 1) = a(j - 1) - kappa * a(m - j - 1);
        a = a_new;

        E *= (1.0 - kappa * kappa);
        if (E <= 0) throw std::runtime_error("Non-positive innovation variance in recursion");
    }

    const double xbar = _sample_mean(x);
    const double c = (1.0 - a.sum()) * xbar;   // intercept so that mean(model) == sample mean

    ARModel<order> model;
    model.set_coefficients(a);
    model.set_intercept(c);
    model.set_noise(E);
    return model;
}
```

## Small questions I asked myself while implementing this

- The class holds parameters + forecasting but the algorithms live outside. This way, I can add/replace estimators without modifying the class.

- `typename ARModel<order>::Vector` — why the `typename`? Inside templates, dependent names might be types or values. `typename` tells the compiler it’s a type.

- `std::array` vs `std::vector`? `std::array<T,N>` is fixed-size (size known at compile time) and stack-allocated while `std::vector<T>` is dynamic-size (runtime) and heap-allocated.

- Why `static_cast<int>(hist.size())`? `.size()` returns `size_t` (unsigned). Converting explicitly avoids signed/unsigned warnings and matches Eigen’s int-based indices.

## Example of usage

```cpp
#include "ar.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<double> x = {0.1, 0.3, 0.7, 0.8, 1.2, 1.0, 0.9};

    auto m = fit_ar_ols<2>(x);
    std::cout << "c=" << m.intercept() << ", sigma^2=" << m.noise()
              << ", phi=" << m.coefficients().transpose() << "\n";

    std::vector<double> hist = {x.back(), x[x.size()-2]}; // [X_T, X_{T-1}]
    std::cout << "one-step forecast: " << m.forecast_one_step(hist) << "\n";
}
```
