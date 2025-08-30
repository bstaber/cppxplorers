#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <numeric>

namespace cppx::ar_models {

template <int order> class ARModel {
  public:
    using Vector = Eigen::Vector<double, order>;

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

template <int order> ARModel<order> fit_ar_ols(const std::vector<double> &x) {
    if (static_cast<int>(x.size()) <= order) {
        throw std::invalid_argument("Time series too short for AR(order)");
    }

    const int T = static_cast<int>(x.size());
    const int n = T - order;

    // Build the design system
    Eigen::MatrixXd X(n, order + 1);
    Eigen::VectorXd Y(n);

    for (int t = 0; t < n; ++t) {
        Y(t) = x[order + t];
        X(t, 0) = 1.0;
        for (int j = 0; j < order; ++j) {
            X(t, j + 1) = x[order + t - 1 - j];
        }
    }

    // Solve least squares
    Eigen::VectorXd beta = X.colPivHouseholderQr().solve(Y);
    // beta(0) = intercept, beta(1..order) = AR coefficients

    // Compute residual variance
    Eigen::VectorXd resid = Y - X * beta;
    double sigma2 = resid.squaredNorm() / static_cast<double>(n - (order + 1));

    // Create AR(p)
    typename ARModel<order>::Vector phi;
    for (int j = 0; j < order; ++j) {
        phi(j) = beta(j + 1);
    }

    ARModel<order> model;
    model.set_coefficients(phi);
    model.set_intercept(beta(0));
    model.set_noise(sigma2);

    return model;
}

inline double _sample_mean(const std::vector<double> &x) {
    double mu = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    return mu;
}
inline double _sample_autocov(const std::vector<double> &x, int k) {
    const int T = static_cast<int>(x.size());
    if (k >= T) {
        throw std::invalid_argument("lag too large");
    }
    const double mu = _sample_mean(x);
    double acc = 0.0;
    for (int t = k; t < T; ++t) {
        acc += (x[t] - mu) * (x[t - k] - mu);
    }
    return acc / static_cast<double>(T);
}

template <int order> ARModel<order> fir_ar_yule_walkter(const std::vector<double> &x) {
    static_assert(order >= 1, "Yule–Walker needs order >= 1");
    if (static_cast<int>(x.size()) <= order) {
        throw std::invalid_argument("Time series too short for AR(order)");
    }

    // r[0..order] sample autocovariances
    std::array<double, order + 1> r{};
    for (int k = 0; k <= order; ++k) {
        r[k] = _sample_autocov(x, k);
    }

    // Levinson–Durbin recursion
    typename ARModel<order>::Vector a;
    a.setZero();
    double E = r[0];
    if (std::abs(E) < 1e-15) {
        throw std::runtime_error("Zero variance");
    }

    for (int m = 1; m <= order; ++m) {
        double acc = r[m];
        for (int j = 1; j < m; ++j)
            acc -= a(j - 1) * r[m - j];
        const double kappa = acc / E;

        // update a (reflection update)
        typename ARModel<order>::Vector a_new = a;
        a_new(m - 1) = kappa;
        for (int j = 1; j < m; ++j) {
            a_new(j - 1) = a(j - 1) - kappa * a(m - j - 1);
        }
        a = a_new;

        E *= (1.0 - kappa * kappa);
        if (E <= 0) {
            throw std::runtime_error("Non-positive innovation variance in recursion");
        }
    }

    // Compute intercept so that unconditional mean matches sample mean (stationarity assumption)
    const double xbar = _sample_mean(x);
    const double one_minus_sum = 1.0 - a.sum();
    const double c = one_minus_sum * xbar;

    ARModel<order> model;
    model.set_coefficients(a);
    model.set_intercept(c);
    model.set_noise(E);

    return model;
}

} // namespace cppx::ar_models