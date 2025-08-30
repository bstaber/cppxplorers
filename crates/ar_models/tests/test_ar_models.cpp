#include "ar.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace cppx::ar_models;

// Tiny helper for floating-point checks
static bool almost_equal(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) <= tol;
}

int main() {
    // Default construction
    {
        ARModel<1> model;
        assert(almost_equal(model.intercept(), 0.0));
        assert(almost_equal(model.noise(), 1.0));
        assert(model.coefficients().size() == 1);
    }

    // OLS fit recovers AR(1)
    {
        // Simulate AR(1): X_t = c + phi X_{t-1} + eps_t
        constexpr int P = 1;
        const double c = 0.4;
        const double phi = 0.65;
        const double sigma = 0.1; // small noise -> tighter estimates
        const int T = 1000;

        std::mt19937 rng(12345);
        std::normal_distribution<double> N(0.0, sigma);

        std::vector<double> x(T);
        x[0] = 0.0;
        for (int t = 1; t < T; ++t) {
            x[t] = c + phi * x[t - 1] + N(rng);
        }

        auto m = fit_ar_ols<P>(x);
        // Coefficients
        assert(m.coefficients().size() == P);
        double phi_hat = m.coefficients()(0);
        // Intercept (your API calls it "intercept()", but you’re storing the intercept there)
        double c_hat = m.intercept();

        // Loose but meaningful tolerances
        assert(std::fabs(phi_hat - phi) < 0.05);
        assert(std::fabs(c_hat - c) < 0.05);
        assert(m.noise() > 0.0);
    }

    // Yule–Walker (Levinson–Durbin) recovers AR(1)
    {
        constexpr int P = 1;
        const double c = 0.2;
        const double phi = 0.5;
        const double sigma = 0.15;
        const int T = 1200;

        std::mt19937 rng(54321);
        std::normal_distribution<double> N(0.0, sigma);

        std::vector<double> x(T);
        x[0] = 0.0;
        for (int t = 1; t < T; ++t) {
            x[t] = c + phi * x[t - 1] + N(rng);
        }

        // NOTE: your function name currently has a typo "fir_ar_yule_walkter"
        auto m = fir_ar_yule_walkter<P>(x);

        double phi_hat = m.coefficients()(0);
        double c_hat = m.intercept();

        assert(std::fabs(phi_hat - phi) < 0.07);
        assert(std::fabs(c_hat - c) < 0.07);
        assert(m.noise() > 0.0);
    }

    // Forecast one-step sanity (AR(1))
    {
        ARModel<1> m;
        // Set a known model: X_t = c + phi X_{t-1} + eps
        ARModel<1>::Vector phi_vec;
        phi_vec << 0.6;
        m.set_coefficients(phi_vec);
        m.set_intercept(0.3); // intercept (your getter name is intercept())
        m.set_noise(0.2);

        // hist = [X_T, X_{T-1}, ...] ; for AR(1) we need 1 value
        std::vector<double> hist = {2.0};
        double yhat = m.forecast_one_step(hist);
        // Expected: c + phi * X_T
        double expected = 0.3 + 0.6 * 2.0;
        assert(almost_equal(yhat, expected, 1e-12));
    }

    // Error handling: series too short
    {
        std::vector<double> tiny = {1.0}; // length 1 cannot fit AR(2), nor AR(1) when n<=p
        bool threw = false;
        try {
            (void) fit_ar_ols<2>(tiny);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        assert(threw);
    }

    return 0;
}
