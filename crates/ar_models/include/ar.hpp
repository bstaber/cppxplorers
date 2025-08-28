#pragma once

#include <Eigen/Dense>

namespace cppx::ar_models {

template <int order> class ARModel {
  public:
		using Vector = Eigen::Vector<double, order>;

    ARModel() = default;
    ARModel(double mean, double noise_variance) : c_(mean), sigma2_(noise_variance){};

    [[nodiscard]] double mean() const noexcept { return c_; }
    [[nodiscard]] double noise() const noexcept { return sigma2_; }
    [[nodiscard]] const Vector &coefficients() const noexcept { return phi_; }

		double forecast_one_step(const std::vector<double>& hist) const { return 0.0; }

  private:
    Vector phi_;
    double c_ = 0.0;
    double sigma2_ = 1.0;
};

template<int order>
ARModel<order> fit_ar_ols(const std::vector<double>& x);

template<int order>
ARModel<order> fir_ar_yule_walkter(const std::vector<double>& x);

} // namespace cppx::ar_models