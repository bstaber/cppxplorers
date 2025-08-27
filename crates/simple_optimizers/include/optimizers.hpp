#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cppx::opt {

class Optimizer {
  public:
    virtual ~Optimizer() = default;

    virtual void step(std::vector<double> &weights, const std::vector<double> &grads) = 0;
};

// --------------------------- GradientDescent ---------------------------
class GradientDescent final : public Optimizer {
  public:
    explicit GradientDescent(double learning_rate) : lr_(learning_rate) {}

    [[nodiscard]] double learning_rate() const noexcept { return lr_; }

    void step(std::vector<double> &weights, const std::vector<double> &grads) override {
        if (weights.size() != grads.size()) {
            throw std::invalid_argument("weights and grads size mismatch");
        }
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= lr_ * grads[i];
        }
    }

  private:
    double lr_{};
};

// ------------------------------- Momentum ------------------------------
class Momentum final : public Optimizer {
  public:
    // struct Params {
    //     double learning_rate;
    //     double momentum;
    //     std::size_t dim;
    // };

    explicit Momentum(double learning_rate, double momentum, std::size_t dim)
        : lr_(learning_rate), mu_(momentum), v_(dim, 0.0) {}

    [[nodiscard]] double learning_rate() const noexcept { return lr_; }
    [[nodiscard]] double momentum() const noexcept { return mu_; }
    [[nodiscard]] const std::vector<double> &velocity() const noexcept { return v_; }

    void step(std::vector<double> &weights, const std::vector<double> &grads) override {
        if (weights.size() != grads.size()) {
            throw std::invalid_argument("weights and grads size mismatch");
        }
        if (v_.size() != weights.size()) {
            throw std::invalid_argument("velocity size mismatch");
        }

        for (std::size_t i = 0; i < weights.size(); ++i) {
            v_[i] = mu_ * v_[i] + lr_ * grads[i]; // v ← μ v + η g
            weights[i] -= v_[i];                  // w ← w − v
        }
    }

  private:
    double lr_{};
    double mu_{};
    std::vector<double> v_;
};

} // namespace cppx::opt
