#include "optimizers.hpp"
#include <cassert>
#include <cmath>
#include <vector>

using namespace cppx::opt;

static bool approx_equal(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) < eps;
}

int main() {
    // GradientDescent constructor
    {
        GradientDescent opt(1e-3);
        assert(approx_equal(opt.learning_rate(), 1e-3));
    }

    // GradientDescent step
    {
        GradientDescent opt(0.1);
        std::vector<double> w{1.0, 2.0, 3.0};
        std::vector<double> g{0.5, 0.5, 0.5};
        opt.step(w, g);
        assert(approx_equal(w[0], 0.95));
        assert(approx_equal(w[1], 1.95));
        assert(approx_equal(w[2], 2.95));
    }

    // Momentum constructor
    {
        Momentum opt(0.01, 0.9, 10);
        assert(approx_equal(opt.learning_rate(), 0.01));
        assert(approx_equal(opt.momentum(), 0.9));
        assert(opt.velocity().size() == 10);
    }

    // Momentum step
    {
        Momentum opt(0.1, 0.9, 3);
        std::vector<double> w{1.0, 2.0, 3.0};
        std::vector<double> g{0.5, 0.5, 0.5};

        opt.step(w, g);
        // after 1st step: same as GD(0.1)
        assert(approx_equal(w[0], 0.95));
        assert(approx_equal(w[1], 1.95));
        assert(approx_equal(w[2], 2.95));

        opt.step(w, g);
        // expected: [0.855, 1.855, 2.855]
        assert(approx_equal(w[0], 0.855));
        assert(approx_equal(w[1], 1.855));
        assert(approx_equal(w[2], 2.855));
    }

    return 0;
};
