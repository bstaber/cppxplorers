#include "ar.hpp"
#include <cassert>

using namespace cppx::ar_models;

int main() {

    {
        ARModel<1> model;
        assert(model.mean() == 0.0);
        assert(model.noise() == 1.0);
        assert(model.coefficients().size() == 1);
    }

    return 0;
};