// tests/test_kalman_filter.cpp
#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <optional>
#include <vector>

#include "kf_linear.hpp"

using Catch::Approx;
using namespace cppx::kf_linear;

TEST_CASE("Predict-only step leaves state at A*x and P -> A P A^T + Q") {
    using KF = KFLinear<2, 1>;
    using StateVec = typename KF::StateVec;
    using StateMat = typename KF::StateMat;
    using ObsMat = typename KF::ObsMat;
    using MeasMat = typename KF::MeasMat;
    using MeasVec = typename KF::MeasVec;

    StateVec x0;
    x0 << 2.0, -1.0;
    StateMat P0;
    P0 << 1.0, 0.2, 0.2, 2.0;

    StateMat A;
    A << 1.0, 1.0, 0.0, 1.0; // constant-velocity
    ObsMat H;
    H << 1.0, 0.0;
    StateMat Q = 0.1 * StateMat::Identity();
    MeasMat R = MeasMat::Identity();

    KF kf(x0, P0, A, H, Q, R);

    kf.step(std::nullopt); // no measurement

    StateVec x_expected = A * x0;
    StateMat P_expected = A * P0 * A.transpose() + Q;

    REQUIRE(kf.state().isApprox(x_expected, 1e-12));
    REQUIRE(kf.covariance().isApprox(P_expected, 1e-12));
}

TEST_CASE("Scalar KF update matches closed-form") {
    using KF = KFLinear<1, 1>;
    using StateVec = typename KF::StateVec;
    using StateMat = typename KF::StateMat;
    using ObsMat = typename KF::ObsMat;
    using MeasMat = typename KF::MeasMat;
    using MeasVec = typename KF::MeasVec;

    // 1D model: x_k = x_{k-1} + w, z_k = x_k + v
    StateVec x0;
    x0 << 0.0;
    StateMat P0;
    P0 << 1.0;

    StateMat A;
    A << 1.0;
    ObsMat H;
    H << 1.0;
    StateMat Q;
    Q << 0.25; // process noise var
    MeasMat R;
    R << 1.0; // measurement noise var

    KF kf(x0, P0, A, H, Q, R);

    // Step 1: no measurement
    kf.step(std::nullopt);
    double Pp = 1.0 + 0.25; // P' = P0 + Q
    double xp = 0.0;        // x' = x0
    REQUIRE(kf.state()(0) == Approx(xp));
    REQUIRE(kf.covariance()(0, 0) == Approx(Pp));

    // Step 2: with measurement z = 1.2
    MeasVec z;
    z << 1.2;
    kf.step(z);

    // After second predict: Ppred = Pp + Q
    double Ppred = Pp + 0.25; // = 1.5
    double K = Ppred / (Ppred + 1.0);
    double x_post = xp + K * (1.2 - xp);
    double P_post = (1.0 - K) * Ppred; // Joseph reduces to this in scalar

    REQUIRE(kf.state()(0) == Approx(x_post).epsilon(1e-12));
    REQUIRE(kf.covariance()(0, 0) == Approx(P_post).epsilon(1e-12));
}

TEST_CASE("Batch filter handles missing and present measurements") {
    using KF = KFLinear<2, 1>;
    using StateVec = typename KF::StateVec;
    using StateMat = typename KF::StateMat;
    using ObsMat = typename KF::ObsMat;
    using MeasMat = typename KF::MeasMat;
    using MeasVec = typename KF::MeasVec;

    StateVec x0;
    x0 << 0.0, 0.0;
    StateMat P0 = StateMat::Identity();

    StateMat A;
    A << 1.0, 1.0, 0.0, 1.0;
    ObsMat H;
    H << 1.0, 0.0;
    StateMat Q = 0.01 * StateMat::Identity();
    MeasMat R = 0.25 * MeasMat::Identity();

    KF kf(x0, P0, A, H, Q, R);

    std::vector<std::optional<MeasVec>> zs;
    zs.push_back(std::nullopt); // t=1: no measurement
    MeasVec z1;
    z1 << 0.8;
    zs.push_back(z1); // t=2
    MeasVec z2;
    z2 << 1.6;
    zs.push_back(z2);           // t=3
    zs.push_back(std::nullopt); // t=4

    auto states = kf.filter(zs);
    REQUIRE(states.size() == zs.size());
}
