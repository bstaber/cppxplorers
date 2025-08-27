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
    using MeasVec = Eigen::Matrix<double, Ny, 1>;
    using MeasMat = Eigen::Matrix<double, Ny, Ny>;
    using ObsMat = Eigen::Matrix<double, Ny, Nx>;

    /// Construct filter with initial condition and model matrices.
    KFLinear(const StateVec &initial_state, const StateMat &initial_covariance,
             const StateMat &transition_matrix, const ObsMat &observation_matrix,
             const StateMat &process_covariance, const MeasMat &measurement_covariance)
        : x_(initial_state), P_(initial_covariance), A_(transition_matrix), H_(observation_matrix),
          Q_(process_covariance), R_(measurement_covariance) {
        std::cout << "DEBUG" << std::endl;
        const auto n = x_.rows();
        if (A_.rows() != n || A_.cols() != n)
            throw std::invalid_argument("A must be n×n and match x dimension");
        if (P_.rows() != n || P_.cols() != n)
            throw std::invalid_argument("P must be n×n");
        if (Q_.rows() != n || Q_.cols() != n)
            throw std::invalid_argument("Q must be n×n");
        if (H_.cols() != n)
            throw std::invalid_argument("H must have n columns");
        const auto m = H_.rows();
        if (R_.rows() != m || R_.cols() != m)
            throw std::invalid_argument("R must be m×m with m = H.rows()");
    }

    /// Predict step (no control).
    void predict() {
        x_ = A_ * x_;
        P_ = A_ * P_ * A_.transpose() + Q_;
    }

    /// Update step with a measurement z.
    void update(const MeasVec &z) {
        // Innovation
        MeasVec nu = z - H_ * x_;

        // Innovation covariance
        MeasMat S = H_ * P_ * H_.transpose() + R_;

        // Solve for K without forming S^{-1}
        Eigen::LDLT<MeasMat> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            throw std::runtime_error("KFLinear::update: LDLT failed (S not SPD?)");
        }

        // K = P H^T S^{-1}   via solve: S * (K^T) = (P H^T)^T
        const auto PHt = P_ * H_.transpose();                                      // (Nx × Ny)
        Eigen::Matrix<double, Nx, Ny> K = ldlt.solve(PHt.transpose()).transpose(); // (Nx × Ny)

        // State update
        x_ += K * nu;

        // Joseph form (use runtime-sized Identity for Dynamic Nx)
        StateMat I = StateMat::Identity(P_.rows(), P_.cols());
        P_ = (I - K * H_) * P_ * (I - K * H_).transpose() + K * R_ * K.transpose();

        // Re-symmetrize for numerical hygiene
        P_ = 0.5 * (P_ + P_.transpose());
    }

    /// One full step: predict then (optionally) update.
    void step(const std::optional<MeasVec> &measurement) {
        predict();
        if (measurement) {
            update(*measurement);
        }
    }

    /// Run over a sequence of (optional) measurements.
    std::vector<StateVec> filter(const std::vector<std::optional<MeasVec>> &measurements) {
        std::vector<StateVec> out;
        out.reserve(measurements.size());
        for (const auto &z : measurements) {
            step(z);
            out.push_back(x_);
        }
        return out;
    }

    // Accessors
    [[nodiscard]] const StateVec &state() const { return x_; }
    [[nodiscard]] const StateMat &covariance() const { return P_; }

    // (Optional) setters if you want to tweak model online
    void set_transition(const StateMat &A) { A_ = A; }
    void set_observation(const ObsMat &H) { H_ = H; }
    void set_process_noise(const StateMat &Q) { Q_ = Q; }
    void set_measurement_noise(const MeasMat &R) { R_ = R; }

  private:
    // Model
    StateMat A_; ///< State transition
    ObsMat H_;   ///< Observation
    StateMat Q_; ///< Process noise covariance
    MeasMat R_;  ///< Measurement noise covariance

    // Estimates
    StateVec x_; ///< State mean
    StateMat P_; ///< State covariance
};
