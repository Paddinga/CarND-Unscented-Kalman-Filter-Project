#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

UKF::UKF() {
    is_initialized_ = false;
    previous_timestamp_ = 0;
    use_laser_ = true;
    use_radar_ = true;
    std_a_ = 0.5;                // Process noise standard deviation longitudinal acceleration in m/s^2
    std_yawdd_ = 0.3;            // Process noise standard deviation yaw acceleration in rad/s^2
    std_laspx_ = 0.15;          // Laser measurement noise standard deviation position1 in m
    std_laspy_ = 0.15;          // Laser measurement noise standard deviation position2 in m
    std_radr_ = 0.3;            // Radar measurement noise standard deviation radius in m
    std_radphi_ = 0.03;         // Radar measurement noise standard deviation angle in rad
    std_radrd_ = 0.3;           // Radar measurement noise standard deviation radius change in m/s
    n_x_ = 5;                   // State dimension
    n_aug_ = 7;                 // Augmented state dimension
    n_z_laser_ = 2;             // Laser measurement dimensions
    n_z_radar_ = 3;             // Radar measurement dimensions
    lambda_ = 3 - n_x_;         // Sigma point spreading parameter
    lambda_aug_ = 3 - n_aug_;   // Sigma point spreading parameter for augmented state
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);    // predicted sigma points matrix
    weights_ = VectorXd(2 * n_aug_ + 1);            // Weights of sigma points
    double weight_0 = lambda_aug_ / (lambda_aug_ + n_aug_);
    weights_(0) = weight_0;
    for (int i=1; i < 2 * n_aug_ + 1; i++) {
        double weight = 0.5 / (n_aug_ + lambda_aug_);
        weights_(i) = weight;
    }
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
/***************************
*   Initialization and first measurement
***************************/
    if (!is_initialized_) {
        x_ = VectorXd(n_x_);
        P_ = MatrixXd(n_x_,n_x_);
        float px, py, v, yaw, yaw_dot;
        if (use_radar_ && (meas_package.sensor_type_ == MeasurementPackage::RADAR)) {
            float rho = meas_package.raw_measurements_[0];
            float phi = meas_package.raw_measurements_[1];
            float rho_dot = meas_package.raw_measurements_[2];
            px = rho * cos(phi);
            py = rho * sin(phi);
            v = rho_dot;
            yaw = 0;
            yaw_dot = 0;
            P_.fill(0.0);
            P_(0,0) = std_radr_ * std_radr_;
            P_(1,1) = std_radr_ * std_radr_;
            P_(2,2) = rho_dot * rho_dot;
            P_(3,3) = M_PI * M_PI;
            P_(4,4) = 0.1 * 0.1;
        }
        else if (use_laser_ && (meas_package.sensor_type_ == MeasurementPackage::LASER)) {
            px = meas_package.raw_measurements_[0];
            py = meas_package.raw_measurements_[1];
            v = 5;
            yaw = 0;
            yaw_dot = 0;
            P_.fill(0.0);
            P_(0,0) = std_laspx_ * std_laspx_;
            P_(1,1) = std_laspy_ * std_laspy_;
            P_(2,2) = 5 * 5;
            P_(3,3) = M_PI * M_PI;
            P_(4,4) = 0.1 * 0.1;
        }
        else{
            // in case use_radar or use_laser are set to false - is_initialized should not be set to true
            return;
        }
        x_ << px, py, v, yaw, yaw_dot;
        is_initialized_ = true;
        previous_timestamp_ = meas_package.timestamp_;
        return;
    }
/***************************
*   Prediction
***************************/
    // elapsed time between current and previous measurement (in seconds)
    float dt = (meas_package.timestamp_ - previous_timestamp_) / 1e6;
    previous_timestamp_ = meas_package.timestamp_;
    Prediction(dt); // perform prediction
/***************************
*   Radar updates
***************************/
    if (use_radar_ && (meas_package.sensor_type_ == MeasurementPackage::RADAR)) {
        UpdateRadar(meas_package);
    }
/***************************
*   Laser updates
***************************/
    else if (use_laser_ && (meas_package.sensor_type_ == MeasurementPackage::LASER)) {
        UpdateLidar(meas_package);
    }
    cout << "x_ = " << x_ << endl;
    cout << "P_ = " << P_ << endl;
}

void UKF::Prediction(double delta_t) {
    // create sigma points for augmented state
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    AugmentedSigmaPoints(&Xsig_aug);
    // predict sigma points
    SigmaPointPrediction(Xsig_aug, delta_t);
    // predict x(k+1|k) and P(k+1|k) based on the predicted sigma points
    PredictMeanAndCovariance();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
    // create sigma points, mean and covariance
    MatrixXd Zsig = MatrixXd(n_z_laser_, 2 * n_aug_ + 1);
    VectorXd z_pred = VectorXd(n_z_laser_);
    MatrixXd S = MatrixXd(n_z_laser_, n_z_laser_);
    // predict measurement
    PredictLidarMeasurement(&Zsig, &z_pred, &S);
    // update state
    UpdateState(meas_package.raw_measurements_, z_pred, S, Zsig);
    // calculate Normalized Innovation Squared
    NIS_laser_ = CalculateNIS(meas_package.raw_measurements_, z_pred, S);
}


void UKF::UpdateRadar(MeasurementPackage meas_package) {
    // create sigma points, mean and covariance
    MatrixXd Zsig = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);
    VectorXd z_pred = VectorXd(n_z_radar_);
    MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
    // predict measurement
    PredictRadarMeasurement(&Zsig, &z_pred, &S);
    // update state
    UpdateState(meas_package.raw_measurements_, z_pred, S, Zsig);
    // calculate Normalized Innovation Squared
    NIS_radar_ = CalculateNIS(meas_package.raw_measurements_, z_pred, S);
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {
    // create augmented mean vector, state covariance and sigma point matrix
    VectorXd x_aug = VectorXd(n_aug_);
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    //create augmented mean state and covariance matrix
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_ * std_a_;
    P_aug(6,6) = std_yawdd_ * std_yawdd_;
    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();
    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i < n_aug_; i++)
    {
        Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_aug_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_aug_ + n_aug_) * L.col(i);
    }
    *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t) {
    // predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);
        // avoid division by zero
        double px_p, py_p;
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin (yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }
        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;
        // add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;
        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;
        // out
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }
}

void UKF::PredictMeanAndCovariance() {
    // mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }
    if (x_(2) < 0) {
        x_(2) = -x_(2);
        x_(3) += M_PI;
    }
    while (x_(3) > M_PI) x_(3) -= 2. * M_PI;
    while (x_(3) < -M_PI) x_(3) += 2. * M_PI;
    //covariance
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    }
}

void UKF::PredictRadarMeasurement(MatrixXd* Zsig_out, VectorXd* z_out, MatrixXd* S_out) {
    // sigma points
    MatrixXd Zsig = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);
        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;
        Zsig(0,i) = sqrt(p_x * p_x + p_y * p_y);
        Zsig(1,i) = atan2(p_y, p_x);
        Zsig(2,i) = (p_x * v1 + p_y * v2 ) / sqrt(p_x * p_x + p_y * p_y);
    }
    // mean
    VectorXd z_pred = VectorXd(n_z_radar_);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }
    // covariance
    MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    // add noise
    MatrixXd R = MatrixXd(n_z_radar_, n_z_radar_);
    R.fill(0.0);
    R(0,0) = std_radr_ * std_radr_;
    R(1,1) = std_radphi_ * std_radphi_;
    R(2,2) = std_radrd_ * std_radrd_;
    S = S + R;
    // out
    *z_out = z_pred;
    *S_out = S;
    *Zsig_out = Zsig;
}

void UKF::PredictLidarMeasurement(MatrixXd* Zsig_out, VectorXd* z_out, MatrixXd* S_out) {
    // sigma points
    MatrixXd Zsig = MatrixXd(n_z_laser_, 2 * n_aug_ + 1);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        Zsig(0,i) = Xsig_pred_(0,i); // p_x
        Zsig(1,i) = Xsig_pred_(1,i); // p_y
    }
    // mean
    VectorXd z_pred = VectorXd(n_z_laser_);
    z_pred.fill(0.0);
    for (int i=0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }
    // covariance
    MatrixXd S = MatrixXd(n_z_laser_, n_z_laser_);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    // add noise
    MatrixXd R = MatrixXd(n_z_laser_, n_z_laser_);
    R.fill(0.0);
    R(0,0) = std_laspx_ * std_laspx_;
    R(1,1) = std_laspy_ * std_laspy_;
    S = S + R;
    // out
    *z_out = z_pred;
    *S_out = S;
    *Zsig_out = Zsig;
}

void UKF::UpdateState(VectorXd z, VectorXd z_pred, MatrixXd S, MatrixXd Zsig) {
    // measurement dimension
    int n_z = z.size();
    // cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    // Kalman gain
    MatrixXd K = Tc * S.inverse();
    // update
    VectorXd z_diff = z - z_pred;
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    // mean
    x_ = x_ + K * z_diff;
    if (x_(2) < 0) {
        x_(2) = -x_(2);
        x_(3) += M_PI;
    }
    while (x_(3) > M_PI) x_(3) -= 2. * M_PI;
    while (x_(3) < -M_PI) x_(3) += 2. * M_PI;
    // covariance
    P_ = P_ - K * S * K.transpose();
}

double UKF::CalculateNIS(VectorXd z, VectorXd z_pred, MatrixXd S) {
    VectorXd z_diff = z - z_pred;
    double e = z_diff.transpose() * S.inverse() * z_diff;
    return e;
}
