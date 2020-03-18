#include <iostream>
#include "ukf.h"

namespace tracking{

/******************************************************************************/
using sensors_fusion::ObjectTrack;
using sensors_fusion::ObjectTrackArray;
using sensors_fusion::ObjectTrackArrayPtr;

UnscentedKF::UnscentedKF()
    {
    // Define parameters
    params_.da_ped_dist_pos = 1.0;
    params_.da_ped_dist_form = 2.0;
    params_.da_car_dist_pos = 2.0;
    params_.da_car_dist_form = 5.0;
    params_.tra_dim_z = 2;
    params_.tra_dim_x = 5;
    params_.tra_dim_x_aug = 7;
    params_.tra_std_lidar_x = 0.15;
    params_.tra_std_lidar_y = 0.15;
    params_.tra_std_acc = 0.4;
    params_.tra_std_yaw_rate = 0.314;
    params_.tra_lambda = 2;

    params_.tra_n_init = 3;
    params_.tra_max_age = 10;

    params_.tra_occ_factor = 2;
    params_.p_init_x = 1;
    params_.p_init_y = 1;
    params_.p_init_v = 10;
    params_.p_init_yaw = 10;
    params_.p_init_yaw_rate = 1;

#if 0
    // Print parameters
    ROS_INFO_STREAM("da_ped_dist_pos " << params_.da_ped_dist_pos);
    ROS_INFO_STREAM("da_ped_dist_form " << params_.da_ped_dist_form);
    ROS_INFO_STREAM("da_car_dist_pos " << params_.da_car_dist_pos);
    ROS_INFO_STREAM("da_car_dist_form " << params_.da_car_dist_form);
    ROS_INFO_STREAM("tra_dim_z " << params_.tra_dim_z);
    ROS_INFO_STREAM("tra_dim_x " << params_.tra_dim_x);
    ROS_INFO_STREAM("tra_dim_x_aug " << params_.tra_dim_x_aug);
    ROS_INFO_STREAM("tra_std_lidar_x " << params_.tra_std_lidar_x);
    ROS_INFO_STREAM("tra_std_lidar_y " << params_.tra_std_lidar_y);
    ROS_INFO_STREAM("tra_std_acc " << params_.tra_std_acc);
    ROS_INFO_STREAM("tra_std_yaw_rate " << params_.tra_std_yaw_rate);
    ROS_INFO_STREAM("tra_lambda " << params_.tra_lambda);

    ROS_INFO_STREAM("tra_n_init " << params_.tra_n_init);
    ROS_INFO_STREAM("tra_max_age " << params_.tra_max_age);

    ROS_INFO_STREAM("tra_occ_factor " << params_.tra_occ_factor);
    ROS_INFO_STREAM("p_init_x " << params_.p_init_x);
    ROS_INFO_STREAM("p_init_y " << params_.p_init_y);
    ROS_INFO_STREAM("p_init_v " << params_.p_init_v);
    ROS_INFO_STREAM("p_init_yaw " << params_.p_init_yaw);
    ROS_INFO_STREAM("p_init_yaw_rate " << params_.p_init_yaw_rate);
#endif

    // Measurement covariance
    R_laser_ = MatrixXd(params_.tra_dim_z, params_.tra_dim_z);
    R_laser_ << params_.tra_std_lidar_x * params_.tra_std_lidar_x, 0,
        0, params_.tra_std_lidar_y * params_.tra_std_lidar_y;

    // Define weights for UKF
    weights_ = VectorXd(2 * params_.tra_dim_x_aug + 1);
    weights_(0) = params_.tra_lambda /
        (params_.tra_lambda + params_.tra_dim_x_aug);
    for (int i = 1; i < 2 * params_.tra_dim_x_aug + 1; i++) {
        weights_(i) = 0.5 / (params_.tra_dim_x_aug + params_.tra_lambda);
    }

    // Random color for track
    rng_(2345);
}

UnscentedKF::~UnscentedKF(){

}

void UnscentedKF::initialize(long long id, const sensors_fusion::ObjectTrack& obj, double time_stamp)
{
  state_ = TrackState::Tentative;

  Track& tr = track_;
  // Add id and increment
  tr.id = id;

  // Initialize history information
  tr.hist.time_since_update = 0;
  tr.hist.hits = 1;
  tr.hist.age = 1;

  // Add state information
  tr.sta.x = VectorXd::Zero(params_.tra_dim_x);
  tr.sta.x[0] = obj.world_pos.point.x;
  tr.sta.x[1] = obj.world_pos.point.y;
//  tr.sta.x[3] = obj.heading;
  tr.sta.z = obj.world_pos.point.z;
  tr.sta.P = MatrixXd::Zero(params_.tra_dim_x, params_.tra_dim_x);
  tr.sta.P << params_.p_init_x,  0,  0,  0,  0,
      0,  params_.p_init_y,  0,  0,  0,
      0,  0,  params_.p_init_v,  0,  0,
      0,  0,  0,params_.p_init_yaw,  0,
      0,  0,  0,  0,  params_.p_init_yaw_rate;
  tr.sta.Xsig_pred = MatrixXd::Zero(params_.tra_dim_x,
      2 * params_.tra_dim_x_aug + 1);

  // Add geometric information
  tr.geo.width = obj.width;
  tr.geo.length = obj.length;
  tr.geo.height = obj.height;
  tr.geo.orientation = obj.orientation;

  // Add semantic information
  tr.sem.name = obj.object_type;
  tr.sem.confidence = obj.confidence;

  // Add unique color
  tr.r = rng_.uniform(0, 255);
  tr.g = rng_.uniform(0, 255);
  tr.b = rng_.uniform(0, 255);

  // Store time stamp for next frame
  last_time_stamp_ = time_stamp;
}

void UnscentedKF::Prediction(const double delta_t){


    // Buffer variables
    VectorXd x_aug = VectorXd(params_.tra_dim_x_aug);
    MatrixXd P_aug = MatrixXd(params_.tra_dim_x_aug, params_.tra_dim_x_aug);
    MatrixXd Xsig_aug =
        MatrixXd(params_.tra_dim_x_aug, 2 * params_.tra_dim_x_aug + 1);

    // Grab track
    Track & track = track_;

    // Update track history
    ++track.hist.age;
    ++track.hist.time_since_update;

    /******************************************************************************
     * 1. Generate augmented sigma points
     */

    // Fill augmented mean state
    x_aug.head(5) = track.sta.x;
    x_aug(5) = 0;
    x_aug(6) = 0;

    // Fill augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = track.sta.P;
    P_aug(5,5) = params_.tra_std_acc * params_.tra_std_acc;
    P_aug(6,6) = params_.tra_std_yaw_rate * params_.tra_std_yaw_rate;

    // Create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    // Create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for(int j = 0; j < params_.tra_dim_x_aug; j++){
      Xsig_aug.col(j + 1) = x_aug +
          sqrt(params_.tra_lambda + params_.tra_dim_x_aug) * L.col(j);
      Xsig_aug.col(j + 1 + params_.tra_dim_x_aug) = x_aug -
          sqrt(params_.tra_lambda + params_.tra_dim_x_aug) * L.col(j);
    }

    /******************************************************************************
     * 2. Predict sigma points
     */

    for(int j = 0; j < 2 * params_.tra_dim_x_aug + 1; j++){

      // Grab values for better readability
      double p_x = Xsig_aug(0,j);
      double p_y = Xsig_aug(1,j);
      double v = Xsig_aug(2,j);
      double yaw = Xsig_aug(3,j);
      double yawd = Xsig_aug(4,j);
      double nu_a = Xsig_aug(5,j);
      double nu_yawdd = Xsig_aug(6,j);

      // Predicted state values
      double px_p, py_p;

      // Avoid division by zero
      if(fabs(yawd) > 0.001){
        px_p = p_x + v/yawd * ( sin (yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
      }
      else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
      }
      double v_p = v;
      double yaw_p = yaw + yawd * delta_t;
      double yawd_p = yawd;

      // Add noise
      px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
      py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
      v_p = v_p + nu_a * delta_t;
      yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
      yawd_p = yawd_p + nu_yawdd * delta_t;

      // Write predicted sigma point into right column
      track.sta.Xsig_pred(0,j) = px_p;
      track.sta.Xsig_pred(1,j) = py_p;
      track.sta.Xsig_pred(2,j) = v_p;
      track.sta.Xsig_pred(3,j) = yaw_p;
      track.sta.Xsig_pred(4,j) = yawd_p;
    }

    /******************************************************************************
     * 3. Predict state vector and state covariance
     */
    // Predicted state mean
    track.sta.x.fill(0.0);
    for(int j = 0; j < 2 * params_.tra_dim_x_aug + 1; j++) {
      track.sta.x = track.sta.x + weights_(j) *
          track.sta.Xsig_pred.col(j);
    }

    // Predicted state covariance matrix
    track.sta.P.fill(0.0);

    // Iterate over sigma points
    for(int j = 0; j < 2 * params_.tra_dim_x_aug + 1; j++) {

      // State difference
      VectorXd x_diff = track.sta.Xsig_pred.col(j) - track.sta.x;

      // Angle normalization
      while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
      while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

      track.sta.P = track.sta.P + weights_(j) * x_diff *
          x_diff.transpose() ;
    }
}

void UnscentedKF::predict(const double& time_stamp)
{
  double delta_t = (time_stamp - last_time_stamp_);
  Prediction(delta_t);

  last_time_stamp_ = time_stamp;
}


void UnscentedKF::update(const sensors_fusion::ObjectTrack& detected_objects)
{

  // Buffer variables
  VectorXd z = VectorXd(params_.tra_dim_z);
  MatrixXd Zsig;
  VectorXd z_pred = VectorXd(params_.tra_dim_z);
  MatrixXd S = MatrixXd(params_.tra_dim_z, params_.tra_dim_z);
  MatrixXd Tc = MatrixXd(params_.tra_dim_x, params_.tra_dim_z);


  // Grab track
  Track & track = track_;

  // Increment bad aging
//  track.hist.bad_age++;
  // If track has found a measurement update it

  // Grab measurement
  z << detected_objects.world_pos.point.x, detected_objects.world_pos.point.y;

  /******************************************************************************
   * 1. Predict measurement
   */
  // Init measurement sigma points
  Zsig = track.sta.Xsig_pred.topLeftCorner(params_.tra_dim_z,
      2 * params_.tra_dim_x_aug + 1);

  // Mean predicted measurement
  z_pred.fill(0.0);
  for(int j = 0; j < 2 * params_.tra_dim_x_aug + 1; j++) {
    z_pred = z_pred + weights_(j) * Zsig.col(j);
  }

  S.fill(0.0);
  Tc.fill(0.0);
  for(int j = 0; j < 2 * params_.tra_dim_x_aug + 1; j++) {

    // Residual
    VectorXd z_sig_diff = Zsig.col(j) - z_pred;
    S = S + weights_(j) * z_sig_diff * z_sig_diff.transpose();

    // State difference
    VectorXd x_diff = track.sta.Xsig_pred.col(j) - track.sta.x;

    // Angle normalization
    while(x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
    while(x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(j) * x_diff * z_sig_diff.transpose();
  }

  // Add measurement noise covariance matrix
  S = S + R_laser_;

  /******************************************************************************
   * 2. Update state vector and covariance matrix
   */
  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = z - z_pred;

  // Update state mean and covariance matrix
  track.sta.x = track.sta.x + K * z_diff;
  track.sta.P = track.sta.P - K * S * K.transpose();

  /******************************************************************************
   * 3. Update geometric information of track
   */
  // Calculate area of detection and track
  float det_area =
      detected_objects.length *
      detected_objects.width;
  float tra_area = track.geo.length * track.geo.width;

  // If track became strongly smaller keep the shape
  if(params_.tra_occ_factor * det_area < tra_area){
    ROS_WARN("Track [%d] probably occluded because of dropping size"
        " from [%f] to [%f]", track.id, tra_area, det_area);
  }
  // Else update the form of the track with measurement
  else{
    track.geo.length =
        detected_objects.length;
    track.geo.width =
        detected_objects.width;
  }

  // Update orientation and ground level
  track.geo.orientation =
      detected_objects.orientation;
  track.sta.z =
      detected_objects.world_pos.point.z;

  // Update History
  ++track.hist.hits;
  track.hist.time_since_update = 0;

  if (state_ == TrackState::Tentative && track.hist.hits >= params_.tra_n_init) {
    state_ = TrackState::Confirmed;
  }
}

} // namespace tracking
