// Copyright (c) 2021 Franka Emika GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <admittance_control/admittance_controller.hpp>
#include <admittance_control/robot_utils.hpp>
#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace {

template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
}
}

namespace admittance_control {

AdmittanceController::AdmittanceController(){
  elapsed_time = 0.0;
  Kp_multiplier = 1; // Initial multiplier for Kp
  Kp = Kp * Kp_multiplier;  // increasing Kp from 0.1 to 1 made robot far less compliant
  control_mode = POSITION_CONTROL; // sets control mode
  //input_control_mode = TARGET_POSITION; // sets position control mode
  D =  2* K.cwiseSqrt(); // set critical damping from the get go
  Kd = 2 * Kp.cwiseSqrt();
}

void AdmittanceController::update_stiffness_and_references(){
  //target by filtering
  /** at the moment we do not use dynamic reconfigure and control the robot via D, K and T **/
  //K = filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * K;
  //D = filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * D;
  nullspace_stiffness_ = filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  //std::lock_guard<std::mutex> position_d_target_mutex_lock(position_and_orientation_d_target_mutex_);
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  
  // Convert the rotation matrix to Euler angles
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
  Eigen::Vector3d euler_angles = orientation_d_.toRotationMatrix().eulerAngles(0, 1, 2); // Roll (X), Pitch (Y), Yaw (Z) 

  /* std::cout << "position_d update_stiffness_and_references is: " << position_d_.transpose() <<  std::endl;  // Debugging */
  //std::cout << "position_d_target update_stiffness_and_references is: " << position_d_target_.transpose() <<  std::endl;  // Debugging
  reference_pose.head(3) << position_d_;
  reference_pose.tail(3) << euler_angles.x(), euler_angles.y(), euler_angles.z();
}


void AdmittanceController::arrayToMatrix(const std::array<double,7>& inputArray, Eigen::Matrix<double,7,1>& resultMatrix)
{
 for(long unsigned int i = 0; i < 7; ++i){
     resultMatrix(i,0) = inputArray[i];
   }
}

void AdmittanceController::arrayToMatrix(const std::array<double,6>& inputArray, Eigen::Matrix<double,6,1>& resultMatrix)
{
 for(long unsigned int i = 0; i < 6; ++i){
     resultMatrix(i,0) = inputArray[i];
   }
}

Eigen::Matrix<double, 7, 1> AdmittanceController::saturateTorqueRate(
  const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
  const Eigen::Matrix<double, 7, 1>& tau_J_d_M) {  
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
  double difference = tau_d_calculated[i] - tau_J_d_M[i];
  tau_d_saturated[i] =
         tau_J_d_M[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}


inline void pseudoInverse(const Eigen::MatrixXd& M_, Eigen::MatrixXd& M_pinv_, bool damped = true) {
  double lambda_ = damped ? 0.2 : 0.0;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_, Eigen::ComputeFullU | Eigen::ComputeFullV);   
  Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals_ = svd.singularValues();
  Eigen::MatrixXd S_ = M_;  // copying the dimensions of M_, its content is not needed.
  S_.setZero();

  for (int i = 0; i < sing_vals_.size(); i++)
     S_(i, i) = (sing_vals_(i)) / (sing_vals_(i) * sing_vals_(i) + lambda_ * lambda_);

  M_pinv_ = Eigen::MatrixXd(svd.matrixV() * S_.transpose() * svd.matrixU().transpose());
}


controller_interface::InterfaceConfiguration
AdmittanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}


controller_interface::InterfaceConfiguration AdmittanceController::state_interface_configuration()
  const {
  std::cout << "STATE INTERFACE CONFIGURATION"  << std::endl;
  controller_interface::InterfaceConfiguration state_interfaces_config;
  state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  
  for (int i = 1; i <= num_joints; ++i) {
    state_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    std::cout << "adding interface " << arm_id_ + "_joint" + std::to_string(i) + "/position" << std::endl;
    state_interfaces_config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
    std::cout << "adding interface " << arm_id_ + "_joint" + std::to_string(i) + "/velocity" << std::endl;
  }
  
  
  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    state_interfaces_config.names.push_back(franka_robot_model_name);
    std::cout << "adding interface " << franka_robot_model_name << std::endl;
  
  }
  

  const std::string full_interface_name = arm_id_ + "/" + state_interface_name_;

  return state_interfaces_config;
}


CallbackReturn AdmittanceController::on_init() {
   auto_declare<std::string>("arm_id", "");
   UserInputServer input_server_obj(&position_d_target_, &rotation_d_target_, &K, &D, &T);
   std::thread input_thread(&UserInputServer::main, input_server_obj, 0, nullptr);
   input_thread.detach();
   return CallbackReturn::SUCCESS;
}


CallbackReturn AdmittanceController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
  std::cout << "CONFIGURING CONTROLLER"  << std::endl;
  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
  franka_semantic_components::FrankaRobotModel(arm_id_ + "/" + k_robot_model_interface_name,
                                               arm_id_ + "/" + k_robot_state_interface_name));
                                               
  try {
    rclcpp::QoS qos_profile(1); // Depth of the message queue
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    franka_state_subscriber = get_node()->create_subscription<franka_msgs::msg::FrankaRobotState>(
    "franka_robot_state_broadcaster/robot_state", qos_profile, 
    std::bind(&AdmittanceController::topic_callback, this, std::placeholders::_1));
    std::cout << "Succesfully subscribed to robot_state_broadcaster" << std::endl;
  }

  catch (const std::exception& e) {
    fprintf(stderr,  "Exception thrown during publisher creation at configure stage with message : %s \n",e.what());
    return CallbackReturn::ERROR;
    }

  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "/robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }

  arm_id_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());



  RCLCPP_DEBUG(get_node()->get_logger(), "configured successfully");
  return CallbackReturn::SUCCESS;
}


CallbackReturn AdmittanceController::on_activate(
  const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

  // Create the subscriber in the on_activate method
  desired_pose_sub = get_node()->create_subscription<geometry_msgs::msg::Pose>(
        "admittance_controller/reference_pose", 
        10,  // Queue size
        std::bind(&AdmittanceController::reference_pose_callback, this, std::placeholders::_1)
    );

  std::array<double, 16> initial_pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_pose.data()));
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
  Eigen::Vector3d euler_angles = orientation_d_.toRotationMatrix().eulerAngles(0, 1, 2); // Roll (X), Pitch (Y), Yaw (Z) 
  //update_stiffness_and_references();
  reference_pose.head(3) = position_d_;
  reference_pose.tail(3) << euler_angles.x(), euler_angles.y(), euler_angles.z();
  x_d.head(3) << position_d_;
  x_d.tail(3) << euler_angles.x(), euler_angles.y(), euler_angles.z();
  /* x_d.head(3) = reference_pose.head(3);
  x_d.tail(3) << reference_pose.tail(3); */
  std::cout << "position_d on_activate is: " << position_d_.transpose() <<  std::endl;    // Debugging
  std::cout << "position_d_target on activation is: " << position_d_target_.transpose() <<  std::endl;    // Debugging
  std::cout << "reference_pose on_activate is: " << reference_pose.head(3) <<  std::endl;    // Debugging
  std::cout << "x_desired head on_activate is: " << x_d.head(3) <<  std::endl;    // Debugging
  std::cout << "x_desired tail on_activate is: " << x_d.tail(3) <<  std::endl;    // Debugging
  std::cout << "Completed Activation process" << std::endl;
  return CallbackReturn::SUCCESS;
}


controller_interface::CallbackReturn AdmittanceController::on_deactivate(
  const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

std::array<double, 6> AdmittanceController::convertToStdArray(const geometry_msgs::msg::WrenchStamped& wrench) {
    std::array<double, 6> result;
    result[0] = wrench.wrench.force.x;
    result[1] = wrench.wrench.force.y;
    result[2] = wrench.wrench.force.z;
    result[3] = wrench.wrench.torque.x;
    result[4] = wrench.wrench.torque.y;
    result[5] = wrench.wrench.torque.z;
    return result;
}

void AdmittanceController::topic_callback(const std::shared_ptr<franka_msgs::msg::FrankaRobotState> msg) {
  O_F_ext_hat_K = convertToStdArray(msg->o_f_ext_hat_k);
  arrayToMatrix(O_F_ext_hat_K, O_F_ext_hat_K_M);
}

// Callback implementation
void AdmittanceController::reference_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg)
{
    // Handle the incoming pose message
    std::cout << "received reference posistion as " <<  msg->position.x << ", " << msg->position.y << ", " << msg->position.z << std::endl;
    position_d_target_ << msg->position.x, msg->position.y, msg->position.z;
    orientation_d_target_.coeffs() << msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w;
    // You can add more processing logic here
    // Update x_d to reflect the new reference pose
/*     x_d.head(3) = position_d_target_;  // New target position
    x_d.tail(3) << msg->orientation.x, msg->orientation.y, msg->orientation.z;  // New target orientation */
}

void AdmittanceController::updateJointStates() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

controller_interface::return_type AdmittanceController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {  

/*   // increment the elapsed time
   elapsed_time += period.seconds();

  // Check if elapsed time is greater than 10 seconds
  if (elapsed_time > 10.0) {
    Kp_multiplier = 10; // Increase Kp multiplier
  }

  // Update Kp based on the multiplier
  Kp = Kp * Kp_multiplier; // Update Kp based on the multiplier
  Kd = 2 * Kp.cwiseSqrt(); // Update Kd based on the new Kp */
  
  std::array<double, 49> mass = franka_robot_model_->getMassMatrix();
  std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  std::array<double, 42> jacobian_array =  franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  std::array<double, 16> pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  jacobian = Eigen::Map<Eigen::Matrix<double, 6, 7>> (jacobian_array.data());
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);
  pseudoInverse(jacobian, jacobian_pinv); 
  M = Eigen::Map<Eigen::Matrix<double, 7, 7>>(mass.data());
  Lambda = (jacobian * M.inverse() * jacobian.transpose()).inverse();
  updateJointStates(); 
  Theta = Lambda;

  // Theta = T * Lambda; // virtual inertia // set another theta here if you don't want it like defined in .hpp

  Eigen::Matrix<double, 6, 6> Q = Lambda * Theta.inverse(); // helper for impedance control law
  /* Eigen::Matrix<double, 6, 6> Q = IDENTITY; // helper for impedance control law */
  w = jacobian * dq_; // cartesian velocity
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(pose.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());

  //Force Updates
  F_ext = 0.001 * F_ext + 0.999 * O_F_ext_hat_K_M; //Filtering

  // Perform the element-wise operation (give a force threshold due to noise with bias)
  //F_ext = (F_ext.array() < 2 && F_ext.array() > -2).select(0, 
  //(F_ext.array() > 2).select(F_ext.array() - 2, F_ext.array() + 2)); // clamp to avoid noise
  
  Eigen::Matrix<double, 6, 1> x_current = Eigen::MatrixXd::Zero(6, 1);
  Eigen::Matrix<double, 6, 1> virtual_error = Eigen::MatrixXd::Zero(6, 1); 

  // compute cartesian error
  error.head(3) << position - position_d_;
  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }

  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.rotation() * error.tail(3);
  //std::cout << "Error outer loop is: " << error.transpose() <<  std::endl;

   // Now, align error.tail(3) to use virtual_error like position error
  // You want to use the virtual_error for rotational error (tail):
  // Set current state
  x_current.head(3) << position;
  x_current.tail(3) << orientation.toRotationMatrix().eulerAngles(0, 1, 2);  // Roll, Pitch, Yaw (X, Y, Z)

  // Admittance control law to compute desired trajectory
  D = 2.05* K.cwiseSqrt() * Lambda.diagonal().cwiseSqrt().asDiagonal();
  //D = 2*K.cwiseSqrt();
  virtual_error.head(3) = x_d.head(3) - position_d_;
  virtual_error.tail(3) = x_d.tail(3) - orientation_d_.toRotationMatrix().eulerAngles(0, 1, 2); // Roll, Pitch, Yaw
  //F_ext again has the opposite sign of what we would expect
  x_ddot_d = Lambda.inverse() * (-F_ext - D * x_dot_d - K * (virtual_error)); // impedance control law
  //x_ddot_d.tail(3).setZero();
  x_dot_d  += x_ddot_d * dt;
  x_d += x_dot_d * dt;

  

  //inner PID position control loop
  //get new inner positional loop error
  Eigen::Quaterniond x_d_orientation_quat = Eigen::AngleAxisd(x_d.tail(3)(0), Eigen::Vector3d::UnitX())
                        * Eigen::AngleAxisd(x_d.tail(3)(1), Eigen::Vector3d::UnitY())
                        * Eigen::AngleAxisd(x_d.tail(3)(2), Eigen::Vector3d::UnitZ());
  
  // normalize the quaternion before calculating the error
  orientation.normalize();
  x_d_orientation_quat.normalize();

  if (x_d_orientation_quat.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }

  //error is overwritten
  error.head(3) << position - x_d.head(3);
  error_quaternion = (orientation.inverse() * x_d_orientation_quat);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.rotation() * error.tail(3);
  //error.tail(3).setZero();

 // std::cout << "postition error pre inner loop is: " << error.head(3).transpose() <<  std::endl;
  
  // change the PID control depending on the control mode
  switch (control_mode)
  {

  case POSITION_CONTROL:
    F_admittance = - Kp * error - Kd * w; // position control, chagning Kp here doesn't have any influence on the compliance, only on the accuracy

  case VELOCITY_CONTROL:
    F_admittance = - Kd * (w - x_dot_d); // velocity control

  }

  /* Kp = 0.1*Kp; 
  Kd = 2 * Kp.cwiseSqrt(); // Update Kd based on the new Kp */

  F_admittance = - Kp * error - Kd * (w); // position control, chagning Kp here doesn't have any influence on the compliance, only on the accuracy

  // Force control and filtering
  N = (Eigen::MatrixXd::Identity(7, 7) - jacobian_pinv * jacobian);
  
  Eigen::VectorXd tau_nullspace(7), tau_d(7);
  tau_nullspace << N * (nullspace_stiffness_ * config_control * (q_d_nullspace_ - q_) - //if config_control = true we control the whole robot configuration
                       (2.0 * sqrt(nullspace_stiffness_)) * dq_);  // if config control ) false we don't care about the joint position

  calculate_tau_friction(); //Gets friction forces for current state
  tau_admittance = jacobian.transpose() * Sm * (F_admittance /*+ F_repulsion + F_potential*/);
  auto tau_total = (tau_admittance + tau_nullspace + coriolis + tau_friction); //add nullspace and coriolis components to desired torque
  tau_d << tau_total;
  tau_d << saturateTorqueRate(tau_d, tau_J_d_M);  // Saturate torque rate to avoid discontinuities
  tau_J_d_M = tau_d;

  for (size_t i = 0; i < 7; ++i) {
    command_interfaces_[i].set_value(tau_d(i));
  }

  
  
  if (outcounter % 1000 / update_frequency == 0){
    /** 
    std::cout << "F_ext_robot [N]" << std::endl;
    std::cout << O_F_ext_hat_K << std::endl;
    std::cout << O_F_ext_hat_K_M << std::endl;
    std::cout << Lambda*Theta.inverse() << std::endl;
    std::cout << "tau_d" << std::endl;
    std::cout << tau_d << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << tau_nullspace << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << coriolis << std::endl;
    std::cout << "Inertia scaling [m]: " << std::endl;
    std::cout << T << std::endl;
    *///
    //std::cout << "Lambda: " << Lambda << std::endl;
    //std::cout << "External Force is: " << F_ext.transpose() <<  std::endl;
    /* std::cout << "Kp multiplier is: " << Kp_multiplier <<  std::endl; */
    /* std::cout << "Kp is: " << Kp_multiplier <<  std::endl; */
    //std::cout << "Current orientation: " << orientation.coeffs().transpose() << std::endl;
    /* std::cout << "Target orientation: " << orientation_d_target_.toRotationMatrix().eulerAngles(0, 1, 2).transpose().transpose() << std::endl;
    std::cout << "Desired orientation: " << orientation_d_.toRotationMatrix().eulerAngles(0, 1, 2).transpose() << std::endl;
    //std::cout << "Error quaternion: " << error_quaternion.coeffs().transpose() << std::endl;
    std::cout << "virtual rotation error is: " << virtual_error.tail(3).transpose() <<  std::endl;
    std::cout << "position is: " << x_current.transpose() <<  std::endl;
    //std::cout << "postition error is: " << error.head(3).transpose() <<  std::endl;
    std::cout << "reference is: " << reference_pose.transpose() <<  std::endl;
    std::cout << "Elapsed time is: " << elapsed_time <<  std::endl;
    std::cout << "Desired Acceleration is: " << x_ddot_d.transpose() <<  std::endl;
    //std::cout << "Desired Velocity is: " << x_dot_d.transpose() <<  std::endl;
    std::cout << "X desired is: " << x_d.transpose() <<  std::endl;
    //std::cout << "position target is: " << position_d_target_.transpose() <<  std::endl;
    //std::cout << "position_d is: " << position_d_.transpose() <<  std::endl;
    //std::cout << "x_current is: " << x_current.transpose() <<  std::endl;
    //std::cout << "Inertia is : " << Lambda <<  std::endl;
    //std::cout << "--------------------------------------------------" <<  std::endl;
    //std::cout << "tau friction is " << tau_friction.transpose() << std::endl;
    //std::cout << "tau desired is " << tau_d.transpose() << std::endl;
    std::cout << "Control mode is: " << control_mode <<  std::endl; */
    //std::cout << "F admittance is: " << F_admittance.transpose() <<  std::endl;
    //std::cout << "Error is: " << error.transpose() <<  std::endl;
  }
  outcounter++;
  update_stiffness_and_references();
  return controller_interface::return_type::OK;
}
}

// namespace admittance_control
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(admittance_control::AdmittanceController,
                       controller_interface::ControllerInterface)