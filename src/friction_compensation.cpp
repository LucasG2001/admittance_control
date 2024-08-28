#include <admittance_control/admittance_controller.hpp>

namespace admittance_control{
    //Calculates the friction forces acting on the robot's joints depending on joint rotational speed. 
    //Exerts torque up to a certain empirically detected static friction threshold. 
    //TODO: Afterwards, goes into the viscous domain and follows a linear raise depending on empiric parameters
    void AdmittanceController::calculate_tau_friction(){
        if (friction){
            double alpha = 0.01;//constant for exponential filter in relation to static friction moment        
            dq_filtered = alpha* dq_ + (1 - alpha) * dq_filtered; //Filtering dq of every joint
            tau_admittance_filtered = alpha* tau_admittance + (1 - alpha) * tau_admittance_filtered; //Filtering tau_admittance
            //Creating and filtering a "fake" tau_admittance with own weights, optimized for friction compensation (else friction compensation would get stronger with higher stiffnesses)
            Eigen::Matrix<double, 7, 1> tau_threshold = jacobian.transpose() * Sm * (-alpha * (D_friction*(jacobian*dq_) + K_friction * error)) + (1 - alpha) * tau_threshold;
            //Creating "fake" dq, that acts only in the impedance-space, else dq in the nullspace also gets compensated, which we do not want due to null-space movement
            Eigen::Matrix<double, 7, 1> dq_compensated = dq_filtered - N * dq_filtered;

            //Calculation of friction force according to Bachelor Thesis: https://polybox.ethz.ch/index.php/s/iYj8ALPijKTAC2z?path=%2FFriction%20compensation
            f = beta.cwiseProduct(dq_compensated) + offset_friction;
            g(4) = (coulomb_friction(4) + (static_friction(4) - coulomb_friction(4)) * exp(-1 * std::abs(dq_compensated(4)/dq_s(4))));
            g(6) = (coulomb_friction(6) + (static_friction(6) - coulomb_friction(6)) * exp(-1 * std::abs(dq_compensated(6)/dq_s(6))));
            dz = dq_compensated.array() - dq_compensated.array().abs() / g.array() * sigma_0.array() * z.array() + 0.025* tau_threshold.array()/*(jacobian.transpose() * K * error).array()*/;
            dz(6) -= 0.02*tau_threshold(6);
            z = 0.001 * dz + z;
            tau_friction = sigma_0.array() * z.array() + 100 * sigma_1.array() * dz.array() + f.array();  
        }

        else
        {
            tau_friction.setZero();

        }
        
    }
}

