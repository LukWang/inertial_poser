#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>


using namespace ceres;

class OrientationCost_rArm {
    private:
        const Eigen::Matrix<double, 3, 3> _rArm_imu_ori;
    
    public:
        OrientationCost_rArm(
            const Eigen::Matrix<double, 3, 3> rArm_imu_ori)
            :_rArm_imu_ori(rArm_imu_ori){}


    template <typename T> bool operator()(const T* hip_joint, const T* spine_joint, const T* rArm_joint, T* cost_ori){
        Eigen::Matrix<T, 3 ,3> ori = Eigen::Matrix<T, 3, 3>::Identity(3, 3);
        Eigen::Matrix<T, 3, 3> residual;
        Eigen::Quaternion<T> q_res;


        ori *= Eigen::AngleAxis<T>(hip_joint[0], Eigen::Vector3d::UnitZ())
            * Eigen::AngleAxis<T>(hip_joint[1], Eigen::Vector3d::UnitX())
            * Eigen::AngleAxis<T>(hip_joint[2], Eigen::Vector3d::UnitY());

        for(int i = 0; i < 9;i = i + 3)
        {
            ori *= Eigen::AngleAxis<T>(spine_joint[i], Eigen::Vector3d::UnitZ())
                 * Eigen::AngleAxis<T>(spine_joint[i+1], Eigen::Vector3d::UnitX())
                 * Eigen::AngleAxis<T>(spine_joint[i+2], Eigen::Vector3d::UnitY());
        }


        for(int i = 0; i < 9;i = i + 3)
        {
            ori *= Eigen::AngleAxis<T>(rArm_joint[i], Eigen::Vector3d::UnitZ())
                 * Eigen::AngleAxis<T>(rArm_joint[i+1], Eigen::Vector3d::UnitX())
                 * Eigen::AngleAxis<T>(rArm_joint[i+2], Eigen::Vector3d::UnitY());
        }

        residual = ori.inverse() * _rArm_imu_ori;
        q_res = residual;
        q_res.normalize();
        cost_ori[0] = q_res.x() * q_res.x()
                    + q_res.y() * q_res.y()
                    + q_res.z() * q_res.z();
        
        return true;
    }

};

class OrientationCost_lArm {
    private:
        const Eigen::Matrix<double, 3, 3> _lArm_imu_ori;
    
    public:
        OrientationCost_lArm(
            const Eigen::Matrix<double, 3, 3> lArm_imu_ori)
            :_lArm_imu_ori(lArm_imu_ori){}


    template <typename T> bool operator()(const T* hip_joint, const T* spine_joint, const T* lArm_joint, T* cost_ori){
        Eigen::Matrix<T, 3 ,3> ori = Eigen::Matrix<T, 3, 3>::Identity(3, 3);
        Eigen::Matrix<T, 3, 3> residual;
        Eigen::Quaternion<T> q_res;


        ori *= Eigen::AngleAxis<T>(hip_joint[0], Eigen::Vector3d::UnitZ())
            * Eigen::AngleAxis<T>(hip_joint[1], Eigen::Vector3d::UnitX())
            * Eigen::AngleAxis<T>(hip_joint[2], Eigen::Vector3d::UnitY());

        for(int i = 0; i < 9;i = i + 3)
        {
            ori *= Eigen::AngleAxis<T>(spine_joint[i], Eigen::Vector3d::UnitZ())
                 * Eigen::AngleAxis<T>(spine_joint[i+1], Eigen::Vector3d::UnitX())
                 * Eigen::AngleAxis<T>(spine_joint[i+2], Eigen::Vector3d::UnitY());
        }


        for(int i = 0; i < 9;i = i + 3)
        {
            ori *= Eigen::AngleAxis<T>(lArm_joint[i], Eigen::Vector3d::UnitZ())
                 * Eigen::AngleAxis<T>(lArm_joint[i+1], Eigen::Vector3d::UnitX())
                 * Eigen::AngleAxis<T>(lArm_joint[i+2], Eigen::Vector3d::UnitY());
        }

        residual = ori.inverse() * _lArm_imu_ori;
        q_res = residual;
        q_res.normalize();
        cost_ori[0] = q_res.x() * q_res.x()
                    + q_res.y() * q_res.y()
                    + q_res.z() * q_res.z();
        
        return true;
    }

};


class OrientationCost_hip {
    private:
        const Eigen::Matrix<double, 3, 3> _hip_imu_ori;
    
    public:
        OrientationCost_hip(
            const Eigen::Matrix<double, 3, 3> hip_imu_ori)
            :_hip_imu_ori(hip_imu_ori){}


    template <typename T> bool operator()(const T* hip_joint, T* cost_ori){
        Eigen::Matrix<T, 3 ,3> ori = Eigen::Matrix<T, 3, 3>::Identity(3, 3);
        Eigen::Matrix<T, 3, 3> residual;
        Eigen::Quaternion<T> q_res;


        ori *= Eigen::AngleAxis<T>(hip_joint[0], Eigen::Vector3d::UnitZ())
            * Eigen::AngleAxis<T>(hip_joint[1], Eigen::Vector3d::UnitX())
            * Eigen::AngleAxis<T>(hip_joint[2], Eigen::Vector3d::UnitY());

        residual = ori.inverse() * _hip_imu_ori;
        q_res = residual;
        q_res.normalize();
        cost_ori[0] = q_res.x() * q_res.x()
                    + q_res.y() * q_res.y()
                    + q_res.z() * q_res.z();
        
        return true;
    }

};
/***
int main(int argc, char** argv)
{
    ros::init(argc, argv, "pose_optimizer");
    ros::NodeHandle nh;



    double hip_joint[3];
    double spine_joint[9];
    double rArm_joint[9];
    double lArm_joint[9];




    return 0;
}
***/
