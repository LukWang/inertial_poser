#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>


using namespace ceres;

template <typename T>
void EulerAnglesToRotationMatrixZXY(const T* euler, const int row_stride, T* R);

class OrientationCost_rArm {
    private:
        const Eigen::Matrix<double, 3, 1> _rArm_imu_acc;
        const Eigen::Matrix<double, 3, 1> _lArm_imu_acc;
        
        const Eigen::Matrix<double, 3, 1> spine_offset;
        const Eigen::Matrix<double, 3, 1> spine1_offset;
        const Eigen::Matrix<double, 3, 1> spine2_offset;
        const Eigen::Matrix<double, 3, 1> spine3_offset;
        const Eigen::Matrix<double, 3, 1> lshoulder_offset;
        const Eigen::Matrix<double, 3, 1> upperlArm_offset;
        const Eigen::Matrix<double, 3, 1> forelArm_offset;
    
    public:
        OrientationCost_rArm(
            const Eigen::Matrix<double, 3, 1>& rArm_imu_acc)
            :_rArm_imu_acc(rArm_imu_acc){}


        template <typename T> bool operator()(const T* const hip_joint, const T* const spine_joint, const T* const rArm_joint, T* cost_ori) const {
            Eigen::Matrix<T, 3, 3> base;
            //Eigen::Matrix<T, 3 ,3> ori;
            Eigen::Matrix<T, 3, 3> residual;
            Eigen::Quaternion<T> q_res;

            T rot[9];

            EulerAnglesToRotationMatrixZXY(hip_joint, 3, rot);
            Eigen::Map<const Eigen::Matrix<T, 3, 3> > ori(rot);

            base = ori.transpose();

            for(int i = 0; i < 3; ++i)
            {
                EulerAnglesToRotationMatrixZXY(spine_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3> > ori(rot);
                base = base * ori.transpose();
            }

            for(int i = 0; i < 3; ++i)
            {
                EulerAnglesToRotationMatrixZXY(rArm_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3> > ori(rot);
                base = base * ori.transpose();
            }

            residual = base.inverse() * _rArm_imu_ori.cast<T>();
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


        template <typename T> bool operator()(const T* const hip_joint, const T* const spine_joint, const T* const lArm_joint, T* cost_ori) const {
            Eigen::Matrix<T, 3, 3> base;
            Eigen::Matrix<T, 3, 3> residual;
            Eigen::Quaternion<T> q_res;

            T rot[9];

            EulerAnglesToRotationMatrixZXY(hip_joint, 3, rot);
            Eigen::Map<const Eigen::Matrix<T, 3, 3> > ori(rot);

            base = ori.transpose();

            for(int i = 0; i < 3; ++i)
            {
                EulerAnglesToRotationMatrixZXY(spine_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3> > ori(rot);
                base = base * ori.transpose();
            }

            for(int i = 0; i < 3; ++i)
            {
                EulerAnglesToRotationMatrixZXY(lArm_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3> > ori(rot);
                base = base * ori.transpose();
            }

            residual = base.inverse() * _lArm_imu_ori.cast<T>();
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
        const Eigen::Matrix<double, 3, 3> _hip_imu_acc;
    
    public:
        OrientationCost_hip(
            const Eigen::Matrix<double, 3, 3> hip_imu_acc
            :_hip_imu_acc(hip_imu_acc){}


        template <typename T> bool operator()(const T* const hip_joint, T* cost_ori) const {
            Eigen::Matrix<T, 3, 3> base;
            Eigen::Matrix<T, 3, 3> residual;
            Eigen::Quaternion<T> q_res;

            T rot[9];

            EulerAnglesToRotationMatrixZXY(hip_joint, 3, rot);
            Eigen::Map<const Eigen::Matrix<T, 3, 3> > ori(rot);

            base = ori.transpose();

            residual = base.inverse() * _hip_imu_ori.cast<T>();
            q_res = residual;
            q_res.normalize();
            cost_ori[0] = q_res.x() * q_res.x()
                        + q_res.y() * q_res.y()
                        + q_res.z() * q_res.z();
        
            return true;
        }

};

template <typename T>
inline void EulerAnglesToRotationMatrixZXY(const T* euler, const int row_stride, T* R){
    const T degrees_to_radians(M_PI / 180.0);

    const T yaw(euler[0] * degrees_to_radians);
    const T pitch(euler[1] * degrees_to_radians);
    const T roll(euler[2] * degrees_to_radians);

    const T c1 = cos(yaw);
    const T s1 = sin(yaw);
    const T c2 = cos(pitch);
    const T s2 = sin(pitch);
    const T c3 = cos(roll);
    const T s3 = sin(roll);

  // Rows of the rotation matrix.
    T* R1 = R;
    T* R2 = R1 + row_stride;
    T* R3 = R2 + row_stride;

    R1[0] = c1*c3 - s1*s2*s3;
    R1[1] = -s1*c2;
    R1[2] = c1*s3+s1*s2*c3;

    R2[0] = s1*c3 + c1*s2*s3;
    R2[1] = c1*c2;
    R2[2] = s1*s3 - c1*s2*c3;

    R3[0] = -c2*s3;
    R3[1] = s2;
    R3[2] = c2*c3;
}
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
