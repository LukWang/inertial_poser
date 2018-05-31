#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>


using namespace ceres;

template <typename T>
void EulerAnglesToRotationMatrixZXY(const T* euler, const int row_stride, T* R);

class OrientationCost_Term {
    private:
        const Eigen::Matrix<double, 3, 3> _rArm_imu_ori;
        const Eigen::Matrix<double, 3, 3> _rArm_offset;

        const Eigen::Matrix<double, 3, 3> _lArm_imu_ori;
        const Eigen::Matrix<double, 3, 3> _lArm_offset;

        const Eigen::Matrix<double, 3, 3> _hip_imu_ori;
        const Eigen::Matrix<double, 3, 3> _hip_offset;

        const Eigen::Matrix<double, 3, 3> _world_to_ref;

    public:
        OrientationCost_Term (
            const Eigen::Matrix<double, 3, 3>& hip_imu_ori,
            const Eigen::Matrix<double, 3, 3>& hip_offset,
            const Eigen::Matrix<double, 3, 3>& rArm_imu_ori,
            const Eigen::Matrix<double, 3, 3>& rArm_offset,
            const Eigen::Matrix<double, 3, 3>& lArm_imu_ori,
            const Eigen::Matrix<double, 3, 3>& lArm_offset,
            const Eigen::Matrix<double, 3, 3>& world_to_ref)
            :_rArm_imu_ori(rArm_imu_ori), _rArm_offset(rArm_offset),
             _lArm_imu_ori(lArm_imu_ori), _lArm_offset(lArm_offset),
             _hip_imu_ori(hip_imu_ori), _hip_offset(hip_offset),
             _world_to_ref(world_to_ref){}

        template <typename T> bool operator()(const T* const hip_joint, const T* const spine_joint, const T* const rArm_joint, const T* const lArm_joint, T* cost_ori) const {
            Eigen::Matrix<T, 3, 3> ite_ori;
            //Eigen::Matrix3d hip_ori;
            Eigen::Matrix<T, 3, 3> spine_ori;
            //Eigen::Matrix3d lArm_ori;
            //Eigen::Matrix3d rArm_ori;
            Eigen::Quaternion<T> q_res;

            T res;

            T rot[9];

            //cost_ori[0] = (T)0;

            EulerAnglesToRotationMatrixZXY(hip_joint, 3, rot);
            Eigen::Map<const Eigen::Matrix<T, 3, 3> > ori(rot);

            ite_ori = _world_to_ref.cast<T>() * ori.transpose();

            //hip_ori = ite_ori;

            q_res = (ite_ori * _hip_offset.cast<T>()).inverse() * _hip_imu_ori.cast<T>();
            q_res.normalize();
            cost_ori[0] = q_res.x() * q_res.x()
                + q_res.y() * q_res.y()
                + q_res.z() * q_res.z();

            for(int i = 0; i < 4; ++i)
            {
                EulerAnglesToRotationMatrixZXY(spine_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3> > ori(rot);
                ite_ori = ite_ori * ori.transpose();
            }

            spine_ori = ite_ori;


            for(int i = 0; i < 3; ++i)
            {
                EulerAnglesToRotationMatrixZXY(rArm_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3> > ori(rot);
                ite_ori = ite_ori * ori.transpose();
            }

            q_res = (ite_ori * _rArm_offset.cast<T>()).inverse() * _rArm_imu_ori.cast<T>();
            q_res.normalize();
            cost_ori[1] = res + q_res.x() * q_res.x()
                + q_res.y() * q_res.y()
                + q_res.z() * q_res.z();

            ite_ori = spine_ori;

            for(int i = 0; i < 3; ++i)
            {
                EulerAnglesToRotationMatrixZXY(lArm_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3> > ori(rot);
                ite_ori = ite_ori * ori.transpose();
            }

            q_res = (ite_ori * _lArm_offset.cast<T>()).inverse() * _lArm_imu_ori.cast<T>();
            q_res.normalize();
            cost_ori[2] = res + q_res.x() * q_res.x()
                  + q_res.y() * q_res.y()
                  + q_res.z() * q_res.z();
            //cost_ori[0] = res * 1.0;

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
