#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

#include <optimizer.h>

using namespace ceres;


class OrientationCost_Term {
    private:
        const Eigen::Matrix<double, 3, 3> _rArm_imu_ori;
        const Eigen::Matrix<double, 3, 3> _rArm_offset;

        const Eigen::Matrix<double, 3, 3> _rHand_imu_ori;
        const Eigen::Matrix<double, 3, 3> _rHand_offset;

        const Eigen::Matrix<double, 3, 3> _lArm_imu_ori;
        const Eigen::Matrix<double, 3, 3> _lArm_offset;

        const Eigen::Matrix<double, 3, 3> _lHand_imu_ori;
        const Eigen::Matrix<double, 3, 3> _lHand_offset;

        const Eigen::Matrix<double, 3, 3> _hip_imu_ori;
        const Eigen::Matrix<double, 3, 3> _hip_offset;

        const Eigen::Matrix<double, 3, 3> _world_to_ref;

        const double ori_weight;

    public:
        OrientationCost_Term (
            const Eigen::Matrix<double, 3, 3>& hip_imu_ori,
            const Eigen::Matrix<double, 3, 3>& hip_offset,
            const Eigen::Matrix<double, 3, 3>& rArm_imu_ori,
            const Eigen::Matrix<double, 3, 3>& rArm_offset,
            const Eigen::Matrix<double, 3, 3>& lArm_imu_ori,
            const Eigen::Matrix<double, 3, 3>& lArm_offset,
            const Eigen::Matrix<double, 3, 3>& rHand_imu_ori,
            const Eigen::Matrix<double, 3, 3>& rHand_offset,
            const Eigen::Matrix<double, 3, 3>& lHand_imu_ori,
            const Eigen::Matrix<double, 3, 3>& lHand_offset,
            const Eigen::Matrix<double, 3, 3>& world_to_ref,
            const double ori_weight)
            :_rArm_imu_ori(rArm_imu_ori), _rArm_offset(rArm_offset),
             _rHand_imu_ori(rHand_imu_ori), _rHand_offset(rHand_offset),
             _lArm_imu_ori(lArm_imu_ori), _lArm_offset(lArm_offset),
             _lHand_imu_ori(lHand_imu_ori), _lHand_offset(lHand_offset),
             _hip_imu_ori(hip_imu_ori), _hip_offset(hip_offset),
             _world_to_ref(world_to_ref), ori_weight(ori_weight){}

        template <typename T> bool operator()(const T* const hip_joint, const T* const spine_joint, const T* const rArm_joint, const T* const rElbow_joint, const T* const rHand_joint, const T* const lArm_joint, const T* const lElbow_joint, const T* const lHand_joint, T* cost_ori) const {
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
            Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);

            ite_ori = _world_to_ref.cast<T>() * ori;

            //hip_ori = ite_ori;

            q_res = (ite_ori * _hip_offset.cast<T>()).inverse() * _hip_imu_ori.cast<T>();
            q_res.normalize();
            cost_ori[0] = q_res.x() * q_res.x()
                + q_res.y() * q_res.y()
                + q_res.z() * q_res.z();
            cost_ori[0] *= (T)ori_weight;


            for(int i = 0; i < 4; ++i)
            {
                EulerAnglesToRotationMatrixZXY(spine_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;
            }

            spine_ori = ite_ori;

            //rArm
            for(int i = 0; i < 2; ++i)
            {
                EulerAnglesToRotationMatrixZXY(rArm_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;
            }

            q_res = (ite_ori * _rArm_offset.cast<T>()).inverse() * _rArm_imu_ori.cast<T>();
            q_res.normalize();
            cost_ori[1] = res + q_res.x() * q_res.x()
                + q_res.y() * q_res.y()
                + q_res.z() * q_res.z();
            cost_ori[1] *= (T)ori_weight;


            //rHand
            //for(int i = 0; i < 2; ++i)
            {
                EulerAnglesToRotationMatrixZXY(rElbow_joint, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;
            }
            {
                EulerAnglesToRotationMatrixZXY(rHand_joint, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;
            }

            q_res = (ite_ori * _rHand_offset.cast<T>()).inverse() * _rHand_imu_ori.cast<T>();
            q_res.normalize();
            cost_ori[2] = res + q_res.x() * q_res.x()
                + q_res.y() * q_res.y()
                + q_res.z() * q_res.z();
            cost_ori[2] *= (T)ori_weight;


            //lArm
            ite_ori = spine_ori;


            for(int i = 0; i < 2; ++i)
            {
                EulerAnglesToRotationMatrixZXY(lArm_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;
            }

            q_res = (ite_ori * _lArm_offset.cast<T>()).inverse() * _lArm_imu_ori.cast<T>();
            q_res.normalize();
            cost_ori[3] = res + q_res.x() * q_res.x()
                  + q_res.y() * q_res.y()
                  + q_res.z() * q_res.z();
            cost_ori[3] *= (T)ori_weight;
            //cost_ori[0] = res * 1.0;
            //lHand
            {
                EulerAnglesToRotationMatrixZXY(lElbow_joint, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;
            }
            {
                EulerAnglesToRotationMatrixZXY(lHand_joint, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;
            }

            q_res = (ite_ori * _lHand_offset.cast<T>()).inverse() * _lHand_imu_ori.cast<T>();
            q_res.normalize();
            cost_ori[4] = res + q_res.x() * q_res.x()
                  + q_res.y() * q_res.y()
                  + q_res.z() * q_res.z();
            cost_ori[4] *= (T)ori_weight;

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
