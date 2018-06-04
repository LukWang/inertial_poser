#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vector>
#include <queue>

#include <optimizer.h>

#include <cmath>


using namespace ceres;
using std::vector;
using std::queue;


/*********************************************/
//cost_imu contains 10 residuals
//0: hips_ori
//1: rArm_ori
//2: rHand_ori
//3: lArm_ori
//4: lHand_ori
//5: hips_acc
//6: rArm_acc
//7: rHand_acc
//8: lArm_acc
//9: lHand_acc
/*********************************************/

class Imu_Term {
    private:
        const Eigen::Matrix<double, 3, 3> _rArm_imu_ori;
        const Eigen::Matrix<double, 3, 1> _rArm_imu_acc;
        const Eigen::Matrix<double, 3, 3> _rArm_offset;

        const Eigen::Matrix<double, 3, 3> _lArm_imu_ori;
        const Eigen::Matrix<double, 3, 1> _lArm_imu_acc;
        const Eigen::Matrix<double, 3, 3> _lArm_offset;

        const Eigen::Matrix<double, 3, 3> _hip_imu_ori;
        const Eigen::Matrix<double, 3, 1> _hip_imu_acc;
        const Eigen::Matrix<double, 3, 3> _hip_offset;

        const Eigen::Matrix<double, 3, 3> _rHand_imu_ori;
        const Eigen::Matrix<double, 3, 1> _rHand_imu_acc;
        const Eigen::Matrix<double, 3, 3> _rHand_offset;

        const Eigen::Matrix<double, 3, 3> _lHand_imu_ori;
        const Eigen::Matrix<double, 3, 1> _lHand_imu_acc;
        const Eigen::Matrix<double, 3, 3> _lHand_offset;

        const Eigen::Matrix<double, 3, 3> _world_to_ref;

        const vector<Eigen::Matrix<double, 3, 1> > _bone_length;

        const vector<Eigen::Matrix<double, 3, 1> > _previous_hips_position;
        const vector<Eigen::Matrix<double, 3, 1> > _previous_lArm_position;
        const vector<Eigen::Matrix<double, 3, 1> > _previous_rArm_position;
        const vector<Eigen::Matrix<double, 3, 1> > _previous_lHand_position;
        const vector<Eigen::Matrix<double, 3, 1> > _previous_rHand_position;

        const double ori_weight, acc_weight;

        const double period = 0.05 * 0.05; //IMU passing ratelHand_imu_acc

    public:
        Imu_Term (
            const Eigen::Matrix<double, 3, 3>& hip_imu_ori,
            const Eigen::Matrix<double, 3, 1>& hip_imu_acc,
            const Eigen::Matrix<double, 3, 3>& hip_offset,
            const vector<Eigen::Matrix<double, 3, 1> >& previous_hips_position,
            const Eigen::Matrix<double, 3, 3>& rArm_imu_ori,
            const Eigen::Matrix<double, 3, 1>& rArm_imu_acc,
            const Eigen::Matrix<double, 3, 3>& rArm_offset,
            const vector<Eigen::Matrix<double, 3, 1> >& previous_rArm_position,
            const Eigen::Matrix<double, 3, 3>& lArm_imu_ori,
            const Eigen::Matrix<double, 3, 1>& lArm_imu_acc,
            const Eigen::Matrix<double, 3, 3>& lArm_offset,
            const vector<Eigen::Matrix<double, 3, 1> >& previous_lArm_position,
            const Eigen::Matrix<double, 3, 3>& rHand_imu_ori,
            const Eigen::Matrix<double, 3, 1>& rHand_imu_acc,
            const Eigen::Matrix<double, 3, 3>& rHand_offset,
            const vector<Eigen::Matrix<double, 3, 1> >& previous_rHand_position,
            const Eigen::Matrix<double, 3, 3>& lHand_imu_ori,
            const Eigen::Matrix<double, 3, 1>& lHand_imu_acc,
            const Eigen::Matrix<double, 3, 3>& lHand_offset,
            const vector<Eigen::Matrix<double, 3, 1> >& previous_lHand_position,
            const Eigen::Matrix<double, 3, 3>& world_to_ref,
            const vector<Eigen::Matrix<double, 3, 1> >& bone_length,
            const double ori_weight, const double acc_weight)
            :_rArm_imu_ori(rArm_imu_ori), _rArm_imu_acc(rArm_imu_acc), _rArm_offset(rArm_offset), _previous_rArm_position(previous_rArm_position),
             _lArm_imu_ori(lArm_imu_ori), _lArm_imu_acc(lArm_imu_acc), _lArm_offset(lArm_offset), _previous_lArm_position(previous_lArm_position),
             _hip_imu_ori(hip_imu_ori), _hip_imu_acc(hip_imu_acc), _hip_offset(hip_offset), _previous_hips_position(previous_hips_position),
             _lHand_imu_ori(lHand_imu_ori), _lHand_imu_acc(lHand_imu_acc), _lHand_offset(lHand_offset), _previous_lHand_position(previous_lHand_position),
             _rHand_imu_ori(rHand_imu_ori), _rHand_imu_acc(rHand_imu_acc), _rHand_offset(rHand_offset), _previous_rHand_position(previous_rHand_position),
             _world_to_ref(world_to_ref), _bone_length(bone_length), ori_weight(ori_weight), acc_weight(acc_weight){}

        template <typename T> bool operator()(const T* const hip_trans, const T* const hip_joint, const T* const spine_joint, const T* const rArm_joint, const T* const lArm_joint, T* cost_imu) const {
            Eigen::Matrix<T, 3, 3> ite_ori;
            Eigen::Map<const Eigen::Matrix<T, 3, 1> > hips_trans(hip_trans);
            Eigen::Matrix<T, 3, 1> ite_trans;
            Eigen::Matrix<T, 3, 1> acc_diff;
            //Eigen::Matrix3d hip_ori;
            Eigen::Matrix<T, 3, 3> spine_ori;
            Eigen::Matrix<T, 3, 1> spine_trans;
            //Eigen::Matrix3d lArm_ori;
            //Eigen::Matrix3d rArm_ori;
            Eigen::Quaternion<T> q_res;

            //printf("I'm here");

            //T acc_weight = (T)0.005;

            T res;

            T rot[9];

            Eigen::Matrix<T, 3, 1> solved_acc;

            ite_trans = hips_trans;

            //Eigen::MatrixXd pos = ite_trans.cast<double>();

            //acc_term_hips

            solved_acc = (ite_trans - (T)2 * _previous_hips_position[1].cast<T>()  + _previous_hips_position[0].cast<T>()) / (T)period;
            acc_diff = solved_acc - _hip_imu_acc.cast<T>();
            cost_imu[5] = (acc_diff(0,0) * acc_diff(0,0) +
                          acc_diff(1,0) * acc_diff(1,0) +
                          acc_diff(2,0) * acc_diff(2,0)) * (T)acc_weight;

            //cost_ori[0] = (T)0;

            EulerAnglesToRotationMatrixZXY(hip_joint, 3, rot);
            Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);

            ite_ori = _world_to_ref.cast<T>() * ori;

            ite_trans += ite_ori * _bone_length[0].cast<T>();

            //hip_ori = ite_ori;

            q_res = (ite_ori * _hip_offset.cast<T>()).inverse() * _hip_imu_ori.cast<T>();
            q_res.normalize();
            cost_imu[0] = q_res.x() * q_res.x()
                + q_res.y() * q_res.y()
                + q_res.z() * q_res.z();
            cost_imu[0] *= (T)ori_weight;

            for(int i = 0; i < 3; ++i)
            {
                EulerAnglesToRotationMatrixZXY(spine_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[1+i].cast<T>();
            }

            spine_ori = ite_ori;
            spine_trans = ite_trans;



            //for right arm
            {
                EulerAnglesToRotationMatrixZXY(spine_joint + 3 * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[4].cast<T>();
            }


            for(int i = 0; i < 2; ++i)
            {
                EulerAnglesToRotationMatrixZXY(rArm_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[5+i].cast<T>();
            }
            //rArm_acc cost
            solved_acc = (ite_trans - (T)2 * _previous_rArm_position[1].cast<T>()  + _previous_rArm_position[0].cast<T>()) / (T)period;
            //cost_imu[6] = (solved_acc - _rArm_imu_acc.cast<T>()).norm() * (T)acc_weight;
            acc_diff = solved_acc - _rArm_imu_acc.cast<T>();
            cost_imu[6] = (acc_diff(0,0) * acc_diff(0,0) +
                          acc_diff(1,0) * acc_diff(1,0) +
                          acc_diff(2,0) * acc_diff(2,0)) * (T)acc_weight;
            //rArm_ori cost
            q_res = (ite_ori * _rArm_offset.cast<T>()).inverse() * _rArm_imu_ori.cast<T>();
            q_res.normalize();
            cost_imu[1] = q_res.x() * q_res.x()
                + q_res.y() * q_res.y()
                + q_res.z() * q_res.z();
            cost_imu[1] *= (T)ori_weight;

            for(int i = 2; i < 4; ++i)
            {
                EulerAnglesToRotationMatrixZXY(rArm_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[5+i].cast<T>();
            }

            //rHand_acc cost
            solved_acc = (ite_trans - (T)2 * _previous_rHand_position[1].cast<T>()  + _previous_rHand_position[0].cast<T>()) / (T)period;
            //cost_imu[7] = (solved_acc - _rHand_imu_acc.cast<T>()).norm() * (T)acc_weight;
            acc_diff = solved_acc - _rHand_imu_acc.cast<T>();
            cost_imu[7] = (acc_diff(0,0) * acc_diff(0,0) +
                          acc_diff(1,0) * acc_diff(1,0) +
                          acc_diff(2,0) * acc_diff(2,0)) * (T)acc_weight;
            //rHand_ori cost
            q_res = (ite_ori * _rHand_offset.cast<T>()).inverse() * _rHand_imu_ori.cast<T>();
            q_res.normalize();
            cost_imu[2] = q_res.x() * q_res.x()
                + q_res.y() * q_res.y()
                + q_res.z() * q_res.z();
            cost_imu[2] *= (T)ori_weight;




            //for left arm
            ite_trans = spine_trans;
            ite_ori = spine_ori;
            {
                EulerAnglesToRotationMatrixZXY(spine_joint + 3 * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[9].cast<T>();
            }

            for(int i = 0; i < 2; ++i)
            {
                EulerAnglesToRotationMatrixZXY(lArm_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[10+i].cast<T>();
            }
            //lArm_acc cost
            solved_acc = (ite_trans - (T)2 * _previous_lArm_position[1].cast<T>()  + _previous_lArm_position[0].cast<T>()) / (T)period;
            //cost_imu[8] = (solved_acc - _lArm_imu_acc.cast<T>()).norm() * (T)acc_weight;
            acc_diff = solved_acc - _lArm_imu_acc.cast<T>();
            cost_imu[8] = (acc_diff(0,0) * acc_diff(0,0) +
                          acc_diff(1,0) * acc_diff(1,0) +
                          acc_diff(2,0) * acc_diff(2,0)) * (T)acc_weight;
            //lArm_oricost
            q_res = (ite_ori * _lArm_offset.cast<T>()).inverse() * _lArm_imu_ori.cast<T>();
            q_res.normalize();
            cost_imu[3] = q_res.x() * q_res.x()
                  + q_res.y() * q_res.y()
                  + q_res.z() * q_res.z();
            cost_imu[3] *= (T)ori_weight;

            for(int i = 2; i < 4; ++i)
            {
                EulerAnglesToRotationMatrixZXY(lArm_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[10+i].cast<T>();
            }

            solved_acc = (ite_trans - (T)2 * _previous_lHand_position[1].cast<T>()  + _previous_lHand_position[0].cast<T>()) / (T)period;
            //cost_imu[9] = (solved_acc - _lHand_imu_acc.cast<T>()).norm() * (T)acc_weight;
            acc_diff = solved_acc - _lHand_imu_acc.cast<T>();
            cost_imu[9] = (acc_diff(0,0) * acc_diff(0,0) +
                          acc_diff(1,0) * acc_diff(1,0) +
                          acc_diff(2,0) * acc_diff(2,0)) * (T)acc_weight;

            q_res = (ite_ori * _lHand_offset.cast<T>()).inverse() * _lHand_imu_ori.cast<T>();
            q_res.normalize();
            cost_imu[4] = q_res.x() * q_res.x()
                  + q_res.y() * q_res.y()
                  + q_res.z() * q_res.z();
            cost_imu[4] *= (T)ori_weight;


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

lHand_acc


    return 0;
}
***/
