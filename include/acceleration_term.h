#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optimizer.h>

#include <vector>

#include <cmath>


using namespace ceres;
using std::vector;


class Acceleration_Term{
private:
    const Eigen::Matrix<double, 3, 1> _hip_imu_acc;

    const Eigen::Matrix<double, 3, 1> _rArm_imu_acc;

    const Eigen::Matrix<double, 3, 1> _lArm_imu_acc;

    const Eigen::Matrix<double, 3, 1> _rHand_imu_acc;

    const Eigen::Matrix<double, 3, 1> _lHand_imu_acc;



    const Eigen::Matrix<double, 3, 3> _world_to_ref;

    const vector<Eigen::Matrix<double, 3, 1> > _bone_length;

    const vector<Eigen::Matrix<double, 3, 1> > _previous_hips_position;
    const vector<Eigen::Matrix<double, 3, 1> > _previous_lArm_position;
    const vector<Eigen::Matrix<double, 3, 1> > _previous_rArm_position;
    const vector<Eigen::Matrix<double, 3, 1> > _previous_lHand_position;
    const vector<Eigen::Matrix<double, 3, 1> > _previous_rHand_position;

    const double acc_weight;

    const double period; //IMU passing ratelHand_imu_acc

    public:
        Acceleration_Term(
            const Eigen::Matrix<double, 3, 1>& hip_imu_acc,
            const vector<Eigen::Matrix<double, 3, 1> >& previous_hips_position,
            const Eigen::Matrix<double, 3, 1>& rArm_imu_acc,
            const vector<Eigen::Matrix<double, 3, 1> >& previous_rArm_position,
            const Eigen::Matrix<double, 3, 1>& lArm_imu_acc,
            const vector<Eigen::Matrix<double, 3, 1> >& previous_lArm_position,
            const Eigen::Matrix<double, 3, 1>& rHand_imu_acc,
            const vector<Eigen::Matrix<double, 3, 1> >& previous_rHand_position,
            const Eigen::Matrix<double, 3, 1>& lHand_imu_acc,
            const vector<Eigen::Matrix<double, 3, 1> >& previous_lHand_position,
            const Eigen::Matrix<double, 3, 3>& world_to_ref,
            const vector<Eigen::Matrix<double, 3, 1> >& bone_length,
            const double acc_weight)
            :_rArm_imu_acc(rArm_imu_acc), _previous_rArm_position(previous_rArm_position),
             _lArm_imu_acc(lArm_imu_acc), _previous_lArm_position(previous_lArm_position),
             _hip_imu_acc(hip_imu_acc), _previous_hips_position(previous_hips_position),
             _lHand_imu_acc(lHand_imu_acc), _previous_lHand_position(previous_lHand_position),
             _rHand_imu_acc(rHand_imu_acc), _previous_rHand_position(previous_rHand_position),
             _world_to_ref(world_to_ref),_bone_length(bone_length), acc_weight(acc_weight), period(0.05 * 0.05){}


        template <typename T> bool operator()(const T* const hip_trans, const T* const hip_joint, const T* const spine_joint, const T* const rArm_joint, const T* const rElbow_joint, const T* const rHand_joint, const T* const lArm_joint, const T* const lElbow_joint, const T* const lHand_joint, T* cost_acc) const {
            Eigen::Matrix<T, 3, 3> ite_ori;
            Eigen::Map<const Eigen::Matrix<T, 3, 1> > hips_trans(hip_trans);
            Eigen::Matrix<T, 3, 1> ite_trans;
            Eigen::Matrix<T, 3, 1> acc_diff;
            //Eigen::Matrix3d hip_ori;
            Eigen::Matrix<T, 3, 3> spine_ori;
            Eigen::Matrix<T, 3, 1> spine_trans;

            Eigen::Matrix<T, 3, 1> solved_acc;

            T rot[9];

            ite_trans = hips_trans;
            //hips_acc
            solved_acc = (ite_trans - (T)2 * _previous_hips_position[1].cast<T>()  + _previous_hips_position[0].cast<T>()) / (T)period;
            acc_diff = solved_acc - _hip_imu_acc.cast<T>();
            cost_acc[0] = acc_diff(0,0) * (T)acc_weight;
            cost_acc[1] = acc_diff(1,0) * (T)acc_weight;
            cost_acc[2] = acc_diff(2,0) * (T)acc_weight;



            EulerAnglesToRotationMatrixZXY(hip_joint, 3, rot);
            Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);

            ite_ori = _world_to_ref.cast<T>() * ori;

            ite_trans += ite_ori * _bone_length[0].cast<T>();

            for(int i = 0; i < 3; ++i)
            {
                EulerAnglesToRotationMatrixZXY(spine_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[1+i].cast<T>();
            }

            spine_ori = ite_ori;
            spine_trans = ite_trans;

            //for right Elbow
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
            acc_diff = solved_acc - _rArm_imu_acc.cast<T>();
            cost_acc[3] = acc_diff(0,0) * (T)acc_weight;
            cost_acc[4] = acc_diff(1,0) * (T)acc_weight;
            cost_acc[5] = acc_diff(2,0) * (T)acc_weight;


            //for right Hand
            {
                EulerAnglesToRotationMatrixZXY(rElbow_joint, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[7].cast<T>();
            }
            {
                EulerAnglesToRotationMatrixZXY(rHand_joint, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[8].cast<T>();
            }
            //rHand_acc cost
            solved_acc = (ite_trans - (T)2 * _previous_rHand_position[1].cast<T>()  + _previous_rHand_position[0].cast<T>()) / (T)period;
            //cost_imu[7] = (solved_acc - _rHand_imu_acc.cast<T>()).norm() * (T)acc_weight;
            acc_diff = solved_acc - _rHand_imu_acc.cast<T>();
            cost_acc[6] = acc_diff(0,0) * (T)acc_weight;
            cost_acc[7] = acc_diff(1,0) * (T)acc_weight;
            cost_acc[8] = acc_diff(2,0) * (T)acc_weight;

            //for left Elbow
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
            cost_acc[9] = acc_diff(0,0) * (T)acc_weight;
            cost_acc[10] = acc_diff(1,0) * (T)acc_weight;
            cost_acc[11] = acc_diff(2,0) * (T)acc_weight;

            //for left Hand
            {
                EulerAnglesToRotationMatrixZXY(lElbow_joint, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[12].cast<T>();
            }
            {
                EulerAnglesToRotationMatrixZXY(lHand_joint, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[13].cast<T>();
            }
            solved_acc = (ite_trans - (T)2 * _previous_lHand_position[1].cast<T>()  + _previous_lHand_position[0].cast<T>()) / (T)period;
            //cost_imu[9] = (solved_acc - _lHand_imu_acc.cast<T>()).norm() * (T)acc_weight;
            acc_diff = solved_acc - _lHand_imu_acc.cast<T>();
            cost_acc[12] = acc_diff(0,0) * (T)acc_weight;
            cost_acc[13] = acc_diff(1,0) * (T)acc_weight;
            cost_acc[14] = acc_diff(2,0) * (T)acc_weight;

            return true;
        }
};
