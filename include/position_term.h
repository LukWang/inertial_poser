#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optimizer.h>
//#include <openpose_ros_msgs/PointWithProb.h>

#include <vector>

#include <cmath>


using namespace ceres;
using std::vector;

//void EulerAnglesToRotationMatrixZXY(const T* euler, const int row_stride, T* R);
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



class Position_Term {
    private:

        const Eigen::Matrix<double, 3, 3> _world_to_ref;

        const vector<Eigen::Matrix<double, 3, 1> > _bone_length;

        const vector<KeyPoints> key_points;

        const Eigen::Matrix<double, 3, 3> camera_ori;
        const Eigen::Matrix<double, 3, 1> camera_trans;

        const double pos_weight;

        //const double period = 0.2 * 0.2; //IMU passing ratelHand_imu_acc

    public:
        Position_Term (
            const Eigen::Matrix<double, 3, 3>& world_to_ref,
            const vector<Eigen::Matrix<double, 3, 1> >& bone_length,
            const vector<KeyPoints>& key_points,
            const Eigen::Matrix<double, 3, 3>& camera_ori,
            const Eigen::Matrix<double, 3, 1>& camera_trans,
            const double pos_weight)
            :_world_to_ref(world_to_ref), _bone_length(bone_length), pos_weight(pos_weight),
             key_points(key_points), camera_ori(camera_ori), camera_trans(camera_trans){}

        template <typename T> bool operator()(const T* const hip_trans, const T* const hip_joint, const T* const spine_joint, const T* const rArm_joint, const T* const rElbow_joint, const T* const lArm_joint, const T* const lElbow_joint, T* pos_cost) const {
            Eigen::Matrix<T, 3, 3> ite_ori;
            Eigen::Map<const Eigen::Matrix<T, 3, 1> > hips_trans(hip_trans);
            Eigen::Matrix<T, 3, 1> ite_trans;
            //Eigen::Matrix3d hip_ori;
            Eigen::Matrix<T, 3, 3> spine_ori;
            Eigen::Matrix<T, 3, 1> spine_trans;
            //Eigen::Matrix3d lArm_ori;
            //Eigen::Matrix3d rArm_ori;
            Eigen::Quaternion<T> q_res;

            Eigen::Matrix<T, 3, 1> cam_space_trans;
            T img_pos_x;
            T img_pos_y;
            T x_diff, y_diff;
            T fx = (T)(1068.2054759 / 2);
            T fy = (T)(1068.22398224 / 2);
            T cx = (T)(964.1001882846 / 2);
            T cy = (T)(538.5221553 / 2);
            //printf("I'm here");

            //T acc_weight = (T)0.005;

            T res;

            T rot[9];

            ite_trans = hips_trans;

            //Eigen::MatrixXd pos = ite_trans.cast<double>();

            //position_term_hips
            //if(key_points[0].p > 0.0)
            {
              cam_space_trans = camera_trans.cast<T>() + camera_ori.cast<T>() * ite_trans;
              img_pos_x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
              img_pos_y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
              x_diff = img_pos_x - (T)key_points[0].x;
              y_diff = img_pos_y - (T)key_points[0].y;

              pos_cost[0] = (x_diff * x_diff + y_diff * y_diff) * (T)key_points[0].p  * (T)pos_weight;
            }
            //cost_ori[0] = (T)0;

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



            //for right arm
            {
                EulerAnglesToRotationMatrixZXY(spine_joint + 3 * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[4].cast<T>();
            }

            for(int i = 0; i < 2; ++i)
            {
                EulerAnglesToRotationMatrixZXY(rArm_joint + 3 * i, 3, rot);
                Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
                ite_ori = ite_ori * ori;

                ite_trans += ite_ori * _bone_length[5 + i].cast<T>();

                //if(key_points[1 + i].p > 0.0)
                {
                  cam_space_trans = camera_trans.cast<T>() + camera_ori.cast<T>() * ite_trans;
                  img_pos_x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
                  img_pos_y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
                  x_diff = img_pos_x - (T)key_points[1 + i].x;
                  y_diff = img_pos_y - (T)key_points[1 + i].y;

                  pos_cost[i+1] = (x_diff * x_diff + y_diff * y_diff) * (T)key_points[1 + i].p * (T)pos_weight;
                }
            }
            //for rWrist
            {
              EulerAnglesToRotationMatrixZXY(rElbow_joint, 3, rot);
              Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
              ite_ori = ite_ori * ori;

              ite_trans += ite_ori * _bone_length[7].cast<T>();

              //if(key_points[1 + i].p > 0.0)
              {
                cam_space_trans = camera_trans.cast<T>() + camera_ori.cast<T>() * ite_trans;
                img_pos_x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
                img_pos_y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
                x_diff = img_pos_x - (T)key_points[3].x;
                y_diff = img_pos_y - (T)key_points[3].y;

                pos_cost[3] = (x_diff * x_diff + y_diff * y_diff) * (T)key_points[3].p * (T)pos_weight;
              }
            }




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

                //if(key_points[4 + i].p > 0.0)
                {
                  cam_space_trans = camera_trans.cast<T>() + camera_ori.cast<T>() * ite_trans;
                  img_pos_x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
                  img_pos_y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
                  x_diff = img_pos_x - (T)key_points[4 + i].x;
                  y_diff = img_pos_y - (T)key_points[4 + i].y;

                  pos_cost[i+4] = (x_diff * x_diff + y_diff * y_diff) * (T)key_points[4 + i].p * (T)pos_weight;
                }
            }
            //for lWrist
            {
              EulerAnglesToRotationMatrixZXY(lElbow_joint, 3, rot);
              Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor> > ori(rot);
              ite_ori = ite_ori * ori;

              ite_trans += ite_ori * _bone_length[12].cast<T>();

              //if(key_points[1 + i].p > 0.0)
              {
                cam_space_trans = camera_trans.cast<T>() + camera_ori.cast<T>() * ite_trans;
                img_pos_x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
                img_pos_y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
                x_diff = img_pos_x - (T)key_points[6].x;
                y_diff = img_pos_y - (T)key_points[6].y;

                pos_cost[6] = (x_diff * x_diff + y_diff * y_diff) * (T)key_points[6].p * (T)pos_weight;
              }
            }
            //pos_cost[0] *= (T)pos_weight;



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
