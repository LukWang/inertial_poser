#include <optimizer.h>

#include "ceres/ceres.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>

#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>

#include <openpose_ros_msgs/OpenPoseHumanList.h>
#include <openpose_ros_msgs/PointWithProb.h>
//#include <orientation_term.h>
//#include <orientation_term.h>
#include <imu_term.h>
#include <position_term.h>
#include <pose_prior_term.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <cmath>
#include <ctime>

#include <pthread.h>

using std::ifstream;
using std::string;
using std::vector;
using std::queue;
using namespace ceres;

class Optimizer{
    public:
        Optimizer(const ros::NodeHandle &priv_nh = ros::NodeHandle("~")):priv_nh(priv_nh){
            char Proj_dir[] = "/home/luk/PCA/Proj";
            char miu_dir[] = "/home/luk/PCA/mean";
            char eigen_dir[] = "/home/luk/PCA/eigen";
            char meandata_dir[] = "/home/luk/PCA/meandata";
            char lower_bound_dir[] = "/home/luk/PCA/lower_bound";
            char upper_bound_dir[] = "/home/luk/PCA/upper_bound";
            char bone_data[] = "/home/luk/PCA/bjh_body";
            ifstream in(Proj_dir);
            {
              double data[240];
              for (int i = 0; i < 30; ++i)
                for(int j = 0; j < 8; ++j)
                {
                    in >> data[ i*8 + j];
                }
              in.close();
              Eigen::Map<const Eigen::Matrix<double, 30, 8>> proj(data);
              PCA_proj = proj;
            }

            in.open(miu_dir);
            {
              double data[30];
              for(int j = 0; j < 30; ++j)
              {
                  in >> data[j];
              }
              in.close();
              Eigen::Map<const Eigen::Matrix<double, 30, 1>> miu(data);
              PCA_miu = miu;
            }

            in.open(eigen_dir);
            {
              double data[30];
              for(int j = 0; j < 30; ++j)
              {
                in >> data[j];
                data[j] = sqrt(data[j]);
                data[j] = 1.0 / data[j];
              }
              in.close();
              Eigen::Map<const Eigen::Matrix<double, 8, 1>> eigen(data);
              PCA_eigenvalue= eigen;
            }

            in.open(upper_bound_dir);
            {
              double data[36];
              for(int j = 0; j < 12; ++j)
              {
                in >> data[j*3+1];
                in >> data[j*3+2];
                in >> data[j*3];
              }
              for(int j = 0; j < 36; ++j)
              {
                joint_upper_bound.push_back(data[j]);
                //std::cout << data[j] << std::endl;
              }
              in.close();
            }

            in.open(lower_bound_dir);
            {
              double data[36];
              for(int j = 0; j < 12; ++j)
              {
                in >> data[j*3+1];
                in >> data[j*3+2];
                in >> data[j*3];
              }
              for(int j = 0; j < 36; ++j)
              {
                joint_lower_bound.push_back(data[j]);
                //cout << data[j] << endl;
              }
              in.close();
            }




            in.open("/home/luk/Public/Total Capture/S3/acting3_BlenderZXY_YmZ.bvh");
            int joint_count = 0;
            while(1)
            {
                char line[256];
                string::size_type idx;
                string motion = "MOTION";
                string channel = "CHANNELS";
                if(in.getline(line, 256).good())
                {
                    string str(line);
                    idx = str.find(channel);
                    if(idx != string::npos)
                    {
                        joint_count++;
                    }
                    idx = str.find(motion);
                    if(idx != string::npos)
                    {
                        in.getline(line, 256);
                        in.getline(line, 256);
                        break;
                    }
                }
            }

            //cout << "joint count: " << joint_count << endl;
            double bvhdata;
            vector<double> init_joints;
            for (int i = 0; i < joint_count * 6; i++)
            {
                int joint_index = i/6;
                int data_index = i%6;
                in >> bvhdata;
                if(!(data_index < 3))
                if(!(joint_index == 5 || joint_index == 6 || joint_index == 11 || joint_index == 12 || joint_index >=17))
                {
                    init_joints.push_back(bvhdata);
                    //cout << data << endl;
                }
            }
            in.close();

            for(int i = 0; i < 3; ++i)
              hips_joint[i] = init_joints[i];
            for(int i = 0; i < 12; ++i)
              spine_joint[i] = init_joints[3 + i];
            for(int i = 0; i < 12; ++i)
              rArm_joint[i] = init_joints[15 + i];
            for(int i = 0; i < 12; ++i)
              lArm_joint[i] = init_joints[27 + i];

            hips_trans[0] = 0;
            hips_trans[1] = 0.8;
            hips_trans[2] = 0;




            vector<string> joint_names;
            joint_names.push_back("ref_to_hip_z");
            joint_names.push_back("hip_z_to_hip_x");
            joint_names.push_back("hip_x_to_hip_y");

            joint_names.push_back("spine_to_spine1_z");
            joint_names.push_back("spine1_z_to_spine1_x");
            joint_names.push_back("spine1_x_to_spine1_y");
            joint_names.push_back("spine1_to_spine2_z");
            joint_names.push_back("spine2_z_to_spine2_x");
            joint_names.push_back("spine2_x_to_spine2_y");
            joint_names.push_back("spine2_to_spine3_z");
            joint_names.push_back("spine3_z_to_spine3_x");
            joint_names.push_back("spine3_x_to_spine3_y");
            joint_names.push_back("spine3_to_neck_z");
            joint_names.push_back("neck_z_to_neck_x");
            joint_names.push_back("neck_x_to_neck_y");

            joint_names.push_back("rShoulder_to_rArm_z");
            joint_names.push_back("rArm_z_to_rArm_x");
            joint_names.push_back("rArm_x_to_rArm_y");
            joint_names.push_back("rArm_to_rForeArm_z");
            joint_names.push_back("rForeArm_z_to_rForeArm_x");
            joint_names.push_back("rForeArm_x_to_rForeArm_y");
            joint_names.push_back("rForeArm_to_rWrist_z");
            joint_names.push_back("rWrist_z_to_rWrist_x");
            joint_names.push_back("rWrist_x_to_rWrist_y");
            joint_names.push_back("rWrist_to_rHand_z");
            joint_names.push_back("rHand_z_to_rHand_x");
            joint_names.push_back("rHand_x_to_rHand_y");

            joint_names.push_back("lShoulder_to_lArm_z");
            joint_names.push_back("lArm_z_to_lArm_x");
            joint_names.push_back("lArm_x_to_lArm_y");
            joint_names.push_back("lArm_to_lForeArm_z");
            joint_names.push_back("lForeArm_z_to_lForeArm_x");
            joint_names.push_back("lForeArm_x_to_lForeArm_y");
            joint_names.push_back("lForeArm_to_lWrist_z");
            joint_names.push_back("lWrist_z_to_lWrist_x");
            joint_names.push_back("lWrist_x_to_lWrist_y");
            joint_names.push_back("lWrist_to_lHand_z");
            joint_names.push_back("lHand_z_to_lHand_x");
            joint_names.push_back("lHand_x_to_lHand_y");

            joint_msg.name = joint_names;
            const double degrees_to_radians(M_PI / 180.0);
            for(int i = 0; i < init_joints.size(); ++i)
            {
              joint_msg.position.push_back(init_joints[i] * degrees_to_radians);
            }

            double world_to_ref_euler[3] = {-90.0, 90.0, 0.0};
            double rot[9];
            EulerAnglesToRotationMatrixZXY(world_to_ref_euler, 3, rot);
            Eigen::Map<const Eigen::Matrix<double, 3, 3> > ori(rot);
            world_to_ref = ori.transpose();


            in.open(bone_data);
            {

              for(int i = 0; i < 14; ++i)
              {
                double data[3];
                Eigen::Matrix<double, 3, 1> bone;
                in >> data[0];
                in >> data[1];
                in >> data[2];
                bone << data[0], data[1], data[2];
                bone_length.push_back(bone);
                std::cout << bone_length[i] << std::endl;
              }
              in.close();
            }

/***
            Eigen::Matrix<double, 3, 1> spine_length;
            Eigen::Matrix<double, 3, 1> spine1_length;
            Eigen::Matrix<double, 3, 1> spine2_length;
            Eigen::Matrix<double, 3, 1> spine3_length;
            Eigen::Matrix<double, 3, 1> left_chest_length;
            Eigen::Matrix<double, 3, 1> lshoulder_length;
            Eigen::Matrix<double, 3, 1> upperlArm_length;
            Eigen::Matrix<double, 3, 1> forelArm_length;
            Eigen::Matrix<double, 3, 1> lHand_length;
            Eigen::Matrix<double, 3, 1> right_chest_length;
            Eigen::Matrix<double, 3, 1> rshoulder_length;
            Eigen::Matrix<double, 3, 1> upperrArm_length;
            Eigen::Matrix<double, 3, 1> forerArm_length;
            Eigen::Matrix<double, 3, 1> rHand_length;

            spine_length << 0.0,0.046171358,-0.06925691;
            spine1_length << 0.0,0.01603502,-0.09093962;
            spine2_length << 0.0,0.008048244,-0.09199118;
            spine3_length << 0.0,0.0,-0.092342462;

            right_chest_length << 0.029176472,-0.048627538,-0.157553406;
            rshoulder_length << 0.144910048,0.0,0.0;
            upperrArm_length << 0.288867596,0.0,5.08e-08;
            forerArm_length << 0.2196027866,0.0,0.0;
            rHand_length << 0.0658931626,0.0,0.0;

            left_chest_length << -0.0291764466,-0.048627538,-0.157553406;
            lshoulder_length << -0.144910048,0.0,0.0;
            upperlArm_length << -0.288867596,0.0,-5.08e-08;
            forelArm_length <<  -0.2196027866 ,0.0,0.0;
            lHand_length << -0.0658931626,0.0,0.0;

            bone_length.push_back(spine_length);
            bone_length.push_back(spine1_length);
            bone_length.push_back(spine2_length);
            bone_length.push_back(spine3_length);

            bone_length.push_back(right_chest_length);
            bone_length.push_back(rshoulder_length);
            bone_length.push_back(upperrArm_length);
            bone_length.push_back(forerArm_length);
            bone_length.push_back(rHand_length);

            bone_length.push_back(left_chest_length);
            bone_length.push_back(lshoulder_length);
            bone_length.push_back(upperlArm_length);
            bone_length.push_back(forelArm_length);
            bone_length.push_back(lHand_length);
***/
            hips_imu_sub = nh.subscribe("/imu_1/imu_stream", 1, &Optimizer::hips_imu_callback, this);
            lArm_imu_sub = nh.subscribe("/imu_2/imu_stream", 1, &Optimizer::lArm_imu_callback, this);
            rArm_imu_sub = nh.subscribe("/imu_3/imu_stream", 1, &Optimizer::rArm_imu_callback, this);
            lHand_imu_sub = nh.subscribe("/imu_4/imu_stream", 1, &Optimizer::lHand_imu_callback, this);
            rHand_imu_sub = nh.subscribe("/imu_5/imu_stream", 1, &Optimizer::rHand_imu_callback, this);

            human_keypoints_sub = nh.subscribe("/openpose_ros/human_list", 1, &Optimizer::human_keypoints_callback, this);

            joint_publisher = nh.advertise<sensor_msgs::JointState>("/arm_ns/joint_states", 1);
            image_pose_publisher = nh.advertise<geometry_msgs::PoseArray>("/inertial_poser/pose2d", 1);

            solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            //solver_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
            //solver_options.num_threads = 4;
            solver_options.max_solver_time_in_seconds = 0.035;
        }


        void hips_imu_callback(sensor_msgs::Imu imu_msg)
        {
            Eigen::Quaterniond q(imu_msg.orientation.w,
                                 imu_msg.orientation.x,
                                 imu_msg.orientation.y,
                                 imu_msg.orientation.z);
            hips_imu_ori = q.normalized().toRotationMatrix();

            hips_imu_acc_pre = hips_imu_acc;
            hips_imu_acc << imu_msg.linear_acceleration.x,
                           imu_msg.linear_acceleration.y,
                           imu_msg.linear_acceleration.z;

            hips_imu_recved = true;
            ROS_INFO("hips_IMU_recved:ori[%.4lf,%.4lf,%.4lf,%.4lf]  acc[%.4lf,%.4lf,%.4lf]", q.x(), q.y(), q.z(), q.w(), hips_imu_acc(0), hips_imu_acc(1), hips_imu_acc(2));

        }

        void lArm_imu_callback(sensor_msgs::Imu imu_msg)
        {
            Eigen::Quaterniond q(imu_msg.orientation.w,
                                 imu_msg.orientation.x,
                                 imu_msg.orientation.y,
                                 imu_msg.orientation.z);
            lArm_imu_ori = q.normalized().toRotationMatrix();

            lArm_imu_acc_pre = lArm_imu_acc;
            lArm_imu_acc << imu_msg.linear_acceleration.x,
                           imu_msg.linear_acceleration.y,
                           imu_msg.linear_acceleration.z;

            lArm_imu_recved = true;
            ROS_INFO("lArm_IMU_recved:ori[%.4lf,%.4lf,%.4lf,%.4lf]  acc[%.4lf,%.4lf,%.4lf]", q.x(), q.y(), q.z(), q.w(), lArm_imu_acc(0), lArm_imu_acc(1), lArm_imu_acc(2));
        }

        void rArm_imu_callback(sensor_msgs::Imu imu_msg)
        {
            Eigen::Quaterniond q(imu_msg.orientation.w,
                                 imu_msg.orientation.x,
                                 imu_msg.orientation.y,
                                 imu_msg.orientation.z);
            rArm_imu_ori = q.normalized().toRotationMatrix();

            rArm_imu_acc_pre = rArm_imu_acc;
            rArm_imu_acc << imu_msg.linear_acceleration.x,
                            imu_msg.linear_acceleration.y,
                            imu_msg.linear_acceleration.z;

            rArm_imu_recved = true;
            ROS_INFO("rArm_IMU_recved");
        }

        void rHand_imu_callback(sensor_msgs::Imu imu_msg)
        {
            Eigen::Quaterniond q(imu_msg.orientation.w,
                                 imu_msg.orientation.x,
                                 imu_msg.orientation.y,
                                 imu_msg.orientation.z);
            rHand_imu_ori = q.normalized().toRotationMatrix();

            rHand_imu_acc_pre = rHand_imu_acc;
            rHand_imu_acc << imu_msg.linear_acceleration.x,
                            imu_msg.linear_acceleration.y,
                            imu_msg.linear_acceleration.z;

            rHand_imu_recved = true;
            ROS_INFO("rHand_IMU_recved");
        }

        void lHand_imu_callback(sensor_msgs::Imu imu_msg)
        {
            Eigen::Quaterniond q(imu_msg.orientation.w,
                                 imu_msg.orientation.x,
                                 imu_msg.orientation.y,
                                 imu_msg.orientation.z);
            lHand_imu_ori = q.normalized().toRotationMatrix();

            lHand_imu_acc_pre = lHand_imu_acc;
            lHand_imu_acc << imu_msg.linear_acceleration.x,
                            imu_msg.linear_acceleration.y,
                            imu_msg.linear_acceleration.z;

            lHand_imu_recved = true;
            ROS_INFO("lHand_IMU_recved");
        }

        void human_keypoints_callback(openpose_ros_msgs::OpenPoseHumanList keypoints)
        {
          int person_num = keypoints.num_humans;
          vector<double> probs;
          if(person_num > 0){
              for(int person = 0;person < person_num; ++person)
              {
                auto body_keypoints = keypoints.human_list[person].body_key_points_with_prob;
                int count = 0;
                double prob_sum = 0.0;
                for(int i = 0; i < body_keypoints.size(); i++)
                {
                  if(body_keypoints[i].prob > 0.0)
                  {
                    prob_sum += body_keypoints[i].prob;
                    count ++;
                  }
                }
                probs.push_back(prob_sum/count);
              }

              auto maxProb = std::max_element(probs.begin(), probs.end());
              if(*maxProb > human_threshold)
              {
                int index = std::distance(probs.begin(), maxProb);
                std::cout << "person count : " << probs.size() << std::endl;
                std::cout << "person count : " << person_num << std::endl;
                std::cout << "person " << index << " is selected" << std::endl;
                auto body_keypoints = keypoints.human_list[index].body_key_points_with_prob;
                key_points.clear();
                //keypoints_count = 0;
                KeyPoints keypoint_element;

                //hips
                if(body_keypoints[8].prob > 0.0 && body_keypoints[11].prob > 0.0){
                  keypoint_element.x = (body_keypoints[8].x + body_keypoints[11].x) / 2;
                  keypoint_element.y = (body_keypoints[8].y + body_keypoints[11].y) / 2;
                  keypoint_element.p = (body_keypoints[8].prob + body_keypoints[11].prob) / 2;
                  key_points.push_back(keypoint_element);
                }
                else{
                  keypoint_element.x = 0.0;
                  keypoint_element.y = 0.0;
                  keypoint_element.p = 0.0;
                  key_points.push_back(keypoint_element);
                }
                for (int i = 2; i < 8; ++i){

                  if(body_keypoints[i].prob > 0.0){
                    keypoint_element.x = body_keypoints[i].x;
                    keypoint_element.y = body_keypoints[i].y;
                    keypoint_element.p = body_keypoints[i].prob;
                    key_points.push_back(keypoint_element);
                  }
                  else{
                    keypoint_element.x = 0.0;
                    keypoint_element.y = 0.0;
                    keypoint_element.p = 0.0;
                    key_points.push_back(keypoint_element);
                  }
                }
                //key_points_prob = *maxProb;
                keypoints_available = true;
                ROS_INFO("keypoints available");
                /**
                for(int i = 0; i < key_points.size(); i++)
                {
                  std::cout << "joint " << i << ":" << std::endl;
                  std::cout << "x: " << key_points[i].x << std::endl;
                  std::cout << "y: " << key_points[i].y << std::endl;
                  std::cout << "prob: " << key_points[i].p << std::endl;

                }
                **/
              }
              else{
                ROS_INFO("Keypoints received, detected person abandoned");
              }
          }
          else{
            ROS_INFO("Keypoints received, no person detected");
          }
        }

        void buildProblem(Problem &problem){
            //CostFunction* oricost_hip = new AutoDiffCostFunction<OrientationCost_hip,1, 3>(new OrientationCost_hip(hips_imu_ori, hips_offset, world_to_ref));
            //CostFunction* oricost_lArm = new AutoDiffCostFunction<OrientationCost_lArm,1, 3, 12, 9>(new OrientationCost_lArm(lArm_imu_ori, lArm_offset, world_to_ref));
            //CostFunction* oricost_rArm = new AutoDiffCostFunction<OrientationCost_rArm,1, 3, 12, 9>(new OrientationCost_rArm(rArm_imu_ori, rArm_offset, world_to_ref));
            //problem.AddResidualBlock(oricost_hip, NULL, hips_joint);
            //problem.AddResidualBlock(oricost_lArm, NULL, hips_joint, spine_joint, lArm_joint);
            //problem.AddResidualBlock(oricost_rArm, NULL, hips_joint, spine_joint, rArm_joint);
            CostFunction* oricost = new AutoDiffCostFunction<Imu_Term, 10, 3, 3, 12, 12, 12>(new Imu_Term(hips_imu_ori, hips_imu_acc_pre, hips_offset, previous_hips_position,
                                                                                                   rArm_imu_ori, rArm_imu_acc_pre, rArm_offset, previous_rArm_position,
                                                                                                   lArm_imu_ori, lArm_imu_acc_pre, lArm_offset, previous_lArm_position,
                                                                                                   rHand_imu_ori, rHand_imu_acc_pre, rHand_offset, previous_rHand_position,
                                                                                                   lHand_imu_ori, lHand_imu_acc_pre, lHand_offset, previous_lHand_position,
                                                                                                   world_to_ref, bone_length, ori_weight, acc_weight));
            problem.AddResidualBlock(oricost, NULL, hips_trans, hips_joint, spine_joint, rArm_joint, lArm_joint);
            if(keypoints_available)
            {
              keypoints_available = false;
              CostFunction* poscost = new AutoDiffCostFunction<Position_Term, 7, 3, 3, 12, 12, 12>(new Position_Term(world_to_ref, bone_length, key_points, camera_ori, camera_trans, pos_weight));
              problem.AddResidualBlock(poscost, new CauchyLoss(150), hips_trans, hips_joint, spine_joint, rArm_joint, lArm_joint);
            }
            if(usePosePrior){
              CostFunction* priotcost_proj = new AutoDiffCostFunction<PoseCost_Project, 1, 12, 9, 9>(new PoseCost_Project(PCA_proj, PCA_miu, pos_proj_weight));
              CostFunction* priotcost_deviat = new AutoDiffCostFunction<PoseCost_Deviation, 1, 12, 9, 9>(new PoseCost_Deviation(PCA_proj, PCA_miu, PCA_eigenvalue, pos_dev_weight));
              problem.AddResidualBlock(priotcost_proj, NULL, spine_joint, lArm_joint, rArm_joint);
              problem.AddResidualBlock(priotcost_deviat, NULL, spine_joint, lArm_joint, rArm_joint);
            }

            if(useConstraint)
            {
              for(int i = 0; i < 3; ++i)
              {
                //problem.SetParameterUpperBound(hips_joint, i, 180.0);
                //problem.SetParameterLowerBound(hips_joint, i, -180.0);
              }
              for(int i = 0; i < 12; ++i)
              {
                problem.SetParameterUpperBound(spine_joint, i, joint_upper_bound[i]);
                problem.SetParameterLowerBound(spine_joint, i, joint_lower_bound[i]);
              }
              for(int i = 0; i < 12; ++i)
              {
                problem.SetParameterUpperBound(rArm_joint, i, joint_upper_bound[i+12]);
                problem.SetParameterLowerBound(rArm_joint, i, joint_lower_bound[i+12]);
              }
              for(int i = 0; i < 12; ++i)
              {
                problem.SetParameterUpperBound(lArm_joint, i, joint_upper_bound[i+24]);
                problem.SetParameterLowerBound(lArm_joint, i, joint_lower_bound[i+24]);
              }
            }
        }

        void solveProblem(Problem &problem){
            Solve(solver_options,&problem, &summary);
            std::cout << summary.BriefReport() << "\n";
            generateJointMsg();
            //joint_msg.header.stamp = ros::Time::now();
            //joint_publisher.publish(joint_msg);
        }

        void generateJointMsg()
        {
            const double degrees_to_radians(M_PI / 180.0);
            joint_msg.position.clear();
            for(int i = 0; i < 3; ++i)
              joint_msg.position.push_back(hips_joint[i] * degrees_to_radians);
            for(int i = 0; i < 12; ++i)
              joint_msg.position.push_back(spine_joint[i] * degrees_to_radians);
            for(int i = 0; i < 12; ++i)
              joint_msg.position.push_back(rArm_joint[i] * degrees_to_radians);
            for(int i = 0; i < 12; ++i)
              joint_msg.position.push_back(lArm_joint[i] * degrees_to_radians);
        }

        bool calculateIMUtoBoneOffset(Eigen::Matrix3d& joint_offset, string joint_name, string imu_name)
        {
            tf::StampedTransform transform;
            try{
                tf_listener.lookupTransform(joint_name.c_str(), imu_name.c_str(),
                                            ros::Time(0), transform);
            }
            catch (tf::TransformException ex){
                ROS_ERROR("%s", ex.what());
                ros::Duration(1.0).sleep();
                return false;
            }

            tf::Quaternion q = transform.getRotation();
            Eigen::Quaterniond q_offset;
            tf::quaternionTFToEigen(q, q_offset);
            joint_offset = q_offset.toRotationMatrix();

            return true;
        }

        static void* publishJointMsgThread(void *arg)
        {
            Optimizer *ptr = (Optimizer *) arg;
            ros::Rate rate(10);
            while(1)
            {
                ptr->publishJointMsg();

                pthread_testcancel(); //thread cancel point

                //ros::spinOnce();

                rate.sleep();
            }
        }

        void publishJointMsg()
        {
            joint_msg.header.stamp = ros::Time::now();
            joint_publisher.publish(joint_msg);

            //savePos();

            static tf::TransformBroadcaster br;
            tf::Transform transform;
            transform.setOrigin(tf::Vector3(hips_trans[0], hips_trans[1], hips_trans[2]));
            tf::Quaternion q_tf(0.0, 0.0, 0.0, 1.0);
            transform.setRotation(q_tf);
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "marker_0", "world"));

        }

        void publishKeyPoints()
        {
          geometry_msgs::PoseArray joint_pos_array;
          geometry_msgs::Pose joint_pos;
          for(int i = 0; i < key_points.size(); ++i)
          {
            joint_pos.position.x = key_points[i].x;
            joint_pos.position.y = key_points[i].y;
            joint_pos_array.poses.push_back(joint_pos);
          }
          joint_pos_array.header.stamp = ros::Time::now();
          image_pose_publisher.publish(joint_pos_array);
        }

        void waitforOffsetCalc(bool replayMode)
        {
            pthread_t id1;
            if(!replayMode){

              void* tret;
              int ret = pthread_create(&id1, NULL, publishJointMsgThread, (void*)this);
            }
            while(ros::ok())
            {
                std::cout << "Enter 'y' to start calculate IMU Offset ..." << std::endl;
                char c;
                std::cin >> c;

                bool success = false;
                int try_count = 0;
                while(!success)
                {
                    int success_count = 0;
                    if(calculateIMUtoBoneOffset(hips_offset, "hip", "imu_1"))
                        success_count++;
                    if(calculateIMUtoBoneOffset(lArm_offset, "lArm", "imu_2"))
                        success_count++;
                    if(calculateIMUtoBoneOffset(rArm_offset, "rArm", "imu_3"))
                        success_count++;
                    if(calculateIMUtoBoneOffset(lHand_offset, "lHand", "imu_4"))
                        success_count++;
                    if(calculateIMUtoBoneOffset(rHand_offset, "rHand", "imu_5"))
                        success_count++;

                    try_count++;
                    if (success_count == 5)
                        success = true;
                    else if (try_count > 10)
                        break;
                }
                if(success)
                    break;
                else
                    std::cout << "IMU not ready, Please try again" << std::endl;
            }
            if(!replayMode)
            int  ret = pthread_cancel(id1);
            char c;
            std::cout << "Offset calculate finished, press any key to start tracking" << std::endl;
            std::cin >> c;
        }

        void run()
        {
              clock_t start, build, solve;
              //double dur;
              ros::Rate rate(60);
              running = true;

              while(ros::ok())
              {
                if(hips_imu_recved && lArm_imu_recved && rArm_imu_recved && lHand_imu_recved && rHand_imu_recved)
                {
                  hips_imu_recved = false;
                  lArm_imu_recved = false;
                  rArm_imu_recved = false;
                  lHand_imu_recved = false;
                  rHand_imu_recved = false;

                  Problem problem;

                  start = clock();
                  //std::cout << "fuck" << hips_imu_ori << std::endl;);
                  buildProblem(problem);
                  build = clock();
                  std::cout << "building problem spends" << (double)(build - start)/CLOCKS_PER_SEC << "s" << std::endl;
                  solveProblem(problem);
                  solve = clock();
                  std::cout << "solving problem spends" << (double)(solve - build)/CLOCKS_PER_SEC << "s" << std::endl;
                  publishJointMsg();

                  //publishKeyPoints();

                  savePos();
                }
                getCameraEx();
                rate.sleep();

                ros::spinOnce();

            }
        }

        void getCameraEx()
        {
          tf::StampedTransform transform;
          try{
            tf_listener.lookupTransform("camera_base", "marker_0", ros::Time(0), transform);
          }
          catch(tf::TransformException){
            return;
          }

          tf::Quaternion q = transform.getRotation();
          tf::Vector3 cam_pos = transform.getOrigin();
          Eigen::Quaterniond q_cam;
          tf::quaternionTFToEigen(q, q_cam);
          camera_ori = q_cam.toRotationMatrix();
          camera_trans << cam_pos.x(), cam_pos.y(), cam_pos.z();
        }

        void getOptimParam()
        {
          priv_nh.param("usePosePrior",usePosePrior, true);
          priv_nh.param("useConstraint",useConstraint, true);
          priv_nh.param("PoseProjectionWeight",pos_proj_weight, 0.0001);
          priv_nh.param("PoseDeviationWeight",pos_dev_weight, 60.0);
          priv_nh.param("OrientationWeight",ori_weight, 1.0);
          priv_nh.param("AccelerationWeight",acc_weight, 0.005);
          priv_nh.param("PositionWeight",pos_weight, 0.01);
          priv_nh.param("HumanThreshold",human_threshold, 0.6);
        }

        void getPos(Eigen::Matrix<double, 3, 1> hips_pos,
                    Eigen::Matrix<double, 3, 1>& rArm_pos,
                    Eigen::Matrix<double, 3, 1>& lArm_pos,
                    Eigen::Matrix<double, 3, 1>& rHand_pos,
                    Eigen::Matrix<double, 3, 1>& lHand_pos)
        {

          Eigen::Matrix3d ite_ori;
          Eigen::Matrix<double ,3, 1> ite_trans;

          Eigen::Matrix<double, 3, 1> cam_space_trans;
          double img_pos_x;
          double img_pos_y;
          //T x_diff, y_diff;
          double fx = 1068.2054759 / 2;
          double fy = 1068.22398224 / 2;
          double cx = 964.1001882846 / 2;
          double cy = 538.5221553 / 2;
          geometry_msgs::PoseArray joint_pos_array;
          geometry_msgs::Pose joint_pos;
          //double world_to_ref[3] = {0.0, 90.0, 0.0};

          double rot[9];

          double period = 0.05 * 0.05;
          Eigen::Matrix<double, 3, 1> solved_acc;
          Eigen::Matrix<double, 3, 1> acc_diff;


          //EulerAnglesToRotationMatrixZXY(world_to_ref, 3, rot);
          //Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
          ite_ori = world_to_ref;
          ite_trans = hips_pos;

          if (running){
            //solved_acc = (ite_trans - 2 * previous_hips_position[1] + previous_hips_position[0]) / period;
            //acc_diff = solved_acc - hips_imu_acc;
            //std::cout << "acc_diff: \n" << acc_diff << std::endl;
          }
          //std::cout << "acc_diff: "<< acc_diff << std::endl;

          {
              EulerAnglesToRotationMatrixZXY(hips_joint, 3, rot);
              Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
              ite_ori = ite_ori * ori;
              //Eigen::Matrix<double, 3, 1> ite_trans;
              ite_trans += ite_ori * bone_length[0];
          }
          /***
          cam_space_trans = camera_trans + camera_ori * ite_trans;
          joint_pos.position.x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
          joint_pos.position.y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
          joint_pos_array.poses.push_back(joint_pos);
          ***/

          for(int i = 0; i < 3; ++i)
          {
              EulerAnglesToRotationMatrixZXY(spine_joint + i * 3, 3, rot);
              Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
              ite_ori = ite_ori * ori;

              ite_trans += ite_ori * bone_length[1+i];
          }

          Eigen::Matrix<double, 3, 1> spine_trans = ite_trans;
          Eigen::Matrix<double, 3, 3> spine_ite_ori = ite_ori;


          //for right arm
          {
              EulerAnglesToRotationMatrixZXY(spine_joint + 3 * 3, 3, rot);
              Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
              ite_ori = ite_ori * ori;

              ite_trans += ite_ori * bone_length[4];
          }
          /***
          cam_space_trans = camera_trans + camera_ori * ite_trans;
          joint_pos.position.x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
          joint_pos.position.y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
          joint_pos_array.poses.push_back(joint_pos);

          ***/


          for(int i = 0; i < 2; ++i)
          {
              EulerAnglesToRotationMatrixZXY(rArm_joint + i * 3, 3, rot);
              Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
              ite_ori = ite_ori * ori;

              ite_trans += ite_ori * bone_length[5+i];
          }
          rArm_pos = ite_trans;
          /***
          cam_space_trans = camera_trans + camera_ori * ite_trans;
          joint_pos.position.x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
          joint_pos.position.y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
          joint_pos_array.poses.push_back(joint_pos);

          ***/

          for(int i = 2; i < 4; ++i)
          {
              EulerAnglesToRotationMatrixZXY(rArm_joint + i * 3, 3, rot);
              Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
              ite_ori = ite_ori * ori;

              ite_trans += ite_ori * bone_length[5+i];
          }

          rHand_pos = ite_trans;
          /***
          cam_space_trans = camera_trans + camera_ori * ite_trans;
          joint_pos.position.x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
          joint_pos.position.y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
          joint_pos_array.poses.push_back(joint_pos);
          ***/

          //for left arm
          ite_trans = spine_trans;
          ite_ori = spine_ite_ori;
          {
              EulerAnglesToRotationMatrixZXY(spine_joint + 3 * 3, 3, rot);
              Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
              ite_ori = ite_ori * ori;

              ite_trans += ite_ori * bone_length[9];
          }
          /***
          cam_space_trans = camera_trans + camera_ori * ite_trans;
          joint_pos.position.x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
          joint_pos.position.y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
          joint_pos_array.poses.push_back(joint_pos);
          ***/
          for(int i = 0; i < 2; ++i)
          {
              EulerAnglesToRotationMatrixZXY(lArm_joint + i * 3, 3, rot);
              Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
              ite_ori = ite_ori * ori;

              ite_trans += ite_ori * bone_length[10+i];
          }

          lArm_pos = ite_trans;
          /***
          cam_space_trans = camera_trans + camera_ori * ite_trans;
          joint_pos.position.x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
          joint_pos.position.y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
          joint_pos_array.poses.push_back(joint_pos);
          ***/

          for(int i = 2; i < 4; ++i)
          {
              EulerAnglesToRotationMatrixZXY(lArm_joint + i * 3, 3, rot);
              Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
              ite_ori = ite_ori * ori;

              ite_trans += ite_ori * bone_length[10+i];
          }

          lHand_pos = ite_trans;
          /**

          cam_space_trans = camera_trans + camera_ori * ite_trans;
          joint_pos.position.x = cam_space_trans(0)/cam_space_trans(2) * fx + cx;
          joint_pos.position.y = cam_space_trans(1)/cam_space_trans(2) * fy + cy;
          joint_pos_array.poses.push_back(joint_pos);

          joint_pos_array.header.stamp = ros::Time::now();
          image_pose_publisher.publish(joint_pos_array);
          ***/

          /***
          static tf::TransformBroadcaster br;
          tf::Transform transform;
          transform.setOrigin(tf::Vector3(ite_trans(0), ite_trans(1), ite_trans(2)));
          tf::Quaternion q_tf(0.0, 0.0, 0.0, 1.0);
          transform.setRotation(q_tf);
          br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "marker_0", "test_trans"));
          ***/
        }

        void savePos()
        {
          Eigen::Map<Eigen::Matrix<double, 3, 1> > hips_pos(hips_trans);
          Eigen::Matrix<double, 3, 1> rArm_pos;
          Eigen::Matrix<double, 3, 1> lArm_pos;
          Eigen::Matrix<double, 3, 1> rHand_pos;
          Eigen::Matrix<double, 3, 1> lHand_pos;
          getPos(hips_pos, rArm_pos, lArm_pos, rHand_pos, lHand_pos);
          previous_hips_position.erase(previous_hips_position.begin());
          previous_hips_position.push_back(hips_pos);
          previous_rArm_position.erase(previous_rArm_position.begin());
          previous_rArm_position.push_back(rArm_pos);
          previous_lArm_position.erase(previous_lArm_position.begin());
          previous_lArm_position.push_back(lArm_pos);
          previous_rHand_position.erase(previous_rHand_position.begin());
          previous_rHand_position.push_back(rHand_pos);
          previous_lHand_position.erase(previous_lHand_position.begin());
          previous_lHand_position.push_back(lHand_pos);

          //std::cout << "hips(t-2): " << previous_hips_position[0] << "  hips(t-1): " << previous_hips_position[1] << std::endl;

        }
        void init()
        {
          Eigen::Map<Eigen::Matrix<double, 3, 1> > hips_pos(hips_trans);
          Eigen::Matrix<double, 3, 1> rArm_pos;
          Eigen::Matrix<double, 3, 1> lArm_pos;
          Eigen::Matrix<double, 3, 1> rHand_pos;
          Eigen::Matrix<double, 3, 1> lHand_pos;
          getPos(hips_pos, rArm_pos, lArm_pos, rHand_pos, lHand_pos);
          previous_hips_position.clear();
          previous_rArm_position.clear();
          previous_lArm_position.clear();
          previous_rHand_position.clear();
          previous_lHand_position.clear();

          previous_hips_position.push_back(hips_pos);
          previous_rArm_position.push_back(rArm_pos);
          previous_lArm_position.push_back(lArm_pos);
          previous_rHand_position.push_back(rHand_pos);
          previous_lHand_position.push_back(lHand_pos);

          previous_hips_position.push_back(hips_pos);
          previous_rArm_position.push_back(rArm_pos);
          previous_lArm_position.push_back(lArm_pos);
          previous_rHand_position.push_back(rHand_pos);
          previous_lHand_position.push_back(lHand_pos);




          hips_imu_recved = false;
          lArm_imu_recved = false;
          rArm_imu_recved = false;
          lHand_imu_recved = false;
          rHand_imu_recved = false;

          keypoints_available = false;

        }



    private:
        ros::NodeHandle nh, priv_nh;

        ros::Subscriber hips_imu_sub;
        ros::Subscriber lArm_imu_sub;
        ros::Subscriber rArm_imu_sub;
        ros::Subscriber lHand_imu_sub;
        ros::Subscriber rHand_imu_sub;

        ros::Subscriber human_keypoints_sub;


        ros::Publisher joint_publisher;
        ros::Publisher image_pose_publisher;

        tf::TransformListener tf_listener;

        Eigen::Matrix3d hips_imu_ori;
        Eigen::Matrix<double, 3, 1> hips_imu_acc;
        Eigen::Matrix<double, 3, 1> hips_imu_acc_pre;

        Eigen::Matrix3d lArm_imu_ori;
        Eigen::Matrix<double, 3, 1> lArm_imu_acc;
        Eigen::Matrix<double, 3, 1> lArm_imu_acc_pre;

        Eigen::Matrix3d rArm_imu_ori;
        Eigen::Matrix<double, 3, 1> rArm_imu_acc;
        Eigen::Matrix<double, 3, 1> rArm_imu_acc_pre;

        Eigen::Matrix3d lHand_imu_ori;
        Eigen::Matrix<double, 3, 1> lHand_imu_acc;
        Eigen::Matrix<double, 3, 1> lHand_imu_acc_pre;

        Eigen::Matrix3d rHand_imu_ori;
        Eigen::Matrix<double, 3, 1> rHand_imu_acc;
        Eigen::Matrix<double, 3, 1> rHand_imu_acc_pre;

        vector<KeyPoints> key_points;
        //int keypoints_count;
        //double key_points_prob;
        double human_threshold;

        Eigen::Matrix3d camera_ori;
        Eigen::Matrix<double, 3, 1> camera_trans;
        //int keypoints_num;

        Eigen::Matrix3d hips_offset;
        Eigen::Matrix3d lArm_offset;
        Eigen::Matrix3d rArm_offset;
        Eigen::Matrix3d lHand_offset;
        Eigen::Matrix3d rHand_offset;

        Eigen::Matrix3d world_to_ref;

        vector<Eigen::Matrix<double, 3, 1> > bone_length;

        sensor_msgs::JointState joint_msg;

        vector<double> init_joints;

        double hips_trans[3];
        double hips_joint[3];
        double spine_joint[12];
        double rArm_joint[12];
        double lArm_joint[12];

        vector<double> joint_upper_bound;
        vector<double> joint_lower_bound;

        vector<Eigen::Matrix<double, 3, 1> > previous_hips_position;
        vector<Eigen::Matrix<double, 3, 1> > previous_lArm_position;
        vector<Eigen::Matrix<double, 3, 1> > previous_rArm_position;
        vector<Eigen::Matrix<double, 3, 1> > previous_lHand_position;
        vector<Eigen::Matrix<double, 3, 1> > previous_rHand_position;

        bool hips_imu_recved, lArm_imu_recved, rArm_imu_recved, lHand_imu_recved, rHand_imu_recved;
        bool keypoints_available;
        bool running;

        //Problem problem;

        Solver::Options solver_options;
        Solver::Summary summary;

        Eigen::Matrix<double, 30, 8> PCA_proj;
        Eigen::Matrix<double, 30, 1> PCA_miu;
        Eigen::Matrix<double, 8, 1> PCA_eigenvalue;

        bool usePosePrior, useConstraint;
        double pos_proj_weight;
        double pos_dev_weight;
        double ori_weight;
        double acc_weight;
        double pos_weight;



};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "pose_optimizer");

    Optimizer poseOptimizer;

    poseOptimizer.getOptimParam();
    poseOptimizer.init();
    if (argc > 1 && argv[1] == "replayMode"){
      ROS_INFO("Starting Program in replay mode");
      poseOptimizer.waitforOffsetCalc(true);
    }
    else
      poseOptimizer.waitforOffsetCalc(false);


    //ros::spinOnce()

    //poseOptimizer.init();

    poseOptimizer.run();

    return 0;
}
