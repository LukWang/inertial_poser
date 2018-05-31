#include <sensor_msgs/JointState.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <fstream>
#include <orientation_term.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <pthread.h>
#include <iostream>
#include <vector>



using namespace std;
static sensor_msgs::JointState joint_msg;
static ros::Publisher joint_publisher;

void init_length(vector<Eigen::Matrix<double, 3, 1> >& bone_length)
{
  Eigen::Matrix<double, 3, 1> spine_length;
  Eigen::Matrix<double, 3, 1> spine1_length;
  Eigen::Matrix<double, 3, 1> spine2_length;
  Eigen::Matrix<double, 3, 1> spine3_length;
  Eigen::Matrix<double, 3, 1> left_chest_length;
  Eigen::Matrix<double, 3, 1> lshoulder_length;
  Eigen::Matrix<double, 3, 1> upperlArm_length;
  Eigen::Matrix<double, 3, 1> forelArm_length;
  Eigen::Matrix<double, 3, 1> right_chest_length;
  Eigen::Matrix<double, 3, 1> rshoulder_length;
  Eigen::Matrix<double, 3, 1> upperrArm_length;
  Eigen::Matrix<double, 3, 1> forerArm_length;

  spine_length << 0.0,0.046171358,-0.06925691;
  spine1_length << 0.0,0.01603502,-0.09093962;
  spine2_length << 0.0,0.008048244,-0.09199118;
  spine3_length << 0.0,0.0,-0.092342462;

  right_chest_length << 0.029176472,-0.048627538,-0.157553406;
  rshoulder_length << 0.144910048,0.0,0.0;
  upperrArm_length << 0.288867596,0.0,5.08e-08;
  forerArm_length << 0.2196027866,0.0,0.0;

  left_chest_length << -0.0291764466 ,  -0.048627538 ,  -0.157553406;
  lshoulder_length << -0.144910048 ,  0.0 ,  0.0;
  upperlArm_length << -0.288867596 ,  0.0 ,  -5.08e-08;
  forelArm_length <<  -0.2196027866 ,  0.0 ,  0.0;

  bone_length.push_back(spine_length);
  bone_length.push_back(spine1_length);
  bone_length.push_back(spine2_length);
  bone_length.push_back(spine3_length);
  bone_length.push_back(right_chest_length);
  bone_length.push_back(rshoulder_length);
  bone_length.push_back(upperrArm_length);
  bone_length.push_back(forerArm_length);
  bone_length.push_back(left_chest_length);
  bone_length.push_back(lshoulder_length);
  bone_length.push_back(upperlArm_length);
  bone_length.push_back(forelArm_length);

}


static void *publishJoint(void* arg)
{
  //sensor_msgs::JointState* joint_msg = (sensor_msgs::JointState*) joint_msg_ptr;
  ros::Rate r(30);
  while(1)
  {
    joint_msg.header.stamp = ros::Time::now();
    joint_publisher.publish(joint_msg);
    pthread_testcancel();
    r.sleep();
  }

}

void saveTrans(double* hip_joint, double* spine_joint, double* rArm_joint, double* lArm_joint, vector<Eigen::Matrix<double, 3, 1> >& bone_length)
{

  Eigen::Matrix3d ite_ori;
  Eigen::Matrix<double ,3, 1> ite_trans;
  double world_to_ref[3] = {0.0, 90.0, 0.0};

  double rot[9];

  EulerAnglesToRotationMatrixZXY(world_to_ref, 3, rot);
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
  ite_ori = ori;

  {
      EulerAnglesToRotationMatrixZXY(hip_joint, 3, rot);
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
      ite_ori = ite_ori * ori;
      //Eigen::Matrix<double, 3, 1> ite_trans;
      ite_trans = ite_ori * bone_length[0];
  }

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


  for(int i = 0; i < 3; ++i)
  {
      EulerAnglesToRotationMatrixZXY(rArm_joint + i * 3, 3, rot);
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
      ite_ori = ite_ori * ori;

      ite_trans += ite_ori * bone_length[5+i];
  }

  //for left arm
  ite_trans = spine_trans;
  ite_ori = spine_ite_ori;
  {
      EulerAnglesToRotationMatrixZXY(spine_joint + 3 * 3, 3, rot);
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
      ite_ori = ite_ori * ori;

      ite_trans += ite_ori * bone_length[8];
  }


  for(int i = 0; i < 3; ++i)
  {
      EulerAnglesToRotationMatrixZXY(lArm_joint + i * 3, 3, rot);
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > ori(rot);
      ite_ori = ite_ori * ori;

      ite_trans += ite_ori * bone_length[9+i];
  }





  static tf::TransformBroadcaster br;
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(ite_trans(0), ite_trans(1), ite_trans(2)));
  tf::Quaternion q_tf(0.0, 0.0, 0.0, 1.0);
  transform.setRotation(q_tf);
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "test_trans"));

}

void sendTF(double* hip_joint, double* spine_joint, double* lArm_joint, Eigen::Matrix3d& offset, Eigen::Matrix3d& ref)
{
            static tf::TransformListener tf_listener;
            Eigen::Matrix<double, 3, 3> base;
            //Eigen::Matrix<T, 3, 3> residual;
            Eigen::Quaternion<double> q;

            double world_to_ref[3] = {0.0, 90.0, 0.0};

            double rot[9];

            EulerAnglesToRotationMatrixZXY(world_to_ref, 3, rot);
            Eigen::Map<const Eigen::Matrix<double, 3, 3> > ori(rot);
            base = ori.transpose();

            {
                EulerAnglesToRotationMatrixZXY(hip_joint, 3, rot);
                Eigen::Map<const Eigen::Matrix<double, 3, 3> > ori(rot);
                base = base * ori.transpose();
            }
            for(int i = 0; i < 4; ++i)
            {
                EulerAnglesToRotationMatrixZXY(spine_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<double, 3, 3> > ori(rot);
                base = base * ori.transpose();
            }

            for(int i = 0; i < 3; ++i)
            {
                EulerAnglesToRotationMatrixZXY(lArm_joint + i * 3, 3, rot);
                Eigen::Map<const Eigen::Matrix<double, 3, 3> > ori(rot);
                base = base * ori.transpose();
            }

            base = base * offset;

            Eigen::Matrix3d res;
            res = base.inverse() * ref;

            Eigen::Quaterniond q_res;

            q_res = res;

            q_res.normalize();

            double cost_ori;

            cost_ori = q_res.x() * q_res.x()
                        + q_res.y() * q_res.y()
                        + q_res.z() * q_res.z();

            cout << cost_ori << endl;

            q = base;
            q.normalize();
            static tf::TransformBroadcaster br;
            tf::Transform transform;
            transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
            tf::Quaternion q_tf(q.x(), q.y(), q.z(), q.w());
            transform.setRotation(q_tf);
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "test_lArm"));


            //base = base * _lArm_offset.cast<T>();

            //residual = base.inverse() * _lArm_imu_ori.cast<T>();
            //q_res = residual;
            //q_res.normalize();
            //cost_ori[0] = q_res.x() * q_res.x()
            //            + q_res.y() * q_res.y()
            //            + q_res.z() * q_res.z();

            //cost_ori[0] *= 100;
            //return true;

}

bool calculateIMUtoBoneOffset(Eigen::Matrix3d& joint_offset, string joint_name, string imu_name)
{
    static tf::TransformListener tf_listener;
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


int main(int argc, char** argv)
{
  double hip_joint[3];
  double spine_joint[12];
  double rArm_joint[9];
  double lArm_joint[9];


  ifstream in;
  in.open("/home/luk/Public/Total Capture/S3/acting1_BlenderZXY_YmZ.bvh");
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
      if(!(joint_index == 5 || joint_index == 6 || joint_index == 10 || joint_index == 11 || joint_index == 12 || joint_index >=16))
      {
          init_joints.push_back(bvhdata);
          //cout << data << endl;
      }
  }
  //static sensor_msgs::JointState joint_msg;
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
  joint_names.push_back("rForeArm_to_rHand_z");
  joint_names.push_back("rHand_z_to_rHand_x");
  joint_names.push_back("rHand_x_to_rHand_y");

  joint_names.push_back("lShoulder_to_lArm_z");
  joint_names.push_back("lArm_z_to_lArm_x");
  joint_names.push_back("lArm_x_to_lArm_y");
  joint_names.push_back("lArm_to_lForeArm_z");
  joint_names.push_back("lForeArm_z_to_lForeArm_x");
  joint_names.push_back("lForeArm_x_to_lForeArm_y");
  joint_names.push_back("lForeArm_to_lHand_z");
  joint_names.push_back("lHand_z_to_lHand_x");
  joint_names.push_back("lHand_x_to_lHand_y");

  joint_msg.name = joint_names;
  const double degrees_to_radians(M_PI / 180.0);
  for(int i = 0; i < init_joints.size(); ++i)
  {
    joint_msg.position.push_back(init_joints[i] * degrees_to_radians);
  }

  for(int i = 0; i < 3; ++i)
    hip_joint[i] = init_joints[i];
  for(int i = 0; i < 12; ++i)
    spine_joint[i] = init_joints[3 + i];
  for(int i = 0; i < 9; ++i)
    rArm_joint[i] = init_joints[15 + i];
  for(int i = 0; i < 9; ++i)
    lArm_joint[i] = init_joints[24 + i];

  Eigen::Matrix3d offset;
  Eigen::Matrix3d ref;

  ros::init(argc, argv, "pose_test");
  ros::NodeHandle nh;
  joint_publisher = nh.advertise<sensor_msgs::JointState>("/arm_ns/joint_states", 10);

  pthread_t id;
  pthread_create(&id, NULL, publishJoint,NULL);

  bool ret = false;
  while(!ret && ros::ok())
    ret = calculateIMUtoBoneOffset(offset, "lHand", "spine3");
  calculateIMUtoBoneOffset(ref, "spine3", "world");

  vector<Eigen::Matrix<double, 3, 1> > bone_length;
  init_length(bone_length);


  while(ros::ok())
  {

    sendTF(hip_joint, spine_joint, lArm_joint, offset, ref);
    saveTrans(hip_joint, spine_joint, rArm_joint, lArm_joint, bone_length);
  }
  ret = pthread_cancel(id);

  return 0;

}
