#include "orientation_term.h"
#include <sensor_msgs/JointState.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;
int main(int argc, char** argv)
{
    vector<string> joint_names;
    joint_names.push_back("hip_to_spine1_z");
    joint_names.push_back("spine1_z_to_spine1_x");
    joint_names.push_back("spine1_x_to_spine1_y");

    joint_names.push_back("spine1_to_spine2_z");
    joint_names.push_back("spine2_z_to_spine2_x");
    joint_names.push_back("spine2_x_to_spine2_y");
    double joints[] = {10.0, 5.0, 30.0, 8.0, 40.0, 0.0};
    const double degrees_to_radians(M_PI / 180.0);

    sensor_msgs::JointState joint_state;
    for(int i = 0; i < 6;++i)
    {
        joint_state.name.push_back(joint_names[i]);
        joint_state.position.push_back(joints[i] * degrees_to_radians);
    }

    double R1[9];
    double R2[9];

    EulerAnglesToRotationMatrixZXY(joints, 3, R1);
    EulerAnglesToRotationMatrixZXY(joints + 3, 3, R2);

    Eigen::Map<const Eigen::Matrix3d> Eigen_R1(R1);
    Eigen::Map<const Eigen::Matrix3d> Eigen_R2(R2);

    cout << Eigen_R1.transpose() << endl;

    Eigen::Quaterniond q;
    Eigen::Matrix3d R = Eigen_R1 * Eigen_R2;
    q = R.transpose();
    //q = ori1;
    q = q.normalized();

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
    tf::Quaternion tf_q(q.x(), q.y(), q.z(), q.w());
    transform.setRotation(tf_q);transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));


    ros::init(argc, argv, "pose_test");
    ros::NodeHandle nh;

    static tf::TransformBroadcaster br;

    ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("/arm_ns/joint_states", 10);
    ros::Rate rate(10);

    while(ros::ok())
    {
        joint_state.header.stamp = ros::Time::now();
        joint_pub.publish(joint_state);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "hip", "test_rot"));
        rate.sleep();
    }

    return 0;

}
