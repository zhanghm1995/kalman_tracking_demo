/*======================================================================
* Author   : Haiming Zhang
* Email    : zhanghm_1995@qq.com
* Version  :　2019年1月2日
* Copyright    :
* Descriptoin  :
* References   :
======================================================================*/
#include <vector>
#include <algorithm>
#include <ros/ros.h>
#include <tf/LinearMath/Transform.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>

#include <boost/algorithm/string.hpp>
#include <boost/timer.hpp>
#include <boost/math/special_functions/round.hpp>

#include <darknet_ros_msgs/ImageWithBBoxes.h>
#include "iv_dynamicobject_msgs/ObjectArray.h"
#include "ukf.h"

using namespace sensors_fusion;
using namespace visualization_msgs;
using namespace tracking;

// Whether save groundtruth data and tracked results for analyzing
static bool _is_save_data = true;

static int64_t gtm()
{
    struct timeval tm;
    gettimeofday(&tm, 0);
    // return us
    int64_t re = (((int64_t)tm.tv_sec) * 1000 * 1000 + tm.tv_usec);
    return re;
}

void toObjectTrackArray(const iv_dynamicobject_msgs::ObjectArray::ConstPtr& msg,
                         ObjectTrackArray& obj_track_array)
{
  for (size_t i = 0; i < msg->list.size(); ++i) {
    ObjectTrack obj;
    obj.length = msg->list[i].length;
    obj.width = msg->list[i].width;
    obj.height = msg->list[i].height;

    // Get velo_pos
    obj.velo_pos.header.frame_id = "velo_link";
    obj.velo_pos.point.x = msg->list[i].velo_pose.point.x;
    obj.velo_pos.point.y = msg->list[i].velo_pose.point.y;
    obj.velo_pos.point.z = msg->list[i].velo_pose.point.z;
    // Get world pos
    obj.world_pos.header = msg->list[i].world_pose.header;
    obj.world_pos.point.x = msg->list[i].world_pose.point.x;
    obj.world_pos.point.y = msg->list[i].world_pose.point.y;
    obj.world_pos.point.z = msg->list[i].world_pose.point.z;
    // Get heading
    obj.orientation = msg->list[i].heading;
    // Get velocity
    obj.velocity = msg->list[i].velocity;

    obj_track_array.push_back(obj);
  }
}

class TrackingProcess
{
public:
  TrackingProcess(ros::NodeHandle& node):
    tfListener_(tfBuffer_, node),
    cloud_sub_(node, "/kitti/velo/pointcloud", 10),
    object_array_sub_(node, "/detection/object_array", 10),
    sync_(MySyncPolicy(2), cloud_sub_, object_array_sub_)
  {
    is_initialized_ = false;
    vis_marker_pub_ = node.advertise<Marker>("/viz/visualization_marker", 1);

    ROS_INFO_STREAM("Entering in UKF tracking...");
    sync_.registerCallback(boost::bind(&TrackingProcess::syncCallback, this, _1, _2));
  }

  void syncCallback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg,
      const iv_dynamicobject_msgs::ObjectArrayConstPtr& obj_msg)
  {
    double time_stamp = obj_msg->header.stamp.toSec();

    // Convert object array message to object track array type
    ROS_WARN_STREAM("Enter in syncCallback..."<<std::setprecision(20)<<
        lidar_msg->header.stamp.toSec()<<" "<<
        obj_msg->header.stamp.toSec());
    ObjectTrackArray obj_track_array;
    toObjectTrackArray(obj_msg, obj_track_array);


    // Transform coordinate
    if(!transformCoordinate(obj_track_array, time_stamp))
      return;

    if (_is_save_data) {// Saving groundtruth
      FILE* fp_groundtruth = fopen("/home/zhanghm/Test_code/catkin_ws_dev/groundtruth.txt", "a");
      fprintf(fp_groundtruth, "%.9f %.3f %.3f %.3f %.3f\n",
              time_stamp,
              obj_track_array[0].world_pos.point.x,
              obj_track_array[0].world_pos.point.y,
              obj_track_array[0].world_pos.point.z,
              obj_track_array[0].heading);
      fclose(fp_groundtruth);
    }

    // Initialize
    ros::WallTime start_, end_;
    start_ = ros::WallTime::now();
    if(is_initialized_){
      ukf.predict(time_stamp);
      ukf.update(obj_track_array[0]);
    }
    else{
      int id = 1;
      ukf.initialize(id, obj_track_array[0], time_stamp);
      is_initialized_ = true;
    }
    end_ = ros::WallTime::now();
    // print results
    double execution_time = (end_ - start_).toNSec() * 1e-3;
    ROS_INFO_STREAM("Exectution time (us): " << execution_time);

    // ----------Visualization-----------------
    sensors_fusion::ObjectTrackArray object_track;
    getObjectTrackArray(object_track, time_stamp);

    showTrackingArrow(vis_marker_pub_, object_track, time_stamp);
  }

  void getObjectTrackArray(sensors_fusion::ObjectTrackArray& track_array, double time_stamp)
  {
    // Grab track
    tracking::Track & track = ukf.track_;

    // Create new message and fill it
    sensors_fusion::ObjectTrack track_msg;
    track_msg.id = track.id;
    track_msg.world_pos.header.frame_id = "world";
    track_msg.world_pos.header.stamp = ros::Time(time_stamp);
    track_msg.world_pos.point.x = track.sta.x[0];
    track_msg.world_pos.point.y = track.sta.x[1];
    track_msg.world_pos.point.z = track.sta.z;

//    try{
//      listener.transformPoint("velo_link",
//          track_msg.world_pos,
//          track_msg.velo_pos);
//    }
//    catch(tf::TransformException& ex){
//      ROS_ERROR("Received an exception trying to transform a point from"
//          "\"velo_link\" to \"world\": %s", ex.what());
//    }

    track_msg.velocity = track.sta.x[2];
    track_msg.heading = track.sta.x[3];

    if (_is_save_data) {// Saving track results
      FILE* fp_track = fopen("/home/zhanghm/Test_code/catkin_ws_dev/track_results.txt", "a");
      fprintf(fp_track, "%.9f %.3f %.3f %.3f %.3f %.3f\n",
                         time_stamp, track.sta.x[0], track.sta.x[1], track.sta.x[2],
                         track.sta.x[3],track.sta.x[4]);
      fclose(fp_track);
    }

    try{
      geometry_msgs::PoseStamped world_pose, velo_pose;
      world_pose.header.frame_id = "world";
      world_pose.header.stamp = ros::Time(time_stamp);
      world_pose.pose.position = track_msg.world_pos.point;
      world_pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0,0,track_msg.heading);
      listener.transformPose("velo_link",
          world_pose,
          velo_pose);

      track_msg.velo_pos.header.frame_id = "velo_link";
      track_msg.velo_pos.header.stamp = world_pose.header.stamp;
      track_msg.velo_pos.point = velo_pose.pose.position;
      track_msg.heading = tf::getYaw(velo_pose.pose.orientation);
    }
    catch(tf::TransformException& ex){
      ROS_ERROR("Received an exception trying to transform a point from"
          "\"world\" to \"velo_link\": %s", ex.what());
    }

    track_msg.width = track.geo.width;
    track_msg.length = track.geo.length;
    track_msg.height = track.geo.height;
    track_msg.orientation = track.geo.orientation;
    track_msg.object_type = track.sem.name;
    track_msg.confidence = track.sem.confidence;

    track_msg.is_valid = true;

    // Push back track message
    track_array.push_back(track_msg);

    ROS_INFO_STREAM("velocity is "<<track_msg.velocity<<" "<<
        "heading is "<<track_msg.heading<<" "<<
        "orientation is "<<track_msg.orientation<<" "<<
        "length is "<<track_msg.length<<" "<<
        "width is "<<track_msg.width<<" "<<
        "height is "<<track_msg.height);
  }

  bool transformCoordinate(sensors_fusion::ObjectTrackArray& obj_array, double time_stamp)
  {
    // Transform objects in camera and world frame
//    try{
//      for(size_t i = 0; i < obj_array.size(); ++i){
//        obj_array[i].velo_pos.header.stamp = ros::Time(time_stamp);
//        listener.transformPoint("world",
//            obj_array[i].velo_pos,
//            obj_array[i].world_pos);
//      }
//      return true;
//    }
//    catch(tf::TransformException& ex){
//      ROS_ERROR("%s", ex.what());
//      return false;
//    }

    try{
      for(size_t i = 0; i < obj_array.size(); ++i){
        geometry_msgs::PoseStamped  velo_pose, world_pose;
        velo_pose.header.frame_id = "velo_link";
        velo_pose.header.stamp = ros::Time(time_stamp);
        velo_pose.pose.position = obj_array[i].velo_pos.point;
        velo_pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0,0,obj_array[i].orientation);
//        listener.waitForTransform("world", "velo_link", ros::Time(time_stamp),ros::Duration(1.0));
//        listener.transformPose("world",
//                        velo_pose,
//                        world_pose);
        if(!tfBuffer_.canTransform("world","velo_link",ros::Time(time_stamp), ros::Duration(0.2)))
          ROS_ERROR("No transform");
        else
          ROS_WARN("Has transform");
        tfBuffer_.transform<geometry_msgs::PoseStamped>(velo_pose, world_pose, "world", ros::Duration(0.2));
        obj_array[i].world_pos.header.frame_id = "world";
        obj_array[i].world_pos.header.stamp = velo_pose.header.stamp;
        obj_array[i].world_pos.point = world_pose.pose.position;
        obj_array[i].heading = tf::getYaw(world_pose.pose.orientation);
      }
      return true;
    }
    catch(tf::TransformException& ex){
      ROS_ERROR("Received an exception trying to transform a point from"
          "\"velo_link\" to \"world\": %s", ex.what());
      return false;
    }
  }

  void showTrackingArrow(ros::Publisher &pub,
                         const ObjectTrackArray& obj_array,
                         double time_stamp)
  {
    for(int i = 0; i < obj_array.size(); i++){
      visualization_msgs::Marker arrowsG;
//      arrowsG.lifetime = ros::Duration(0.2);

      if(obj_array[i].velocity < 0.5)//TODO: maybe add is_static flag member
        continue;

      arrowsG.header.frame_id = "velo_link";
//      arrowsG.header.stamp = ros::Time(time_stamp);
      arrowsG.ns = "arrows";
      arrowsG.action = visualization_msgs::Marker::ADD;
      arrowsG.type =  visualization_msgs::Marker::ARROW;

      // green
      arrowsG.color.g = 1.0f;
      arrowsG.color.a = 1.0;
      arrowsG.id = i;

      double tv   = obj_array[i].velocity;
      double tyaw = obj_array[i].heading;

      // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
#if 0
      arrowsG.pose.position.x = obj_array[i].world_pos.point.x;
      arrowsG.pose.position.y = obj_array[i].world_pos.point.y;
      arrowsG.pose.position.z = obj_array[i].world_pos.point.z + obj_array[i].height/2;
#else
      arrowsG.pose.position.x = obj_array[i].velo_pos.point.x;
      arrowsG.pose.position.y = obj_array[i].velo_pos.point.y;
      arrowsG.pose.position.z = obj_array[i].velo_pos.point.z + obj_array[i].height/2;
#endif
      // convert from 3 angles to quartenion
      tf::Matrix3x3 obs_mat;
      obs_mat.setEulerYPR(tyaw, 0, 0); // yaw, pitch, roll
      tf::Quaternion q_tf;
      obs_mat.getRotation(q_tf);
      arrowsG.pose.orientation.x = q_tf.getX();
      arrowsG.pose.orientation.y = q_tf.getY();
      arrowsG.pose.orientation.z = q_tf.getZ();
      arrowsG.pose.orientation.w = q_tf.getW();

      // Set the scale of the arrowsG -- 1x1x1 here means 1m on a side
      arrowsG.scale.x = tv;
      arrowsG.scale.y = 0.15;
      arrowsG.scale.z = 0.15;

      pub.publish(arrowsG);
    }
  }

private:
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
  message_filters::Subscriber<iv_dynamicobject_msgs::ObjectArray> object_array_sub_;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, iv_dynamicobject_msgs::ObjectArray> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync_;

  tf::TransformListener listener;

  tf2_ros::Buffer tfBuffer_;
  tf2_ros::TransformListener tfListener_;

  ros::Publisher vis_marker_pub_;

  bool is_initialized_;
  UnscentedKF ukf;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "tracking_node");
  ros::NodeHandle node;
  TrackingProcess track_process(node);

  ros::spin();
}
