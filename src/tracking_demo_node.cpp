/*======================================================================
* Author   : Haiming Zhang
* Email    : zhanghm_1995@qq.com
* Version  :　2019年1月1日
* Copyright    :
* Descriptoin  :
* References   :
======================================================================*/
#include <vector>
#include <ros/ros.h>
#include <tf/LinearMath/Transform.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <cv_bridge/cv_bridge.h>

#include <boost/algorithm/string.hpp>
#include <boost/timer.hpp>
#include <boost/math/special_functions/round.hpp>

#include "iv_dynamicobject_msgs/ObjectArray.h"
#include "ukf.h"

using std::vector;

using namespace sensors_fusion;
using namespace visualization_msgs;
using namespace tracking;

tf::TransformListener* listener;
UnscentedKF ukf;

bool is_initialized_ = false;
void showTrackingArrow(ros::Publisher &pub,
                       const ObjectTrackArray& obj_array)
{
  for(int i = 0; i < obj_array.size(); i++){
    visualization_msgs::Marker arrowsG;
    arrowsG.lifetime = ros::Duration(0.2);

    if(obj_array[i].velocity < 0.5)//TODO: maybe add is_static flag member
      continue;

    arrowsG.header.frame_id = "/velo_link";
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
    arrowsG.pose.position.x = obj_array[i].velo_pos.point.x;
    arrowsG.pose.position.y = obj_array[i].velo_pos.point.y;
    arrowsG.pose.position.z = 0.5;

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

bool transformCoordinate(sensors_fusion::ObjectTrackArray& obj_array, double time_stamp)
{
  // Transform objects in camera and world frame
  try{
    for(size_t i = 0; i < obj_array.size(); ++i){
      obj_array[i].velo_pos.header.stamp = ros::Time(time_stamp);
      listener->transformPoint("world",
          obj_array[i].velo_pos,
          obj_array[i].world_pos);
    }
    return true;
  }
  catch(tf::TransformException& ex){
    ROS_ERROR("%s", ex.what());
    return false;
  }
}

void toObjectTrackArray(const iv_dynamicobject_msgs::ObjectArray::ConstPtr& msg,
                         ObjectTrackArray& obj_track_array)
{
  for(size_t i = 0; i < msg->list.size(); ++i){
    ObjectTrack obj;
    obj.length = msg->list[i].length;
    obj.width = msg->list[i].width;
    obj.height = msg->list[i].height;

    obj.velo_pos.header.frame_id = "velo_link";
    obj.velo_pos.point.x = msg->list[i].velo_pose.point.x;
    obj.velo_pos.point.y = msg->list[i].velo_pose.point.y;
    obj.velo_pos.point.z = msg->list[i].velo_pose.point.z;
  }
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

  try{
    listener->transformPoint("velo_link",
        track_msg.world_pos,
        track_msg.velo_pos);
  }
  catch(tf::TransformException& ex){
    ROS_ERROR("Received an exception trying to transform a point from"
        "\"velo_link\" to \"world\": %s", ex.what());
  }

  track_msg.heading = track.sta.x[3];
  track_msg.velocity = track.sta.x[2];
  track_msg.width = track.geo.width;
  track_msg.length = track.geo.length;
  track_msg.height = track.geo.height;
  track_msg.orientation = track.geo.orientation;
  track_msg.object_type = track.sem.name;
  track_msg.confidence = track.sem.confidence;

  track_msg.is_valid = true;

  // Push back track message
  track_array.push_back(track_msg);
}

void detectionCallBack(ros::Publisher &pub, const iv_dynamicobject_msgs::ObjectArray::ConstPtr& msg)
{
  double time_stamp = msg->header.stamp.toSec();

  // Convert to sensors_fusion::ObjectTrackArray
  ObjectTrackArray obj_track_array;
  toObjectTrackArray(msg, obj_track_array);

  // Transform coordinate
  transformCoordinate(obj_track_array, time_stamp);

  // Initialize
  if(is_initialized_){
    ukf.predict(time_stamp);
    ukf.update(obj_track_array[0]);
  }
  else{
      int id = 1;
      ukf.initialize(id, obj_track_array[0], time_stamp);
  }

  sensors_fusion::ObjectTrackArray object_track;
  getObjectTrackArray(object_track, time_stamp);

  showTrackingArrow(pub, object_track);
}

void syncCallback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg,
    const sensor_msgs::ImageConstPtr& image_msg,
    const iv_dynamicobject_msgs::ObjectArrayConstPtr& obj_msg)
{
  ROS_WARN("Enter in syncCallback...");

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tracking_demo_node");
  ros::NodeHandle node;

  ros::Publisher vis_marker_pub = node.advertise<Marker>("/viz/visualization_marker", 1);
  tf::TransformListener tran(ros::Duration(10));
  tf::StampedTransform
  listener = &tran;

//  // Subscriber
//  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_(node, "/kitti_player/hdl64e", 10);
//  message_filters::Subscriber<sensor_msgs::Image> image_sub_(node, "/kitti_player/hdl64e", 10);
//  message_filters::Subscriber<iv_dynamicobject_msgs::ObjectArray> object_array_sub_(node, "/kitti_player/hdl64e", 10);
//  typedef message_filters::sync_policies::ExactTime<sensor_msgs::PointCloud2, sensor_msgs::Image, iv_dynamicobject_msgs::ObjectArray> MySyncPolicy;
//  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), cloud_sub_, image_sub_, object_array_sub_);
//
//  sync.registerCallback(boost::bind(&syncCallback,_1, _2,_3));

  ros::spin();
  return 0;
}
