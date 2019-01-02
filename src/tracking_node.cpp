/*======================================================================
* Author   : Haiming Zhang
* Email    : zhanghm_1995@qq.com
* Version  :　2019年1月2日
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
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>

#include <boost/algorithm/string.hpp>
#include <boost/timer.hpp>
#include <boost/math/special_functions/round.hpp>

#include <darknet_ros_msgs/ImageWithBBoxes.h>
#include "iv_dynamicobject_msgs/ObjectArray.h"
#include "ukf.h"


class TrackingProcess
{
public:
  TrackingProcess(ros::NodeHandle& node):
    cloud_sub_(node, "/kitti/velo/pointcloud", 10),
    image_sub_(node, "/darknet_ros/image_with_bboxes", 10),
    object_array_sub_(node, "/detection/object_array", 10),
    sync_(MySyncPolicy(10), cloud_sub_, image_sub_, object_array_sub_)
  {
    sync_.registerCallback(boost::bind(&TrackingProcess::syncCallback, this, _1, _2,_3));
  }

  void syncCallback(const sensor_msgs::PointCloud2ConstPtr& lidar_msg,
      const darknet_ros_msgs::ImageWithBBoxesConstPtr& image_msg,
      const iv_dynamicobject_msgs::ObjectArrayConstPtr& obj_msg)
  {
    ROS_WARN("Enter in syncCallback...");

  }

private:
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
  message_filters::Subscriber<darknet_ros_msgs::ImageWithBBoxes> image_sub_;
  message_filters::Subscriber<iv_dynamicobject_msgs::ObjectArray> object_array_sub_;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
      darknet_ros_msgs::ImageWithBBoxes, iv_dynamicobject_msgs::ObjectArray> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync_;

  tf::TransformListener listener;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "tracking_node");
  ros::NodeHandle node;
  TrackingProcess track_process(node);

  ros::spin();
}
