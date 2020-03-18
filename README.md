# Tracking_demo_node
This is a ROS package for validate UKF tracking performance in 3D object.

## Introduction


## Dependencies
This ROS package need following 3rd ROS packages for compiling and runnning:

- kitti_tracking_player
- iv_dynamicobject_msgs
- darknet_ros_msgs

## Features
- ROS Messages synchronization
- Demonstration of parsing KITTI Raw dataset `tracklet_labels.xml` file, and publish related ROS topics


## How to run it?
```
roslaunch tracking_demo_node tracking_demo_node.launch
```

## Details
1. First sync all needed messages
2. Convert `iv_dynamicobject_msgs::ObjectArray` message to ObjectTrackArray type by using `toObjectTrackArray` function;
3. `transformCoordinate` function uses tf transfrom to convert detected object `velo_link` coordinate to `world` coordinate;
4. **Tracking processes are handled in world coordinate**;
5. if the first track, initialize the ukf;
6. then using `predict` and `update` functions to update target object states.

## Bug reporting
Please use github's issue tracker or email me use `zhanghm_1995@qq.com` to report bugs.