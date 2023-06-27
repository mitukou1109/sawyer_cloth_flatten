cmake_minimum_required(VERSION 3.0.2)
project(sawyer_cloth_flatten)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  actionlib
  dynamic_reconfigure
  geometry_msgs
  message_generation
)

catkin_python_setup()

add_action_files(FILES
  ApplyDisplacement.action
  MoveToPosition.action
  PickAndMoveCloth.action
)

add_service_files(FILES
  CloseGripper.srv
  Get3DPointFromPixel.srv
  GetHighestPositionOfCloth.srv
  OpenGripper.srv
  ResetSawyer.srv
)

generate_messages(
  DEPENDENCIES
    actionlib_msgs
    geometry_msgs
)

generate_dynamic_reconfigure_options(
  cfg/RealSenseHandler.cfg
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rospy
    actionlib
    dynamic_reconfigure
    message_runtime
)

catkin_install_python(
  PROGRAMS
    scripts/sawyer_cloth_flatten
    scripts/sawyer_controller_node
    scripts/realsense_handler_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)