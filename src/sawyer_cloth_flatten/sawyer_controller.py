#!/usr/bin/env python

import math

import rospy

import actionlib
import geometry_msgs
from geometry_msgs import msg as geometry_msgs
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from std_msgs import msg as std_msgs
from tf import transformations

from robotiq_2f_gripper_control.msg._Robotiq2FGripper_robot_output \
    import Robotiq2FGripper_robot_output as GripperCommand

import intera_interface
from intera_interface import CHECK_VERSION

from sawyer_cloth_flatten.msg import ApplyDisplacementAction, MoveToPositionAction
from sawyer_cloth_flatten.srv import CloseGripper, OpenGripper, ResetSawyer

class SawyerController:
    def __init__(self, joint_state_publish_rate=10.0):
        self.GRIPPER_Z_OFFSET = 0.18
        self.INITIAL_POSITION = (0.45, 0.16, 0.40-self.GRIPPER_Z_OFFSET)

        self._joint_state_publish_rate = joint_state_publish_rate

        self._joint_state_publish_rate_pub = rospy.Publisher(
            'robot/joint_state_publish_rate', std_msgs.UInt16, queue_size=10)
        self._gripper_command_pub = rospy.Publisher(
            'Robotiq2FGripperRobotOutput', GripperCommand, queue_size=10)

        self._close_gripper_service_server = rospy.Service(
            'close_gripper', CloseGripper, self._close_gripper_cb)
        self._open_gripper_service_server = rospy.Service(
            'open_gripper', OpenGripper, self._open_gripper_cb)
        self._open_gripper_service_server = rospy.Service(
            'reset_sawyer', ResetSawyer, self._reset_sawyer_cb)

        self._apply_displacement_action_server = actionlib.SimpleActionServer(
            'apply_displacement', ApplyDisplacementAction, self._apply_displacement_execute_cb, auto_start=False)
        self._move_to_position_action_server = actionlib.SimpleActionServer(
            'move_to_position', MoveToPositionAction, self._move_to_position_execute_cb, auto_start=False)

        self._right_arm = intera_interface.limb.Limb('right')
        self._right_joint_names = self._right_arm.joint_names()

        moveit_commander_args = ['joint_states:=/robot/joint_states']
        moveit_commander.roscpp_initialize(moveit_commander_args)

        self._sawyer = MoveGroupCommander('sawyer')
        self._sawyer.set_num_planning_attempts(2)
        self._sawyer.set_planning_time(10)

        self._set_planning_limit()

        rospy.loginfo('Getting robot state...')
        self._robot = intera_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._robot.state()

        rospy.loginfo('Enabling robot...')
        self._robot.enable()

        self._gripper_command = GripperCommand()
        self._gripper_command.rACT = 1
        self._gripper_command.rGTO = 1
        self._gripper_command.rSP = 255
        self._gripper_command.rFR = 150

        self._joint_state_publish_rate_pub.publish(self._joint_state_publish_rate)

        rospy.sleep(3)
        self.reset()
        self._apply_displacement_action_server.start()
        self._move_to_position_action_server.start()

        rospy.loginfo('Robot ready')

    def _close_gripper_cb(self, request):
        self.close_gripper()
        return ()

    def _open_gripper_cb(self, request):
        self.open_gripper()
        return ()

    def _reset_sawyer_cb(self, request):
        self.reset()
        return ()

    def _apply_displacement_execute_cb(self, goal):
        if self.apply_displacement(goal.displacement):
            self._apply_displacement_action_server.set_succeeded()
        else:
            self._apply_displacement_action_server.set_aborted()
            rospy.logwarn('Aborted goal')

    def _move_to_position_execute_cb(self, goal):
        if self.move_to_position(goal.position):
            self._move_to_position_action_server.set_succeeded()
        else:
            self._move_to_position_action_server.set_aborted()
            rospy.logwarn('Aborted goal')

    def _set_planning_limit(self):
        self._planning_scene = PlanningSceneInterface()

        upper_limit = geometry_msgs.PoseStamped()
        upper_limit.header.frame_id = self._sawyer.get_planning_frame()
        upper_limit.pose.position.x = 1.15
        upper_limit.pose.position.y = 0
        upper_limit.pose.position.z = 0.95
        self._planning_scene.add_box('upper_limit', upper_limit, (4, 4, 0.2))
        rospy.sleep(1)

        lower_limit = geometry_msgs.PoseStamped()
        lower_limit.header.frame_id = self._sawyer.get_planning_frame()
        lower_limit.pose.position.x = 1.15
        lower_limit.pose.position.y = 0
        lower_limit.pose.position.z = -0.45
        self._planning_scene.add_box('lower_limit', lower_limit, (2, 2, 0.65))
        rospy.sleep(0.1)

    def _set_gripper(self, rPR):
        self._gripper_command.rPR = rPR
        self._gripper_command_pub.publish(self._gripper_command)
        rospy.sleep(1)

    def reset(self):
        self.open_gripper()
        self._right_arm.move_to_neutral()
        self.move_to_position(geometry_msgs.Point(*self.INITIAL_POSITION))

    def on_shutdown(self):
        rate = rospy.Rate(self._joint_state_publish_rate)
        self.close_gripper()
        for _ in range(10):
            if rospy.is_shutdown():
                break
            self._right_arm.exit_control_mode()
            self._joint_state_publish_rate_pub.publish(100)
            rate.sleep()

    def close_gripper(self):
        self._set_gripper(255)

    def open_gripper(self):
        self._set_gripper(0)

    def apply_displacement(self, displacement):
        goal_position = self._sawyer.get_current_pose().pose.position
        goal_position.x += displacement.x
        goal_position.y += displacement.y
        goal_position.z += displacement.z - self.GRIPPER_Z_OFFSET

        return self.move_to_position(goal_position)

    def move_to_position(self, goal_position):
        goal_position.z += self.GRIPPER_Z_OFFSET

        quat = transformations.quaternion_from_euler(0, math.pi, 0)
        goal_pose = geometry_msgs.Pose(
            position=goal_position, orientation=geometry_msgs.Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]))

        self._sawyer.set_goal_tolerance(0.001)
        self._sawyer.set_pose_target(goal_pose)

        succeeded = self._sawyer.go(wait=True)
        self._sawyer.stop()
        self._sawyer.clear_pose_targets()

        return succeeded

def main():
    rospy.init_node('sawyer_controller')

    sawyer_controller = SawyerController()
    rospy.on_shutdown(sawyer_controller.on_shutdown)

    rospy.spin()

if __name__ == '__main__':
    main()