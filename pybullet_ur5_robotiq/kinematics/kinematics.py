import pybullet as p
import pybullet_data
import pybullet_planning

class LoadURDF:
    """ Load urdf description of the robot arm """

    def __init__(self, robot_urdf):

        self.robot_urdf = robot_urdf
        self.robot_desc = p.loadURDF(self.robot_urdf, [0, 0, 0])

    def get_desc(self):
        """ Return description of the robot from loaded urdf file """

        return self.robot_desc

class InvKinematics:
    """ Inverse kinematics using Pybullet library
        Get robot description as a init parameter """

    def __init__(self, robot_desc):

        self.robot_desc = robot_desc

        arm_joints = p.getNumJoints(self.robot_desc)
        self.arm_joints = arm_joints - 1

    def calculate_kinematics(self, get_position):
        """ Get position in meters [0, 0, 0] and return solution """

        inverse_kinematics = p.calculateInverseKinematics(
           self.robot_desc,
           self.arm_joints,
           get_position)

        return inverse_kinematics

class MotionPlanning:
    """ Motion planning Class using Pybullet-Planning library
        Get as a parameter robot desciprion and active joints as a list"""

    def __init__(self, robot_desc, jointsActive):

        self.robot_desc = robot_desc
        self.jointsActive = jointsActive

    def solution(self, startPose, endPose, obstacles):
        """ Get start pose and end pose in format [0,0,0] and
            obstacles list and return motion planing solution as a tuple"""

        self.obstacles = obstacles

        # joints limits
        limits=pybullet_planning.get_custom_limits(
           self.robot_desc,
           self.jointsActive,
           circular_limits=pybullet_planning.CIRCULAR_LIMITS)

        joints = pybullet_planning.get_joints(self.robot_desc)

        sample_fn = pybullet_planning.get_sample_fn(
           self.robot_desc,
           self.jointsActive,
           custom_limits=limits)

        extend_fn = pybullet_planning.get_extend_fn(
            self.robot_desc,
            self.jointsActive,
            resolutions=None)

        collision_fn = pybullet_planning.get_collision_fn(
           self.robot_desc,
           self.jointsActive,
           obstacles=self.obstacles,
           custom_limits=limits)

        solution = pybullet_planning.lazy_prm(
           start=startPose,
           goal=endPose,
           sample_fn=sample_fn,
           extend_fn=extend_fn,
           collision_fn=collision_fn)

        return solution

    def get_waypoints(self, solution):
        """ Get as a parameter solution from MotionPlanning and return
            waypoint list that we can use directly to drive arm """

        self.solution = solution

        waypoint = pybullet_planning.waypoints_from_path(self.solution)

        return waypoint
