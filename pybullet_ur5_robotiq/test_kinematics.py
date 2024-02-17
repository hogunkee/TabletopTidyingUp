import numpy as np
import pybullet as p
import pybullet_planning
import kinematics.kinematics as kinematics

def main():
    p.connect(p.GUI)
    jointsActive = [1,2,3,4,5,6]
    box = pybullet_planning.create_box(0.05, 0.05, 0.12)
    obstacles = [box]

    robot = kinematics.LoadURDF("urdf/ur5_robotiq_85.urdf").get_desc()

    startPose = kinematics.InvKinematics(robot).calculate_kinematics(
       [-0.0, -0.02, 0.02]
    )

    endPose = kinematics.InvKinematics(robot).calculate_kinematics(
       [0.15, -0.22, 0.05]
     )

    position_list = np.array(
       [[startPose, endPose],
        [endPose, startPose]]
    )
    print('position')
    print(position_list)

    solution_list = []
    print('motion planning - solution')
    for startPose, endPose in position_list:

        startPose = startPose[jointsActive]
        endPose = endPose[jointsActive]
        solution = kinematics.MotionPlanning(robot, jointsActive).solution(
           startPose,
           endPose,
           obstacles=obstacles
        )
        print(len(solution))

        solution_list.append(solution[0])

    for i in range(len(position_list)):
        del solution_list[i][0]

    print('waypoint')
    for i in range(len(solution_list)):
        waypoint = kinematics.MotionPlanning(robot, jointsActive).get_waypoints(
           solution_list[i]
        )
        print(waypoint)

        #execute_waypoint(waypoint, available_ids)

    #dxl_servo.close()

if __name__ == "__main__":
    main()
