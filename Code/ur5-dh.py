import numpy as np
import vedo
from vedo import Sphere, Point

from Configurations import ConfigurationSpace
from Robots import UR5


def main2():  # inverse kinematics example
    robot = UR5()
    thetas = (0, -90, -90, 0, 90, 0)
    robot.set_joint_angles(*thetas, rad=False)
    print(f"Direct kinematics to: {np.round(thetas, 1)}")
    solutions, singularities = robot.calculate_inverse_kinematics(robot.joints[-1].abs_tf, rad=False,
                                                                  theta_6_if_singularity=thetas[-1])
    closest, min_diff = None, 1e99
    for i, (solution, is_singularity) in enumerate(zip(solutions, singularities)):
        diff = np.linalg.norm(np.array(solution) - thetas)
        if diff < min_diff:
            closest, min_diff = solution, diff
        if diff < 1e-3:
            print("Direct and inverse kinematics are matching!")
            break

    clones = []
    for i, (solution, is_singularity) in enumerate(zip(solutions, singularities)):
        robot_cpy = UR5()
        print(f"solution {i + 1}: {np.round(solution, 1)} {'' if not is_singularity else '(singularity)'}")
        robot_cpy.set_joint_angles(solution, rad=False)
        xs, ys, zs, lines = robot_cpy.vedo_elements()
        clones.append((xs[-1], ys[-1], zs[-1], lines))
    if min_diff > 1e-3:
        robot_cpy = UR5()
        robot_cpy.set_joint_angles(closest, rad=False)
        vedo.show(Sphere(r=.01), robot.vedo_elements(), robot_cpy.vedo_elements(), axes=1, interactive=True)
        raise AssertionError("Direct and inverse kinematics conflicting!")
    meshes = robot.meshes()
    vedo.show(Sphere(r=.01), clones, meshes,
              [Point(m.center_of_mass(), c="red") for m in meshes],
              axes=1,
              interactive=True)


def main2b():
    thetas = (0, -23, -146, 173, -95, 181)
    robot = UR5()
    robot.set_joint_angles(*thetas, rad=False)

    solutions, _ = robot.calculate_inverse_kinematics(robot.joints[-1].abs_tf, rad=False,
                                                      theta_6_if_singularity=thetas[-1])
    robot2 = UR5()
    elms = []
    for solution in solutions:
        if robot.hitting_itself():
            robot2.set_joint_angles(*solution, rad=False)
            if robot2.hitting_itself():
                continue
        else:
            robot2 = robot
        xs, ys, zs, lines = robot2.vedo_elements()
        elms.append(
            (xs[-1], ys[-1], zs[-1], lines, robot2.meshes(), robot2.vedo_elements(),
             [Point(m.center_of_mass()) for m in robot2.meshes()]))
        print(np.round(np.rad2deg(robot2.get_joint_angles()), 1))
        break
        # if not robot2.hitting_itself():
        #     break
    else:
        raise AssertionError("No valid solution because of collisions")
    vedo.show(elms, Point(robot.joints[-1].abs_tf.transl(), c="blue"), Point((0, 0, 0), c="blue"),
              axes=1,
              interactive=True)


def main3():  # shortest path example
    c = ConfigurationSpace()
    # c.calculate(10)
    robot2 = c.robot
    c.in_obs_space((5, 5, 5, 5, 5, 5), False)
    obs = list(c.obstacle_space)
    for i in range(100):
        robot2.set_joint_angles(obs[np.random.randint(0, len(obs))], rad=False)
        elms = (robot2.meshes(), robot2.vedo_elements(),
                [Point(m.center_of_mass()) for m in robot2.meshes()])
        vedo.show(elms, Point(robot2.joints[-1].abs_tf.transl(), c="blue"), Point((0, 0, 0), c="blue"), axes=1,
                  interactive=True, new=True)
    planner = PlanningModule()
    planner.shortest_path((0, 0, 0, 0, 0, 0), (0, -90, -90, 0, 90, 0))
    vedo.show(Sphere(r=.01), list(planner.vedo_elements()), axes=1, interactive=True)


if __name__ == '__main__':
    main3()
    # viz online: https://robodk.com/robot/Universal-Robots/UR5#View3D
