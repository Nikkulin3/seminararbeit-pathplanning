import numpy as np
import vedo
from vedo import Sphere, Point, Spline

from Code.Graph import Graph
from Code.PathPlanning import PlanningModule
from Robots import UR5


def main2(thetas=(0, -90, -90, 0, 90, 0)):  # inverse kinematics example
    robot = UR5()
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
    # if min_diff > 1e-3:
    #     robot_cpy = UR5()
    #     robot_cpy.set_joint_angles(closest, rad=False)
    #     vedo.show(Sphere(r=.01), robot.vedo_elements(), robot_cpy.vedo_elements(), axes=1, interactive=True)
    #     raise AssertionError("Direct and inverse kinematics conflicting!")
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
    planner = PlanningModule()
    start = (-68, -109, -74, -248, 83, 32)
    end = (-15, -93, -67, -109, 105, 28)
    include_self_collision = True
    include_floor_collision = True
    loose_end_configuration = True
    pts = [[0, 0, 0],
           [0.5, 0.6, 0.8],
           [1, 1, 1]]
    line = Spline(pts, easing="OutCubic", res=100)

    planner.shortest_path([start, end], rad=False, number_of_points=100)


    r1, r2 = UR5(np.deg2rad(start)), UR5(np.deg2rad(end))
    meshes = [*r1.meshes(), *r2.meshes()]
    for m in meshes:
        m.c("blue")
    meshes += [*r1.vedo_elements(), *r2.vedo_elements()]
    # vedo.show(*meshes, axes=1, interactive=True) # comment in to enable preview preview
    try:
        all_paths, singularities = planner.all_configurations_for_each_target_tf()
        g = Graph(all_paths, rad=False, include_self_collision=include_self_collision, include_floor_collision=include_floor_collision)
    except AssertionError as e:
        print(e)
        vedo.show(*meshes, axes=1)
        exit()
    # vedo.show(*meshes, axes=1, interactive=True)
    print(singularities)
    print(start, end)
    print()
    print(np.round(np.rad2deg(all_paths[0])))
    print()
    print(np.round(np.rad2deg(all_paths[-1])))
    print()
    paths, lengths = g.dijkstra(start_config=start, end_config=None if loose_end_configuration else end)
    assert len(paths) > 0
    plt, elms = None, None
    while True:
        print(f"iterating {len(paths)} different movements...")
        for i, path in enumerate(paths):
            print("version",i, lengths[i])
            plt, elms = planner.robot.animate_configurations(g.translate(path)[::2], plt=plt, elm=elms, rad=False,
                                                             extras=meshes)
        # plt, elms = planner.robot.animate_configurations(planner.first_configuration_for_each_target_tf(), plt=plt,
        #                                                  elm=elms)
        # plt.interactive()  # freeze on last frame


if __name__ == '__main__':
    # main2((0, -45, 324, 90, 0, 123))
    main3()
    # viz online: https://robodk.com/robot/Universal-Robots/UR5#View3D
