import numpy as np
import vedo
from vedo import Sphere, Point, Spline

from Code.Configurations import ConfigurationSpace
from Code.Graph import Graph
from Code.PathPlanning import PlanningModule
from Robots import UR5


def inverse_kinematics_example(thetas=(0, -90, -90, 0, 90, 0)):  # inverse kinematics example
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


def direct_path_example(start=None, end=None):
    planner = PlanningModule()
    configs = ConfigurationSpace()
    if start is None:
        start = (-68, -109, -74, -248, 83, 32)
    if end is None:
        end = (-15, -93, -67, -109, 105, 28)

    include_wall_collision = True
    loose_end_configuration = True

    r1, r2 = UR5(np.deg2rad(start)), UR5(np.deg2rad(end))
    meshes = [*r1.meshes(), *r2.meshes()]
    for m in meshes:
        m.c("blue")
    meshes += [*r1.vedo_elements(), *r2.vedo_elements()]
    paths = planner.direct_path(start, end, rad=False, number_of_points=100,
                                loose_end_configuration=loose_end_configuration)
    valid_paths = [p for p in paths if
                   configs.is_valid_path(p, rad=False, include_wall_collision=include_wall_collision)]

    print(f"all paths: {len(paths)}, valid paths: {len(valid_paths)}")

    pos = r1.get_endeffector_transform().transl()
    print(pos)
    if len(valid_paths) == 0:
        print("WARNING no valid paths, displaying 15 shortest paths")
        valid_paths = [p for p in paths if configs.is_valid_path(p, include_wall_collision=False, rad=False)]
        vedo.show(*meshes, axes=1, interactive=True)
    valid_paths.sort(key=lambda x: planner.path_length(x, rad=False))
    plt, elms = None, None
    while True:
        print(f"iterating {len(valid_paths)} different movements...")
        for i, path in enumerate(valid_paths):
            print("version", i, "length", planner.path_length(path, rad=False))
            plt, elms = planner.robot.animate_configurations(path, plt=plt, nth=2,
                                                             elm=elms, rad=False, extras=meshes)


def shortest_path_example(start=None, end=None, prev_plt=None):  # shortest path example
    planner = PlanningModule()
    configs = ConfigurationSpace()
    robot = UR5()
    include_self_collision = True
    include_wall_collision = True
    loose_end_configuration = True
    constraints = np.rad2deg(robot.ANGLE_CONSTRAINTS)
    angle_ranges = [c2 - c1 for c1, c2 in constraints]
    lower_limit = [c1 for c1, _ in constraints]
    while start is None or not configs.is_valid_path([start], rad=False, include_wall_collision=include_wall_collision):
        start = np.random.random(6) * angle_ranges + lower_limit  # (-68.1, -109.1, -74.1, -248.1, 83.1, 32.1)
    while end is None or not configs.is_valid_path([end], rad=False, include_wall_collision=include_wall_collision):
        end = np.random.random(6) * angle_ranges + lower_limit  # (-15, -93, -67, -109, 105, 28)

    planner.shortest_path([start, end], rad=False, number_of_points=100)

    r1, r2 = UR5(np.deg2rad(start)), UR5(np.deg2rad(end))
    meshes = [*r1.meshes(), *r2.meshes()]
    for m in meshes:
        m.c("blue")
    meshes += [*r1.vedo_elements(), *r2.vedo_elements()]
    # vedo.show(*meshes, axes=1, interactive=True) # comment in to enable preview
    try:
        all_paths, singularities = planner.all_configurations_for_each_target_tf()
        g = Graph(all_paths, rad=False, include_self_collision=include_self_collision,
                  include_wall_collision=include_wall_collision, configuration_space=configs)
        paths, _ = g.dijkstra(start_config=start, end_config=None if loose_end_configuration else end)
    except AssertionError as e:
        print(e)
        print("unable to calculate shortest path, trying direct path instead...")
        direct_path_example(start, end)
        return

    paths = [g.translate(p) for p in paths if planner.max_velocity_step(p) < 10.]
    paths = [paths[1]]
    assert len(paths) > 0

    if prev_plt is None:
        plt, elms = None, None
    else:
        plt, elms = prev_plt

    print(f"iterating {len(paths)} different movements...")
    for i, path in enumerate(paths):
        print("version", i)
        plt, elms = planner.robot.animate_configurations(path, nth=2, plt=plt, elm=elms, rad=False, extras=meshes)
    return plt, elms


if __name__ == '__main__':
    # inverse_kinematics_example((0, -45, 324, 90, 0, 123))
    # direct_path_example()
    plt = None
    while True:
        plt = shortest_path_example(prev_plt=plt)
    # viz online: https://robodk.com/robot/Universal-Robots/UR5#View3D
