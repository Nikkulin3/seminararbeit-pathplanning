import numpy as np
import vedo
from vedo import Sphere, Point

from Code.PathPlanning import PlanningModule
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


class Graph:
    def __init__(self, all_paths, exclude_over_angle_distance=25):

        configs = ConfigurationSpace()
        vertices = {}  # dict of { (configuration_tuple): vertex_index }
        # connections = [] # list of (vertex_i, vertex_j, distance)
        graph = {}

        for node in all_paths:
            for v in node:
                vertices[v] = len(vertices)

        for node, next_node in zip(all_paths, all_paths[1:]):
            for i, config in enumerate(node):
                vertex_i = vertices[config]
                for next_config in next_node:
                    vertex_j = vertices[next_config]
                    if configs.in_obs_space(next_config, rad=False):
                        continue
                    distance = np.array(config) - next_config
                    distance = np.abs(distance)
                    distance = np.min([distance, np.abs(distance - 360)], axis=0)
                    max_dist = np.max(distance)
                    if max_dist > exclude_over_angle_distance:
                        continue
                    if vertex_i not in graph:
                        graph[vertex_i] = {}
                    graph[vertex_i][vertex_j] = np.round(max_dist, 1)
        self.graph = graph
        self.vertices_translation = {v: k for k, v in vertices.items()}

    # A function used by dfs
    def dfs_recursive(self, v, visited):
        visited.add(v)
        path = [v]
        for i in self.graph[v].keys():
            if i not in visited:
                try:
                    dfs = self.dfs_recursive(i, visited)
                    path += dfs if dfs is not None else []
                except KeyError:
                    path += [i]
        return path if path[-1] > list(self.graph.keys())[-1] else None

    def dfs(self):
        visited = set()
        return self.dfs_recursive(list(self.graph.keys())[0], visited)

    def translate(self, vertices_list):
        return [self.vertices_translation[v] for v in vertices_list]


def main3():  # shortest path example
    planner = PlanningModule()
    start = (0, 0, 0, 0, 0, 0)
    planner.shortest_path(start, (0, -90, -90, 0, 90, 0), rad=False, number_of_points=100)
    all_paths, singularities = planner.all_configurations_for_each_target_tf(rad=False)

    g = Graph(all_paths, exclude_over_angle_distance=25)
    path = g.dfs()
    transl = [start] + g.translate(path)

    plt, elms = None, None
    while True:
        plt, elms = planner.robot.animate_configurations(transl, plt=plt, elm=elms, rad=False)
        # plt, elms = planner.robot.animate_configurations(planner.first_configuration_for_each_target_tf(), plt=plt,
        #                                                  elm=elms)
    # plt.interactive()  # freeze on last frame


if __name__ == '__main__':
    main3()
    # viz online: https://robodk.com/robot/Universal-Robots/UR5#View3D
