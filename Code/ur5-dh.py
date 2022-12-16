import numpy as np
import vedo
from vedo import Sphere, Point, Spline

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
    def __init__(self, all_paths, exclude_over_angle_distance=1e99, rad=False):
        self.rad = rad
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
                if vertex_i not in graph:
                    graph[vertex_i] = {}
                # added = {}
                alt = {}
                for next_config in next_node:
                    vertex_j = vertices[next_config]
                    if configs.in_obs_space(next_config, rad=rad):
                        continue
                    distance = np.array(config) - next_config
                    distance = np.abs(distance)
                    distance = np.min([distance, np.abs(distance - 360)], axis=0)
                    max_dist = np.max(distance)
                    if max_dist > exclude_over_angle_distance:
                        alt[vertex_j] = max_dist
                        continue
                    graph[vertex_i][vertex_j] = float(max_dist)
                    # added[vertex_j] = float(max_dist)
                if len(graph[vertex_i]) == 0:
                    assert len(alt) > 0, "some path configuration is always in obstacle space"
                    print(f"WARNING: no connection after node {node}. guessing...")
                    best_option = np.argmin(list(alt.values()))
                    graph[vertex_i][list(alt.keys())[best_option]] = list(alt.values())[best_option]
        for config in all_paths[-1]:
            k = vertices[config]
            graph[k] = {}

        self.graph = graph
        self.vertices_translation = {v: k for k, v in vertices.items() if v in self.graph}
        self.node_translation = {k: v for v, k in self.vertices_translation.items()}

    def dijkstra(self, start_config, end_config):

        _360 = 360 if not self.rad else 2 * np.pi

        def norm_cfg(_cfg):
            _cfg = np.array(_cfg) % 360
            return [x if x <= _360 / 2 else x - _360 for x in _cfg]

        possible_starts = []
        possible_ends = []
        for cfg in list(self.node_translation.keys())[:10]:
            delta = np.abs(norm_cfg(np.array(cfg) - start_config))
            # print(np.linalg.norm(delta), list(delta))
            if np.linalg.norm(delta) < 1e-3:
                possible_starts.append(cfg)
        # print()
        for cfg in list(self.node_translation.keys())[-10:]:
            delta = np.abs(norm_cfg(np.array(cfg) - start_config))
            # print(np.linalg.norm(delta), list(delta))
            if np.linalg.norm(delta) < 1e-3:
                possible_ends.append(cfg)
        assert len(possible_starts) > 0, f"unable to find {start_config=} in graph"
        assert len(possible_ends) > 0, f"unable to find {end_config=} in graph"
        paths = []
        path_lengths = []

        def get_path(connections: dict):
            k = list(connections.keys())[-1]
            pth = []
            while True:
                pth.append(k)
                try:
                    k = connections[k]
                except KeyError:
                    break
            pth.reverse()
            return pth

        for cfg in possible_starts:
            current = self.node_translation[cfg]
            distances = self.trim_unreachable_nodes(current)
            unvisited = {node: None for node in distances.keys()}
            visited = {}
            shortest_connection_from = {}
            current_distance = 0
            unvisited[current] = current_distance
            while True:
                for neighbour, distance in distances[current].items():
                    if neighbour not in unvisited:
                        continue
                    new_distance = current_distance + distance
                    if unvisited[neighbour] is None or unvisited[neighbour] > new_distance:
                        unvisited[neighbour] = new_distance
                        shortest_connection_from[neighbour] = current
                visited[current] = current_distance
                del unvisited[current]
                if len(unvisited) == 0:
                    break
                candidates = [node for node in unvisited.items() if node[1] is not None]
                current, current_distance = sorted(candidates, key=lambda x: x[1])[0]
            for end_cfg in possible_ends:
                end_node = self.node_translation[end_cfg]
                if end_node in visited:
                    paths.append(get_path(shortest_connection_from))
                    path_lengths.append(visited[end_node])
                    break
        print(f"found: {path_lengths=}")
        return paths, path_lengths

    def translate(self, vertices_list):
        return [self.vertices_translation[v] for v in vertices_list]

    def trim_unreachable_nodes(self, start_node):
        queue = []
        graph = {k: v for k, v in self.graph.items()}
        visited = set()

        # Add the start node to the queue and visited set
        queue.append(start_node)
        visited.add(start_node)

        # Perform breadth-first search
        while queue:
            node = queue.pop(0)

            # Add its neighbors to the queue and visited set
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

        # Trim the graph by removing all unreachable nodes
        for node in list(graph.keys()):
            if node not in visited:
                del graph[node]
            else:
                for neighbor in list(graph[node].keys()):
                    if neighbor not in visited:
                        del graph[node][neighbor]
        return graph


def main3():  # shortest path example
    planner = PlanningModule()
    start = (180, 0, -90, 0, 0, 0)
    end = (0, -90, -90, 0, 90, 0)

    pts = [[0, 0, 0],
           [0.5, 0.6, 0.8],
           [1, 1, 1]]
    line = Spline(pts, easing="OutCubic", res=100)

    planner.shortest_path([start, end], rad=False, number_of_points=100)
    all_paths, singularities = planner.all_configurations_for_each_target_tf(rad=False)

    g = Graph(all_paths, exclude_over_angle_distance=180, rad=False)
    paths, lengths = g.dijkstra(start_config=start, end_config=end)

    plt, elms = None, None
    while True:
        for path in paths:
            transl = g.translate(path)
            plt, elms = planner.robot.animate_configurations(transl, plt=plt, elm=elms, rad=False)
            plt.add(line)
        # plt, elms = planner.robot.animate_configurations(planner.first_configuration_for_each_target_tf(), plt=plt,
        #                                                  elm=elms)
    # plt.interactive()  # freeze on last frame


if __name__ == '__main__':
    main3()
    # viz online: https://robodk.com/robot/Universal-Robots/UR5#View3D
