import numpy as np
import vedo
from vedo import Sphere, Point, Spline

from Code.PathPlanning import PlanningModule
from Configurations import ConfigurationSpace
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


class Graph:
    def __init__(self, all_paths, exclude_over_angle_distance=np.pi / 4, rad=False, include_floor_collision=False, include_self_collision=True):
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
                    if include_self_collision and configs.in_obs_space(next_config, rad=rad):
                        continue
                    if include_floor_collision and configs.floor_collision(next_config, rad=rad, floor_height=0):
                        continue
                    distance = self.config_compare(config, next_config)
                    max_dist = np.max(distance)
                    if max_dist > exclude_over_angle_distance:
                        alt[vertex_j] = max_dist
                        continue
                    graph[vertex_i][vertex_j] = np.sum(distance)
                if len(graph[vertex_i]) == 0:
                    assert len(alt) > 0, "some path configuration is always in obstacle space"
                    print(f"WARNING: no connection after node {vertex_i}. guessing...")
                    best_option = np.argmin(list(alt.values()))
                    graph[vertex_i][list(alt.keys())[best_option]] = list(alt.values())[best_option]
        for config in all_paths[-1]:
            k = vertices[config]
            graph[k] = {}

        self.graph = graph
        self.vertices_translation = {v: k for k, v in vertices.items() if v in self.graph}
        self.node_translation = {k: v for v, k in self.vertices_translation.items()}

    @staticmethod
    def config_compare(cfg1, cfg2):
        def norm_cfg(_cfg):
            _cfg = np.array(_cfg) % (2 * np.pi)
            return [x if x <= np.pi else x - (2 * np.pi) for x in _cfg]

        return np.abs(norm_cfg(np.array(cfg1) - cfg2))

    def dijkstra(self, start_config, end_config=None):

        if not self.rad:
            start_config = np.deg2rad(start_config)

        _360 = 2 * np.pi
        _180 = np.pi

        # def norm_cfg(_cfg):
        #     _cfg = np.array(_cfg) % _360
        #     return [x if x <= _180 else x - _360 for x in _cfg]

        def match_config(to_match, samples):
            out = []
            for _cfg in samples:
                _delta = self.config_compare(_cfg, to_match)  # np.abs(norm_cfg(np.array(_cfg) - to_match))
                if np.linalg.norm(_delta) < 1e-3:
                    out.append(_cfg)
            return out

        possible_starts = match_config(start_config, list(self.node_translation.keys())[:8])
        if end_config is None:
            possible_ends = [self.vertices_translation[i] for i in self.graph.keys() if len(self.graph[i]) == 0]
        else:
            end_config = np.deg2rad(end_config)
            possible_ends = match_config(end_config, list(self.node_translation.keys())[-8:])
        assert len(possible_starts) > 0, f"unable to find {start_config=} in graph"
        assert len(possible_ends) > 0, f"unable to find {end_config=} in graph"
        paths = []
        path_lengths = []

        def get_path(connections: dict, start_at: int):
            k = start_at
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
                    paths.append(get_path(shortest_connection_from, end_node))
                    path_lengths.append(visited[end_node])
                    break
        path_lengths = np.array(path_lengths) / (2 * np.pi)
        print(f"found: {path_lengths=} (in 360deg rotations)")
        return paths, path_lengths

    def translate(self, vertices_list):
        tr = [self.vertices_translation[v] for v in vertices_list]
        if self.rad:
            return tr
        return np.rad2deg(tr)

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
    start = (0, -45, 10, 180, 90, 0)
    end = (0, -90, -10, 180, 90, 0)

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
    try:
        all_paths, singularities = planner.all_configurations_for_each_target_tf()
        g = Graph(all_paths, rad=False, include_self_collision=False)
    except AssertionError as e:
        print(e)
        vedo.show(*meshes, axes=1)
        exit()

    print(start, end)
    print()
    print(np.round(np.rad2deg(all_paths[0])))
    print()
    print(np.round(np.rad2deg(all_paths[-1])))
    print()
    paths, lengths = g.dijkstra(start_config=start, end_config=None)
    assert len(paths) > 0
    plt, elms = None, None
    while True:
        for i, path in enumerate(paths):
            plt, elms = planner.robot.animate_configurations(g.translate(path), plt=plt, elm=elms, rad=False,
                                                             extras=meshes)
        # plt, elms = planner.robot.animate_configurations(planner.first_configuration_for_each_target_tf(), plt=plt,
        #                                                  elm=elms)
        plt.interactive()  # freeze on last frame


if __name__ == '__main__':
    # main2((0, -45, 324, 90, 0, 123))
    main3()
    # viz online: https://robodk.com/robot/Universal-Robots/UR5#View3D
