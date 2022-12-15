from Code.Configurations import ConfigurationSpace
from Code.Transform import T
from Graph import Graph
from Robots import UR5
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PlanningModule:
    target_diff_angles = (5, 5, 5, 5, 5, 5)

    def __init__(self):
        self.robot = UR5()
        self.targets = []
        self.path = []

    def slerp(self, list_of_transforms: List[T], number_of_points: int = 10):
        assert len(list_of_transforms) > 1, "number of transforms supllied must be greater than one"
        times = np.linspace(0, len(list_of_transforms) - 1, number_of_points)
        key_rots = Rotation.from_matrix([t.rot() for t in list_of_transforms])
        slerp = Slerp(list(range(len(list_of_transforms))), key_rots)
        interp_rots = slerp(times)
        delta_transl = [(tf0.transl(), tf1.transl() - tf0.transl()) for tf0, tf1 in
                        zip(list_of_transforms, list_of_transforms[1:])]
        translations = []
        for t in times[:-1]:
            tr0, vec = delta_transl[int(t)]
            dt = t - int(t)
            translations.append(tr0 + vec * dt)
        translations.append(list_of_transforms[-1].transl())
        return [T.from_rotation_and_translation(r, t) for r, t in zip(interp_rots.as_matrix(), translations)]

    @staticmethod
    def partial_translation(tr: np.ndarray, fraction: float) -> np.ndarray:
        return tr * fraction

    def shortest_path(self, thetas1, thetas2, rad=True, number_of_points=10):
        self.robot.set_joint_angles(thetas1, rad=rad)
        tf1 = self.robot.get_endeffector_transform()
        self.robot.set_joint_angles(thetas2, rad=rad)
        tf2 = self.robot.get_endeffector_transform()
        self.targets = self.slerp([tf1, tf2], number_of_points)

        return self.targets

    def first_configuration_for_each_target_tf(self, rad=True):
        self.path = [self.robot.calculate_inverse_kinematics(tf, True)[0] for tf in self.targets]
        if not rad:
            return np.rad2deg(self.path)
        return self.path

    def all_configurations_for_each_target_tf(self, rad=True) -> Tuple[List[List[tuple]], List[tuple]]:
        all_paths = []
        singularities = []
        for i, tf in enumerate(self.targets):
            all_paths.append([])
            for j, (configuration, singularity_found) in enumerate(zip(*self.robot.calculate_inverse_kinematics(tf))):
                if singularity_found:
                    singularities.append((i, j))
                all_paths[-1].append(tuple(configuration))

        if not rad:
            return [[tuple(y) for y in np.rad2deg(x)] for x in all_paths], singularities
        return all_paths, singularities

    def vedo_elements(self):
        for thetas, singularities in self.first_configuration_for_each_target_tf():
            self.robot.set_joint_angles(thetas)
            yield self.robot.vedo_elements()

    def random_tree(self, start, target):
        g = Graph([])
        vertices = [start]
        c = ConfigurationSpace()
        step_size = np.deg2rad(10)
        for i in range(0, 1000):
            x_trg = c.random_configuration()
            x_origin = vertices[np.random.randint(0, len(vertices))]  # for rrt select closest config to x_trg instead

            vec = np.array(x_trg) - x_origin
            vec /= np.linalg.norm(vec)
            _, x_nearest_connecting = c.nearest_free(vec * step_size + x_origin)

            g.add(x_origin, x_nearest_connecting)
            if g.is_connected(start, target):
                break


if __name__ == '__main__':
    m = PlanningModule()
    c = ConfigurationSpace()
    c.nearest_free([-0.13236761, 0.00073756, -0.04037701, 0.01959659, -0.08921494,
                    -0.05446447])
    # m.random_tree(tuple([0 for _ in range(6)]), tuple([np.pi for _ in range(6)]))
    pass
