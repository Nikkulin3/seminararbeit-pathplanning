from Code.Configurations import ConfigurationSpace
from Code.Transform import T
from Robots import UR5
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PlanningModule:
    target_diff_angles = (5, 5, 5, 5, 5, 5)

    def __init__(self):
        self.thetas_list = None
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

    def shortest_path(self, thetas_list, rad=True, number_of_points=10):
        tfs = []
        for thetas in thetas_list:
            self.robot.set_joint_angles(thetas, rad=rad)
            tfs.append(self.robot.get_endeffector_transform())
        self.targets = [tfs[0]] + self.slerp(tfs, number_of_points) + [tfs[-1]]
        if not rad:
            thetas_list = np.deg2rad(thetas_list)
        self.thetas_list = [t + np.random.rand() * 1e-10 for t in thetas_list]
        return self.targets

    def first_configuration_for_each_target_tf(self, rad=True):
        self.path = [self.robot.calculate_inverse_kinematics(tf, True)[0] for tf in self.targets]
        if not rad:
            return np.rad2deg(self.path)
        return self.path

    def all_configurations_for_each_target_tf(self, rad=True) -> Tuple[List[List[tuple]], List[tuple]]:
        all_paths = []
        singularities = []
        t6_0 = self.thetas_list[0][5]
        t6_1 = self.thetas_list[-1][5]
        t6_delta = (t6_1 - t6_0)
        for i, tf in enumerate(self.targets):
            all_paths.append([])
            t6 = t6_0 + (i + 1) / len(self.targets) * t6_delta
            for j, (configuration, singularity_found) in enumerate(
                    zip(*self.robot.calculate_inverse_kinematics(tf, theta_6_if_singularity=t6, rad=True))):
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


if __name__ == '__main__':
    m = PlanningModule()
    c = ConfigurationSpace()
    c.nearest_free([-0.13236761, 0.00073756, -0.04037701, 0.01959659, -0.08921494,
                    -0.05446447])
    # m.random_tree(tuple([0 for _ in range(6)]), tuple([np.pi for _ in range(6)]))
    pass
