from Robots import UR5
from typing import List

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

    def shortest_path(self, thetas1, thetas2):
        self.robot.set_joint_angles(thetas1)
        tf1 = self.robot.get_endeffector_transform()
        self.robot.set_joint_angles(thetas2)
        tf2 = self.robot.get_endeffector_transform()
        self.targets = self.slerp([tf1, tf2])

    def targets_to_any_path(self):
        self.path = [self.robot.calculate_inverse_kinematics(tf, True) for tf in self.targets]
        return self.path

    def vedo_elements(self):
        for thetas, singularities in self.targets_to_any_path():
            self.robot.set_joint_angles(thetas)
            yield self.robot.vedo_elements()

    def rrt_star(self):
        pass
