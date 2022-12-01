from typing import Optional

import numpy as np
import vedo
from scipy.spatial.transform import Rotation
from vedo import Cylinder, Arrow, Line, Sphere

DH = {
    # "theta": (0, 0, 0, 0, 0, 0),
    "a": (0, -.425, -.3922, 0, 0, 0),
    "d": (0.089159, 0, 0, 0.10915, 0.09465, 0.0823),
    "alpha": (np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0)
}


class Joint:
    def __init__(self, a, d, alpha, prev_joint=None):

        self.theta = 0
        self.a = a
        self.d = d
        self.alpha = alpha
        self.prev_joint: Joint = prev_joint
        self.tf = self.__get_tf(0)
        self.abs_tf = self.tf

    @staticmethod
    def zero_joint():
        j = Joint(0, 0, 0)
        j.tf = j.abs_tf = np.identity(4)
        return j

    def __get_tf(self, theta):
        cos = np.cos
        sin = np.sin
        self.theta = theta
        alpha = self.alpha
        d = self.d
        a = self.a

        if self.prev_joint is not None:
            alpha = self.prev_joint.alpha
            a = self.prev_joint.a

        ct = cos(theta)
        ca = cos(alpha)
        st = sin(theta)
        sa = sin(alpha)

        if self.prev_joint is None:  # conventional DH
            tf = np.array([
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1],
            ])
        else:  # optimized DH
            tf = np.array([
                [ct, -st, 0, a],
                [st * ca, ct * ca, -sa, -sa * d],
                [st * sa, ct * sa, ca, ca * d],
                [0, 0, 0, 1],
            ])
        return tf  # np.matmul(tf2_1, tf2_2)

    def move_to(self, theta):
        self.tf = self.__get_tf(theta)
        return self

    def set_abs_tf(self, prev_tf):
        self.abs_tf = np.matmul(prev_tf, self.tf)
        return self.abs_tf

    def transform_vector(self, vec, tf=None):
        tf = self.tf if tf is None else tf
        return tf[:3, 3], np.matmul(tf, [x for x in vec] + [1])[:-1]

    def get_pos(self):
        return self.abs_tf[:-1, 3]


# def rot_matrix_to_axis_angle(m):
#     r = Rotation.from_matrix(m)
#     vec = r.as_rotvec()
#     angle = np.linalg.norm(vec)
#     if angle == 0:
#         axis = (1, 0, 0)
#     else:
#         axis = vec / angle
#     return angle, axis


class UR5:
    DH = {
        # "theta": (0, 0, 0, 0, 0, 0),
        "a": (0, -.425, -.3922, 0, 0, 0),
        "d": (0.089159, 0, 0, 0.10915, 0.09465, 0.0823),
        "alpha": (np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0)
    }

    def __init__(self):
        self.joints = [Joint.zero_joint()]
        DH = self.DH
        for i in range(6):
            j = Joint(DH["a"][i], DH["d"][i], DH["alpha"][i], self.joints[-1])
            self.joints.append(j)
        self.joints = self.joints[1:]

    def set_joint_angles(self, *thetas, rad=False):
        prev: Optional[Joint] = None
        assert len(thetas) == 6, "number of joint angles supplied must be 6"
        if not rad:
            thetas = np.deg2rad(thetas)
        for j, theta in zip(self.joints, thetas):
            j.move_to(theta)
            if prev is None:
                j.set_abs_tf(np.identity(4))
            else:
                j.set_abs_tf(prev.abs_tf)
            prev = j
        print(f"Successfully moved robot to position theta={np.round(np.rad2deg(self.get_joint_angles()), 0)}")

    def calculate_inverse_kinematics(self, target_tf):  # target tf relative to Joint zero transform of robot

        print("POSITIONS")
        positions_org = [j.get_pos() for j in self.joints]
        tf_org = [j.tf for j in self.joints]
        tf_abs_org = [j.abs_tf for j in self.joints]
        for j in self.joints:
            print(j.get_pos())
        print("----")
        d1, d2, d3, d4, d5, d6 = [self.joints[i].d for i in range(6)]
        a1, a2, a3, a4, a5, a6 = [self.joints[i].a for i in range(6)]
        # tf01, tf12, tf23, tf34, tf45, tf56 = [None for _ in range(6)]
        p_06 = target_tf[:, 3]
        p_06x, p_06y, p_06z, _ = p_06
        sin, cos = np.sin, np.cos
        inverted_target_tf = np.linalg.inv(target_tf)

        # theta 1
        p05 = np.matmul(target_tf, [0, 0, -d6, 1])
        p05x, p05y, _, _ = p05
        theta_1 = [
            np.arctan2(p05y, p05x) + sgn * (np.arccos(d4 / np.linalg.norm((p05x, p05y))) + np.pi / 2)
            for sgn in [-1, 1]
        ]

        print(f"theta_1={np.rad2deg(theta_1)} (0.)")

        theta_1 = theta_1[-1]  # todo multiple thetas
        org_tf = self.joints[0].tf
        tf01 = self.joints[0].move_to(theta_1).tf
        print(np.linalg.norm(tf01 - org_tf))

        # theta_5
        theta_5 = [
            sgn * (
                np.arccos((p_06x * sin(theta_1) - p_06y * cos(theta_1) - d4) / d6)
            )
            for sgn in [-1, 1]
        ]
        print(f"theta_5={np.rad2deg(theta_5)} (0.)")

        theta_5 = theta_5[-1]
        org_tf = self.joints[4].tf
        tf45 = self.joints[4].move_to(theta_5).tf
        print(np.linalg.norm(tf45 - org_tf))

        # theta_6

        x_60x, x_60y = inverted_target_tf[:2, 0]
        y_60x, y_60y = inverted_target_tf[:2, 1]

        numerator1 = -x_60y * sin(theta_1) + y_60y * cos(theta_1)
        numerator2 = x_60x * sin(theta_1) - y_60x * cos(theta_1)
        denominator = sin(theta_5)

        if denominator == 0:
            theta_6 = 0  # any angle
        else:
            theta_6 = np.arctan2(numerator1 / denominator, numerator2 / denominator)

        theta_6 = 0
        org_tf = self.joints[5].tf
        tf56 = self.joints[5].move_to(theta_6).tf

        print(f"theta_6={np.rad2deg(theta_6)} (any)")
        print(np.linalg.norm(tf56 - org_tf))

        # theta_3
        tf06 = target_tf
        tf16 = np.matmul(np.linalg.inv(tf01), tf06)
        tf46 = np.matmul(tf45, tf56)
        tf64 = np.linalg.inv(tf46)
        tf14 = np.matmul(tf16, tf64)
        p14 = tf14[:3, 3]
        p14x, _, p14z = p14

        theta_3 = [
            sgn * np.arccos(
                (np.linalg.norm((p14x, p14z)) ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
            )
            for sgn in [-1, 1]
        ]

        print(f"theta_3={np.rad2deg(theta_3)} (90)")

        pass

    def vedo_elements(self):
        amplitude = .05
        x = Arrow((0, 0, 0), (amplitude, 0, 0)).c("red")
        y = Arrow((0, 0, 0), (0, amplitude, 0)).c("green")
        z = Arrow((0, 0, 0), (0, 0, amplitude)).c("blue")
        zs = [z]
        ys = [y]
        xs = [x]
        print("-----------ROBOT-DH---------")
        for i, j in enumerate(self.joints):
            alpha, theta = np.rad2deg((j.alpha, j.theta))
            print(f"JOINT {i + 1}: d={j.d}, a={j.a}, {alpha=}, {theta=}")
            absolute_tf = j.abs_tf
            xs.append(Arrow(*j.transform_vector((amplitude, 0, 0), absolute_tf), c="red"))
            ys.append(Arrow(*j.transform_vector((0, amplitude, 0), absolute_tf), c="green"))
            zs.append(Arrow(*j.transform_vector((0, 0, amplitude), absolute_tf), c="blue"))
        lines = [Line(a.pos(), b.pos()) for a, b in zip(zs, zs[1:])]
        return xs + ys + zs, lines

    def get_joint_angles(self):
        return tuple([j.theta for j in self.joints])


def main2():
    robot = UR5()
    robot.set_joint_angles(0, -90, 90, 0, 0, 0)
    elms = robot.vedo_elements()

    print(robot.joints[-1].abs_tf)

    robot.calculate_inverse_kinematics(robot.joints[-1].abs_tf)

    # robot2 = UR5()
    # robot2.set_joint_angles(0, -90, 90, 0, 0, 0)
    # elms2 = robot2.vedo_elements()

    vedo.show(Sphere(r=.01).wireframe(), elms, axes=1,
              interactive=True)


if __name__ == '__main__':
    main2()
    # viz online: https://robodk.com/robot/Universal-Robots/UR5#View3D
