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


class T:
    def __init__(self, matrix_4_4: np.array):
        if type(matrix_4_4) is T:
            self.mat = matrix_4_4.mat
        else:
            self.mat: np.array = matrix_4_4

    def __mul__(self, other):
        return T(np.matmul(self.mat, other.mat))

    def __invert__(self):
        return T(np.linalg.inv(self.mat))

    def rot(self):
        return self.mat[:3, :3]

    def transl(self):
        return self.mat[:3, 3]

    def mul_vec(self, vec):
        if len(vec) == 3:
            vec = list(vec)
            vec.append(1)
        return np.matmul(self.mat, vec)

    def axis_angle(self):
        r = Rotation.from_matrix(self.rot())
        vec = r.as_rotvec()
        angle = np.linalg.norm(vec)
        if angle == 0:
            axis = (1, 0, 0)
        else:
            axis = vec / angle
        return angle, axis

    def __str__(self):
        return str(self.mat)

    def x_rot(self):
        return self.mat[:3, 0]

    def y_rot(self):
        return self.mat[:3, 1]

    def z_rot(self):
        return self.mat[:3, 2]


class Joint:
    def __init__(self, a, d, alpha, prev_joint=None):

        self.theta = 0
        self.a = a
        self.d = d
        self.alpha = alpha
        self.prev_joint: Joint = prev_joint
        self.tf: T = self.__get_tf(0)
        self.abs_tf: T = self.tf

    @staticmethod
    def zero_joint():
        j = Joint(0, 0, 0)
        j.tf = j.abs_tf = T(np.identity(4))
        return j

    def __get_tf(self, theta) -> T:
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
        return T(tf)  # np.matmul(tf2_1, tf2_2)

    def move_to(self, theta):
        assert int(theta) is not True, "theta must be a number, not an iterable"
        self.tf = self.__get_tf(theta)
        return self

    def set_abs_tf(self, prev_tf: T):
        self.abs_tf = prev_tf * self.tf
        return self.abs_tf

    def transform_vector(self, vec, tf: T = None):
        tf = self.tf if tf is None else tf
        return tf.mat[:3, 3], np.matmul(tf.mat, [x for x in vec] + [1])[:-1]

    def get_pos(self):
        return self.abs_tf.transl()


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

    def set_joint_angles(self, *thetas, rad=True):
        prev: Optional[Joint] = None
        assert len(thetas) == 6, "number of joint angles supplied must be 6"
        if not rad:
            thetas = np.deg2rad(thetas)
        for j, theta in zip(self.joints, thetas):
            j.move_to(theta)
            if prev is None:
                j.set_abs_tf(T(np.identity(4)))
            else:
                j.set_abs_tf(T(prev.abs_tf))
            prev = j
        print(f"Successfully moved robot to position theta={np.round(np.rad2deg(self.get_joint_angles()), 0)}")

    @staticmethod
    def __theta_1(target_tf, d4, d6):
        p05 = target_tf.mul_vec([0, 0, -d6, 1])
        p05x, p05y, _, _ = p05
        return [
            np.arctan2(p05y, p05x) + sgn * (np.arccos(d4 / np.linalg.norm((p05x, p05y))) + np.pi / 2)
            for sgn in [-1, 1]
        ]

    @staticmethod
    def __theta_5(target_tf, d4, d6, theta_1):
        p_06 = target_tf.transl()
        p_06x, p_06y, p_06z = p_06
        sin, cos = np.sin, np.cos
        cos_val = (p_06x * sin(theta_1) - p_06y * cos(theta_1) - d4) / d6
        return [
            sgn * (
                np.arccos(cos_val) if -1 <= cos_val <= 1 else np.nan
            )
            for sgn in [-1, 1]
        ]

    @staticmethod
    def __theta_6(inverted_target_tf, theta_1, theta_5):
        sin, cos = np.sin, np.cos

        x_60x, x_60y, _ = inverted_target_tf.x_rot()
        y_60x, y_60y, _ = inverted_target_tf.y_rot()

        numerator1 = -x_60y * sin(theta_1) + y_60y * cos(theta_1)
        numerator2 = x_60x * sin(theta_1) - y_60x * cos(theta_1)
        denominator = sin(theta_5)

        if np.abs(denominator) < 1e-7:
            theta_6 = 0.  # any angle
        else:
            theta_6 = np.arctan2(numerator1 / denominator, numerator2 / denominator)
        return theta_6

    @staticmethod
    def __theta_3(target_tf, tf01, tf45, tf56, a2, a3):
        tf06 = target_tf
        tf16 = ~tf01 * tf06
        tf46 = tf45 * tf56
        tf14 = tf16 * ~tf46
        p14 = tf14.transl()
        p14x, _, p14z = p14
        l_p14xz = np.linalg.norm((p14x, p14z))
        return [
                   sgn * np.arccos(
                       (l_p14xz ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
                   )
                   for sgn in [-1, 1]
               ], tf14, l_p14xz

    @staticmethod
    def __theta_2(a3, tf14: T, l_p14xz, theta_3):
        sin = np.sin
        p14x, p14y, p14z = tf14.transl()
        return np.arctan2(-p14z, -p14x) - np.arcsin(-a3 * sin(theta_3) / l_p14xz)

    @staticmethod
    def __theta_4(tf12, tf23, tf14):
        tf34 = ~(tf12 * tf23) * tf14
        x34x, x34y, _ = tf34.x_rot()
        return np.arctan2(x34y, x34x)

    def calculate_inverse_kinematics(self, target_tf: T,
                                     return_one_solution=False,
                                     rad=True):  # target tf relative to Joint zero transform of robot
        d1, d2, d3, d4, d5, d6 = [self.joints[i].d for i in range(6)]
        a1, a2, a3, a4, a5, a6 = [self.joints[i].a for i in range(6)]
        inverted_target_tf = ~target_tf

        solutions = []

        thetas_1 = self.__theta_1(target_tf, d4, d6)
        for theta_1 in thetas_1:
            if np.isnan(theta_1):
                continue
            tf01 = self.joints[0].move_to(theta_1).tf
            thetas_5 = self.__theta_5(target_tf, d4, d6, theta_1)
            for theta_5 in thetas_5:
                if np.isnan(theta_5):
                    continue
                tf45 = self.joints[4].move_to(theta_5).tf
                theta_6 = self.__theta_6(inverted_target_tf, theta_1, theta_5)
                if np.isnan(theta_6):
                    continue
                tf56 = self.joints[5].move_to(theta_6).tf
                thetas_3, tf14, l_p14xz = self.__theta_3(target_tf, tf01, tf45, tf56, a2, a3)
                for theta_3 in thetas_3:
                    if np.isnan(theta_3):
                        continue
                    tf23 = self.joints[2].move_to(theta_3).tf
                    theta_2 = self.__theta_2(a3, tf14, l_p14xz, theta_3)
                    if np.isnan(theta_2):
                        continue
                    tf12 = self.joints[1].move_to(theta_2).tf
                    theta_4 = self.__theta_4(tf12, tf23, tf14)
                    if np.isnan(theta_4):
                        continue
                    solutions.append([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])
                    if return_one_solution:
                        print(f"found first solution (others skipped): {np.round(np.rad2deg(solutions[0]))}")
                        if not rad:
                            solutions = np.rad2deg(solutions)
                        return solutions[0]

        print(f"found {len(solutions)} solutions:\n{np.round(np.rad2deg(solutions))}")
        if not rad:
            solutions = np.rad2deg(solutions)
        return solutions

    def vedo_elements(self):
        amplitude = .05
        x = Arrow((0, 0, 0), (amplitude, 0, 0)).c("red")
        y = Arrow((0, 0, 0), (0, amplitude, 0)).c("green")
        z = Arrow((0, 0, 0), (0, 0, amplitude)).c("blue")
        zs = [z]
        ys = [y]
        xs = [x]
        print("------VISUALIZING-DH--------")
        for i, j in enumerate(self.joints):
            alpha, theta = np.round(np.rad2deg((j.alpha, j.theta)), 1)
            print(f"JOINT {i + 1}: d={j.d}, a={j.a}, {alpha=}, {theta=}")
            absolute_tf = j.abs_tf
            xs.append(Arrow(*j.transform_vector((amplitude, 0, 0), absolute_tf), c="red"))
            ys.append(Arrow(*j.transform_vector((0, amplitude, 0), absolute_tf), c="green"))
            zs.append(Arrow(*j.transform_vector((0, 0, amplitude), absolute_tf), c="blue"))
        lines = [Line(a.pos(), b.pos()) for a, b in zip(zs, zs[1:])]
        print("----------------------------")
        return xs + ys + zs, lines

    def get_joint_angles(self):
        return tuple([j.theta for j in self.joints])


def main2():
    robot = UR5()
    thetas = (10, -90, 90, 20, 40, 10)
    robot.set_joint_angles(*thetas, rad=False)
    solutions = robot.calculate_inverse_kinematics(robot.joints[-1].abs_tf, rad=False)
    for solution in solutions:
        diff = np.linalg.norm(np.array(solution) - thetas)
        if diff < 1e-6:
            print("Direct and inverse kinematics are matching!")
            break
    else:
        raise AssertionError("Direct and inverse kinematics conflicting!")

    vedo.show(Sphere(r=.01), robot.vedo_elements(), axes=1, interactive=True)


if __name__ == '__main__':
    main2()
    # viz online: https://robodk.com/robot/Universal-Robots/UR5#View3D
