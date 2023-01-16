import time
from typing import Optional, List, Tuple, Union

import numpy as np
import vedo
from vedo import Cylinder, Arrow, Line, Sphere, Mesh, Point

from Transform import T


class Joint:
    def __init__(self, a, d, alpha, prev_joint=None):

        self.theta = 0
        self.a = a
        self.d = d
        self.alpha = alpha
        self.prev_joint: Joint = prev_joint
        self.tf: T = self.__get_tf(0)
        self.abs_tf: Optional[T] = None

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
        assert prev_tf is not None, "previous transforms have not been set, run set_joint_angles or use target_tf"
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
        "d": (0.089159, 0, 0, 0.10915, 0.09465, 0.0823 + .126),
        "alpha": (np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0)
    }

    ANGLE_CONSTRAINTS = [
        (np.deg2rad(-90), np.deg2rad(90)),
        # (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
    ]

    def __init__(self, thetas_rad=None):
        self.joints = [Joint.zero_joint()]
        DH = self.DH
        for i in range(6):
            j = Joint(DH["a"][i], DH["d"][i], DH["alpha"][i], self.joints[-1])
            self.joints.append(j)
        self.joints = self.joints[1:]
        self.vedo_e = None
        self.vedo_meshes = None
        self.color = np.random.random(3)
        if thetas_rad is not None:
            self.set_joint_angles(thetas_rad)

    def within_joint_constraints(self, thetas, rad=True):
        if not rad:
            thetas = np.deg2rad(thetas)
        for t, (c1, c2) in zip(thetas, self.ANGLE_CONSTRAINTS):
            if not (c1 <= t <= c2):
                assert np.max([c1 - t, t - c2]) < 4 * np.pi
                return False
        return True

    def set_joint_angles(self, *thetas, rad=True):
        prev: Optional[Joint] = None
        assert len(thetas) == 6 or (
                len(thetas) == 1 and len(thetas[0]) == 6), "number of joint angles supplied must be 6"
        if len(thetas) == 1:
            thetas = thetas[0]
        if not rad:
            thetas = np.deg2rad(thetas)
        for j, theta in zip(self.joints, thetas):
            assert np.deg2rad(-360) <= theta <= np.deg2rad(360)
            j.move_to(theta)
            if prev is None:
                j.set_abs_tf(T(np.identity(4)))
            else:
                j.set_abs_tf(T(prev.abs_tf))
            prev = j
        self.vedo_e, self.vedo_meshes = None, None
        # print(f"Successfully moved robot to position theta={np.round(np.rad2deg(self.get_joint_angles()), 0)}")

    @staticmethod
    def __theta_1(target_tf, d4, d6):
        p05 = target_tf.transl() - d6 * target_tf.z_rot()
        p05x, p05y, _ = p05
        p05xy = [p05x, p05y]
        a_tan = np.arctan2(p05y, p05x)
        a_cos = d4 / np.linalg.norm(p05xy)
        if not (-1 <= a_cos <= 1):
            return [np.nan]
        a_cos = np.arccos(a_cos)
        th1 = [
            a_tan + sgn * a_cos + np.pi / 2
            for sgn in [-1, 1]
        ]
        return th1

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
    def __theta_6(inverted_target_tf, theta_1, theta_5, theta_6_if_singularity=0.):
        sin, cos = np.sin, np.cos

        x_60x, x_60y, _ = inverted_target_tf.x_rot()
        y_60x, y_60y, _ = inverted_target_tf.y_rot()

        numerator1 = -x_60y * sin(theta_1) + y_60y * cos(theta_1)
        numerator2 = x_60x * sin(theta_1) - y_60x * cos(theta_1)
        denominator = sin(theta_5)

        is_singularity = np.abs(denominator) < 1e-7
        if is_singularity:
            theta_6 = theta_6_if_singularity  # any angle
        else:
            theta_6 = np.arctan2(numerator1 / denominator, numerator2 / denominator)
        return theta_6, is_singularity

    @staticmethod
    def __theta_3(target_tf, tf01, tf45, tf56, a2, a3):
        tf06 = target_tf
        tf16 = ~tf01 * tf06
        tf46 = tf45 * tf56
        tf14 = tf16 * ~tf46
        p14 = tf14.transl()
        p14x, _, p14z = p14
        l_p14xz = np.linalg.norm((p14x, p14z))
        arc_cos = (l_p14xz ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
        if not (-1 <= arc_cos <= 1):
            return [np.nan], tf14, l_p14xz

        return [
            sgn * np.arccos(arc_cos) for sgn in [-1, 1]
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
                                     rad=True,
                                     theta_6_if_singularity=0.) -> Union[
        Tuple[tuple, bool], Tuple[List[tuple], List[bool]]]:  # target tf relative to Joint zero transform of robot
        d1, d2, d3, d4, d5, d6 = [self.joints[i].d for i in range(6)]
        a1, a2, a3, a4, a5, a6 = [self.joints[i].a for i in range(6)]
        inverted_target_tf = ~target_tf
        if not rad:
            theta_6_if_singularity = np.deg2rad(theta_6_if_singularity)

        solutions = []
        singularities = []
        self.vedo_e, self.vedo_meshes = None, None
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
                theta_6, is_singularity = self.__theta_6(inverted_target_tf, theta_1, theta_5, theta_6_if_singularity)
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
                    singularities.append(is_singularity)
                    if return_one_solution:
                        if not rad:
                            solutions = np.rad2deg(solutions)
                        return tuple(solutions[0]), singularities[0]

        if not rad:
            solutions = np.rad2deg(solutions)
        assert len(solutions) > 0, "unable to calculate any solution (out of reach)"
        return solutions, singularities

    def mesh_definition(self):
        d1, d2, d3, d4, d5, d6 = [self.joints[i].d for i in range(6)]
        xs, ys, zs, _ = self.vedo_elements()
        dx = [(x.top - x.base) for x in xs]
        dz = [(z.top - z.base) for z in zs]
        dz = [z / np.linalg.norm(z) for z in dz]
        shoulder_width = d4 / 2 - .01
        shoulder_dist = d4 * 1.2
        underarm_width = .75 * shoulder_width
        wrist_width = d4
        links = [
            xs[0].base,
            xs[1].base,
            xs[1].base + dz[2] * shoulder_dist,
            xs[3].base + dz[2] * shoulder_dist,
            xs[3].base,
            xs[4].base - dz[4] * wrist_width
        ]
        return [
            {"base": links[0], "top": links[1], "radius": shoulder_width},
            {"base": links[1], "top": links[2], "radius": shoulder_width},
            {"base": links[2], "top": links[3], "radius": shoulder_width},
            {"base": links[3], "top": links[4], "radius": underarm_width},
            {"base": links[4], "top": links[5], "radius": underarm_width},
            {"base": links[5], "top": xs[4].base, "radius": underarm_width},
            {"base": xs[4].base, "top": xs[5].base, "radius": underarm_width},
            {"base": xs[5].base, "top": xs[6].base, "radius": underarm_width},
        ]

    def spheres_equivalent(self):
        mesh_def = self.mesh_definition()
        out = []
        for i, x in enumerate(mesh_def):
            base, top, radius = x["base"], x["top"], x["radius"]
            sphere_distance = 1 * radius
            point = base.copy()
            vec = top - base
            height, cyl_height = 0, np.linalg.norm(vec)
            vec = vec / cyl_height * sphere_distance
            while height < cyl_height - sphere_distance * 1.5:
                point += vec
                height += sphere_distance
                out.append((point.copy(), radius))

            if i < len(mesh_def) - 1:
                out.append((top.copy(), radius))
        return out

    def mesh_spheres(self):
        out = []
        for x, r in self.spheres_equivalent():
            out.append(Sphere(pos=x, r=r, c="blue").opacity(.2))
            out.append(Point(pos=x, c="red"))
        return out

    def meshes(self) -> List[Mesh]:
        if self.vedo_meshes is not None:
            return self.vedo_meshes

        def c(b, t, d):
            return Cylinder(pos=[b, t], r=d, c="grey").opacity(.2).wireframe()

        def s(x, r):
            return Sphere(pos=x, r=r, c="grey").opacity(.2).wireframe()

        mesh_def = self.mesh_definition()
        self.vedo_meshes = []
        for i, x in enumerate(mesh_def):
            self.vedo_meshes.append(c(x["base"], x["top"], x["radius"]))
            if i < len(mesh_def) - 1:
                self.vedo_meshes.append(s(x["top"], x["radius"]))
        return self.vedo_meshes

    def hitting_itself(self):
        spheres = self.spheres_equivalent()

        def intersect(o1, o2):
            p1, r1 = o1
            p2, r2 = o2
            center_distance = np.linalg.norm(np.array(p1) - p2)
            intersecting_distance = r1 + r2
            return center_distance <= intersecting_distance

        for i, sphere1 in enumerate(spheres):
            for sphere2 in spheres[i + 4:]:
                if intersect(sphere1, sphere2):
                    return True
        return False

    def vedo_elements(self):
        if self.vedo_e is not None:
            return self.vedo_e
        amplitude = .05
        x = Arrow((0, 0, 0), (amplitude, 0, 0)).c("red")
        y = Arrow((0, 0, 0), (0, amplitude, 0)).c("green")
        z = Arrow((0, 0, 0), (0, 0, amplitude)).c("blue")
        zs = [z]
        ys = [y]
        xs = [x]
        # print("------VISUALIZING-DH--------")
        for i, j in enumerate(self.joints):
            #     alpha, theta = np.round(np.rad2deg((j.alpha, j.theta)), 1)
            #     print(f"JOINT {i + 1}: d={j.d}, a={j.a}, {alpha=}, {theta=}")
            absolute_tf = j.abs_tf
            xs.append(Arrow(*j.transform_vector((amplitude, 0, 0), absolute_tf), c="red"))
            ys.append(Arrow(*j.transform_vector((0, amplitude, 0), absolute_tf), c="green"))
            zs.append(Arrow(*j.transform_vector((0, 0, amplitude), absolute_tf), c="blue"))
        # print("----------------------------")
        lines = [Line(a.pos(), b.pos(), c=tuple(self.color), lw=5) for a, b in zip(zs, zs[1:])]
        self.vedo_e = xs, ys, zs, lines
        return self.vedo_e

    def get_joint_angles(self):
        return tuple([j.theta for j in self.joints])

    def get_joint_positions(self):
        return np.array([j.get_pos() for j in self.joints])

    def get_endeffector_transform(self):
        return self.joints[-1].abs_tf

    def animate_configurations(self, list_of_configs, nth=None, rad=True, plt=None, elm=None, extras=None) -> Tuple[
        vedo.Plotter, List[vedo.BaseActor]]:
        list_of_configs = [list_of_configs[0] for _ in range(10)] + \
                          list(list_of_configs[::nth]) + [list_of_configs[-1]]
        if extras is None:
            extras = []
        points = []
        self.set_joint_angles(*self.get_joint_angles(), rad=rad)
        if elm is None:
            elm = []
        if plt is None:
            plt = vedo.Plotter(interactive=False)
            axes = vedo.Axes(xrange=(-.45, .84), yrange=(-.4, 1.05), zrange=(0, .5), xygrid=True)
            plt += [__doc__, *elm, axes]
            plt.show(resetcam=False, viewup='z')

        for thetas in list_of_configs:
            plt.remove(*elm)
            self.set_joint_angles(thetas, rad=rad)
            a, b, c, d = self.vedo_elements()
            points.append(self.get_endeffector_transform().transl())
            pts = vedo.Points(points).c("red")
            elm = *a, *b, *c, *d, *self.meshes(), *extras, pts
            plt.add(*elm)
            plt.show(resetcam=False, viewup='z')
        time.sleep(.5)
        return plt, elm


def main2b():
    robot = UR5()
    plt, elm = robot.animate_configurations([(i % 360, -23, -146, 173, -95, 181) for i in range(0, 360, 5)], rad=False)
    plt, elm = robot.animate_configurations([(0, (-23 + i) % 360, -146, 173, -95, 181) for i in range(0, 360, 5)],
                                            rad=False, plt=plt, elm=elm)
    plt, elm = robot.animate_configurations([(0, -23, (-146 + i) % 360, 173, -95, 181) for i in range(0, 360, 5)],
                                            rad=False, plt=plt, elm=elm)
    plt, elm = robot.animate_configurations([(0, -23, -146, (173 + i) % 360, -95, 181) for i in range(0, 360, 5)],
                                            rad=False, plt=plt, elm=elm)
    plt, elm = robot.animate_configurations([(0, -23, -146, 173, (-95 + i) % 360, 181) for i in range(0, 360, 5)],
                                            rad=False, plt=plt, elm=elm)
    plt, elm = robot.animate_configurations([(0, -23, -146, 173, -95, (181 + i) % 360) for i in range(0, 360, 5)],
                                            rad=False, plt=plt, elm=elm)
    plt.interactive()  # keep current frame


if __name__ == '__main__':
    r = UR5()
    plt = vedo.Plotter()
    elms = []
    while True:
        r.set_joint_angles(*tuple(np.random.randint(low=-180, high=180, size=6)), rad=False)
        print(r.hitting_itself())
        plt.remove(*elms)
        elms = *r.meshes(), *r.mesh_spheres()
        plt.add(elms)
        vedo.show(elms, interactive=True, axes=1)
