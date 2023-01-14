import pickle

import numpy as np
from scipy.spatial import KDTree

from Robots import UR5


class ConfigurationSpace:
    CARTESIAN_CONSTRAINTS = {
        "x": (-.45, .84),
        "y": (-.4, 1.05),
        "z": (.04, 1e10)
    }

    def __init__(self):
        self.resolution = 10

        self.floor_height = 0

        try:
            self.obstacle_space, self.free_space = self.load_previous()
        except FileNotFoundError:
            self.obstacle_space = []
            self.free_space = []
        self.robot = UR5()
        self.obstacle_space = set([x[1:-1] for x in self.obstacle_space])
        self.obstacle_space_l = list(self.obstacle_space)
        self.free_space_l = list(self.free_space)

        self.obs_tree = self.free_tree = None

    @staticmethod
    def load_previous():
        with open("obstacle_space.pkl", "rb") as f:
            with open("free_space.pkl", "rb") as f2:
                return set(pickle.load(f)), set(pickle.load(f2))

    def __problem_generator(self):
        resolution_degrees = self.resolution
        t1, t6 = 0, 0
        # for t1 in range(0, 360, resolution_degrees): # t1 does not matter
        for t2 in range(0, 360, resolution_degrees):
            for t3 in range(0, 360, resolution_degrees):
                for t4 in range(0, 360, resolution_degrees):
                    for t5 in range(0, 360, resolution_degrees):
                        # for t6 in range(0, 360, resolution_degrees): # t6 does not matter
                        yield t1, t2, t3, t4, t5, t6

    def nearest_free(self, thetas, rad=True):
        if rad:
            thetas = np.rad2deg(thetas)
        thetas_ = [x % 360 for x in thetas]
        if self.free_tree is None:
            self.free_tree = KDTree(self.free_space_l)
        dt = [x - y for x, y in zip(thetas, thetas_)]
        d, i = self.free_tree.query(tuple(thetas_[1:-1]))
        thetas_new = [dx + x for dx, x in
                      zip(dt, [thetas_[0]] + list(self.free_space_l[i]) + [thetas_[-1]])
                      ]
        thetas_new[0], thetas_new[-1] = [int(np.round(x / self.resolution) * self.resolution) for x in
                                         [thetas_new[0], thetas_new[-1]]]
        if rad:
            thetas_new = np.deg2rad(thetas_new)
        return d, thetas_new

    @staticmethod
    def angle_distance(thetas1, thetas2):
        return np.sum([np.abs(x) for x in np.array(thetas1) - thetas2])

    @staticmethod
    def cartesian_distance(thetas1, thetas2, rad=True):
        robot = UR5()
        robot.set_joint_angles(*thetas1, rad=rad)
        tf1 = robot.get_endeffector_transform()
        robot.set_joint_angles(*thetas2, rad=rad)
        tf2 = robot.get_endeffector_transform()
        return np.linalg.norm(tf2.transl() - tf1.transl())

    def nearest_obs(self, thetas, rad):
        if self.obs_tree is None:
            self.obs_tree = KDTree(self.obstacle_space_l)
        d, i = self.obs_tree.query(thetas)
        return d, self.obstacle_space_l[i]

    @staticmethod
    def random_configuration(rad=True):
        r = np.random.random_sample(6)
        r = r * 4 * np.pi - 2 * np.pi
        if rad:
            return r
        return np.rad2deg(r)

    def calculate(self):
        from pathos.multiprocessing import ProcessingPool as Pool, cpu_count

        pool = Pool(processes=cpu_count())

        def solver(problem):
            robot = UR5()
            robot.set_joint_angles(*problem, rad=False)
            if robot.hitting_itself():
                print(problem)
                return problem
            return None

        problems = list(self.__problem_generator())
        results = pool.map(solver, problems)
        solution = [r for r in results if r is not None]
        with open("obstacle_space.pkl", "wb") as f:
            pickle.dump(solution, f)
        self.obstacle_space = solution
        self.free_space = [x[1:-1] for x in self.__problem_generator() if x not in self.obstacle_space]
        with open("free_space.pkl", "wb") as f:
            pickle.dump(self.free_space, f)
        # n = 0
        # t0 = time()
        # tot = int(360 / resolution_degrees) ** 6
        # n += 1
        # for problem in problem_generator():
        #     t1 = time()
        #     if t1 - t0 > 2:
        #         t0 = t1
        #         print(
        #             f"{n}/{tot} ({np.round(n / tot * 100, 2)}%), found: {len(self.obstacle_space)}")
        #     self.robot.set_joint_angles(*problem, rad=False)
        #     if self.robot.hitting_itself():
        #         self.obstacle_space.add(problem)
        print("done")

    def round_theta(self, theta):
        res = self.resolution
        return int(res * np.round(theta / res)) % 360

    def in_obs_space(self, thetas, rad):
        new_thetas = []
        if rad:
            thetas = np.rad2deg(thetas)
        for i, t in enumerate(thetas):
            if i == 0 or i == 5:
                new_thetas.append(0)
            else:
                rounded_theta = self.round_theta(t)
                new_thetas.append(rounded_theta)

        self_collision = tuple(new_thetas[1:-1]) in self.obstacle_space
        return self_collision

    def wall_collision(self, thetas, rad):
        self.robot.set_joint_angles(thetas, rad=rad)
        for i, p in enumerate(self.robot.get_joint_positions()):
            i += 1
            if not (self.CARTESIAN_CONSTRAINTS["x"][0] <= p[0] <= self.CARTESIAN_CONSTRAINTS["x"][1]):
                # print(f"X{i} {self.cartesian_constraints['x']} vs. {p[0]}")
                return True
            if not (self.CARTESIAN_CONSTRAINTS["y"][0] <= p[1] <= self.CARTESIAN_CONSTRAINTS["y"][1]):
                # print(f"Y{i} {self.cartesian_constraints['y']} vs. {p[1]}")
                return True
            if not (self.CARTESIAN_CONSTRAINTS["z"][0] <= p[2] <= self.CARTESIAN_CONSTRAINTS["z"][1]):
                # print(f"Z{i} {self.cartesian_constraints['z']} vs. {p[2]}")
                return True
        return False

    def is_valid_path(self, path, rad=True, include_wall_collision=False):
        for i, configuration in enumerate(path):
            if not self.robot.within_joint_constraints(configuration, rad):
                return False
            if include_wall_collision and self.wall_collision(configuration, rad):
                return False
            if self.in_obs_space(configuration, rad):
                return False
        return True
