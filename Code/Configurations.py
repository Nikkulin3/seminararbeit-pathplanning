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
    RESOLUTION = 10

    def __init__(self, obs_space=None):
        if obs_space is not None:
            self.obstacle_space = obs_space
        else:
            try:
                self.obstacle_space = self.load_previous()
            except FileNotFoundError:
                self.obstacle_space = []
        self.robot = UR5()
        self.obstacle_space = set([x[1:-1] for x in self.obstacle_space])
        self.obstacle_space_l = list(self.obstacle_space)

        self.obs_tree = None

    @staticmethod
    def load_previous():
        with open("obstacle_space.pkl", "rb") as f:
            return set(pickle.load(f))

    @staticmethod
    def __problem_generator():
        resolution_degrees = ConfigurationSpace.RESOLUTION
        t1, t6 = 0, 0
        # for t1 in range(0, 360, resolution_degrees): # t1 does not matter
        for t2 in range(0, 360, resolution_degrees):
            for t3 in range(0, 360, resolution_degrees):
                for t4 in range(0, 360, resolution_degrees):
                    for t5 in range(0, 360, resolution_degrees):
                        # for t6 in range(0, 360, resolution_degrees): # t6 does not matter
                        config = (t1, t2, t3, t4, t5, t6)
                        yield config

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

    @staticmethod
    def calculate():
        from pathos.multiprocessing import ProcessingPool as Pool, cpu_count
        ConfigurationSpace.tested = 0
        pool = Pool(processes=cpu_count())
        print("pool created")

        def solver(problem):
            print(problem)
            robot = UR5()
            robot.set_joint_angles(*problem, rad=False)
            ConfigurationSpace.tested += 1
            if robot.hitting_itself():
                return problem
            if ConfigurationSpace.tested % 100 == 0:
                print(
                    f"tested: {ConfigurationSpace.tested}/{len(problems)}, ({ConfigurationSpace.tested / len(problems) * 100:.1f}%)")
            return None

        problems = list(ConfigurationSpace.__problem_generator())
        print("problems created")
        results = pool.uimap(solver, problems)
        solution = [r for r in results if r is not None]
        with open("obstacle_space.pkl", "wb") as f:
            pickle.dump(solution, f)
        print("done")
        return ConfigurationSpace()

    def round_theta(self, theta):
        res = self.RESOLUTION
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
