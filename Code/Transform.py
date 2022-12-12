import numpy as np
from scipy.spatial.transform import Rotation
from sympy import Quaternion, Matrix


class T:
    def __init__(self, matrix_4_4: np.array):
        if type(matrix_4_4) is T:
            self.mat: np.array = matrix_4_4.mat
        else:
            self.mat: np.array = matrix_4_4

    def __mul__(self, other):
        return T(np.matmul(self.mat, other.mat))

    def __invert__(self):
        inv = np.zeros_like(self.mat)
        rot_t = self.rot().transpose()
        return self.from_rotation_and_translation(rot_t, np.matmul(-rot_t, self.transl()))

    def rot(self) -> np.array:
        return self.mat[:3, :3]

    def transl(self) -> np.ndarray:
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
        return axis, angle

    def euler_angles(self, convention='zyx'):
        angles = Rotation.from_matrix(self.rot()).as_euler(convention)
        return np.array(angles)

    def quaternion(self):
        return Quaternion.from_rotation_matrix(Matrix(self.rot()))

    @staticmethod
    def from_rotation_and_translation(rot: np.ndarray, transl: np.ndarray):
        mat = np.zeros((4, 4))
        mat[:3, :3] = rot
        mat[3, 3] = 1
        mat[:3, 3] = transl
        return T(mat)

    def __str__(self):
        return str(self.mat)

    def x_rot(self):
        return self.mat[:3, 0]

    def y_rot(self):
        return self.mat[:3, 1]

    def z_rot(self):
        return self.mat[:3, 2]
