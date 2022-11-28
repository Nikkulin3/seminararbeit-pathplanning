import numpy as np

DH = {
    # "theta": (0, 0, 0, 0, 0, 0),
    "a": (0, -.425, -.3922, 0, 0, 0),
    "d": (.1625, 0, 0, .1333, .0997, .0996),
    "alpha": (np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0)
}


def tf_matrix(joint_no, theta):
    assert 0 <= joint_no <= 6
    c = np.cos
    s = np.sin
    th = theta
    # al = DH["alpha"][joint_no]
    d = DH["d"][joint_no]
    # a = DH["a"][joint_no]

    a_minus = 0 if joint_no == 0 else DH["a"][joint_no - 1]
    al_minus = 0 if joint_no == 0 else DH["alpha"][joint_no - 1]

    tf = np.array([
        [c(th), -s(th), 0, a_minus],
        [s(th) * c(al_minus), c(th) * c(al_minus), -s(al_minus), -s(al_minus) * d],
        [s(th) * s(al_minus), c(th) * s(al_minus), c(al_minus), c(al_minus) * d],
        [0, 0, 0, 1],
    ])

    return tf


def position(tf):
    return tf[:3, 3]


def euler(tf):
    R = tf[:3, :3]
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return np.rad2deg(np.array((alpha, beta, gamma)))


def direct(theta_angles):
    tf = np.identity(4)
    for joint, theta in enumerate(theta_angles):
        mat = tf_matrix(joint, theta)
        print(np.round(mat, 3))
        tf = np.matmul(tf, mat)
    print()
    print(euler(tf))
    print(position(tf) * 1000)
    print(np.round(tf, 3))


def main():
    x = 0
    direct((x, x, x, x, x, x))


if __name__ == '__main__':
    main()
