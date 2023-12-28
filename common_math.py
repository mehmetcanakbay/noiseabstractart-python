import numpy as np


def RotationMatrix(theta, beta, alpha):
    """
    Returns a 3x3 rotation matrix.
    X, Y, Z -> theta, beta, alpha
    """
    alpha = alpha * (np.pi / 180.0)
    beta = beta * (np.pi / 180.0)
    theta = theta * (np.pi / 180.0)
    matrix = [
        [
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta) * np.sin(theta)
            - np.sin(alpha) * np.cos(theta),
            np.cos(alpha) * np.sin(beta) * np.cos(theta)
            + np.sin(alpha) * np.sin(theta),
        ],
        [
            np.sin(alpha) * np.cos(beta),
            np.sin(alpha) * np.sin(beta) * np.sin(theta)
            + np.cos(alpha) * np.cos(theta),
            np.sin(alpha) * np.sin(beta) * np.cos(theta)
            - np.cos(alpha) * np.sin(theta),
        ],
        [-np.sin(beta), np.cos(beta) * np.sin(theta), np.cos(beta) * np.cos(theta)],
    ]
    return np.array(matrix, dtype=np.float32)


def rotate_in_3d(coords, rot_in_x, rot_in_y, rot_in_z):
    """
    Rotation in 3D.

    coords: 3d vector.
    rot_in_x: Roll
    rot_in_y: Pitch
    rot_in_z: Yaw
    """
    rotate_matrix = RotationMatrix(rot_in_x, rot_in_y, rot_in_z)
    out = np.dot(rotate_matrix, coords)
    return list(map(lambda x: int(x), out))


def rotate_in_2d(coords, amount):
    theta = amount * (np.pi / 180)
    rot_matrix = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    out = np.matmul(np.array(rot_matrix), coords)
    return list(map(lambda x: int(x), out))


def dist(vec1, vec2):
    sum_ = 0
    for j in range(len(vec1)):
        sum_ += np.square(vec1[j] - vec2[j])
    return np.sqrt(sum_)
