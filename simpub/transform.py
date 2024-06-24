import numpy as np


def mat2euler(matrix):
    u, *_ = np.linalg.svd(matrix[:3, :3])
    pitch = np.arctan2(-u[2, 0], np.sqrt(u[2, 1]**2 + u[2, 2]**2))
    roll = np.arctan2(u[2, 1], u[2, 2])
    yaw = np.arctan2(-u[1, 0], u[0, 0])
    return np.array([roll, pitch, yaw])


def mat2transform(matrix):
    pos = matrix[:3, 3]
    return mat2euler(matrix[:3, :3]), pos


def quat2euler(q):
    if q is None:
        return np.array([0, 0, 0])

    q = q / np.linalg.norm(q)    # Normalize the quaternion

    # Calculate the Euler angles
    sin_pitch = 2 * (q[0] * q[2] - q[3] * q[1])
    pitch = np.arcsin(sin_pitch)

    if np.abs(sin_pitch) >= 1:
        # Gimbal lock case
        roll = np.arctan2(q[0] * q[1] + q[2] * q[3], 0.5 - q[1]**2 - q[2]**2)
        yaw = 0
    else:
        roll = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1]**2 + q[2]**2))
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2))
    return np.array([roll, pitch, yaw])


def euler2mat(euler_angles):
    """
    Convert Euler angles to a 4x4 transform matrix.

    Parameters:
    euler_angles (numpy array): Euler angles in radians, shape (3,)

    Returns:
    transform_matrix (numpy array): 4x4 transform matrix, shape (4, 4)
    """
    # Extract Euler angles
    alpha, beta, gamma = euler_angles

    # Calculate rotation matrices
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha), 0],
                   [0, np.sin(alpha), np.cos(alpha), 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[np.cos(beta), 0, np.sin(beta), 0],
                   [0, 1, 0, 0],
                   [-np.sin(beta), 0, np.cos(beta), 0],
                   [0, 0, 0, 1]])

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                    [np.sin(gamma), np.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Calculate the final rotation matrix
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Create the 4x4 transform matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R[:3, :3]

    return transform_matrix