import math
import numpy as np


def clamp_norm(vec, max_norm):
    norm = np.linalg.norm(vec)
    if norm < 1e-9 or norm <= max_norm:
        return vec
    return vec * (max_norm / norm)


def skew(vec):
    x, y, z = vec
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def axis_angle_to_matrix(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    axis_hat = skew(axis)
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * axis_hat
        + (1.0 - math.cos(theta)) * (axis_hat @ axis_hat)
    )


def matrix_to_rotvec(rot):
    trace = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(trace)
    if theta < 1e-9:
        return np.zeros(3, dtype=np.float64)
    denom = 2.0 * math.sin(theta)
    axis = np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]],
        dtype=np.float64,
    ) / denom
    return axis * theta


def rotation_error(target_rot, current_rot):
    relative = target_rot @ current_rot.T
    return matrix_to_rotvec(relative)


def matrix_to_rot6d(rot):
    return rot[:, :2].reshape(-1)


def quat_wxyz_to_xyzw(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
