"""Policy inputs from robot joint readings, in the sim world frame the policies were trained in."""
import numpy as np

SIM_BASE = np.array([-0.56, 0.0, 0.912])  # robot base in sim world; no rotation
FLANGE_OFFSET = 0.107
GRIP_SITE_OFFSET = 0.2405  # robosuite grip site, along link-7 z (0.2035 + 3.7 cm adapter)
ACTION_SCALE = 0.05  # m per step, base frame

# Franka modified DH (a, d, alpha) for joints 1-7.
DH = ((0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2))


def link7_pose(q):
    """4x4 pose of link 7 in the robot base frame."""
    T = np.eye(4)
    for (a, d, alpha), theta in zip(DH, np.asarray(q, dtype=np.float64)):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
        T = T @ np.array([[ct, -st, 0.0, a],
                          [st * ca, ct * ca, -sa, -d * sa],
                          [st * sa, ct * sa, ca, d * ca],
                          [0.0, 0.0, 0.0, 1.0]])
    return T


def grip_site_base(q):
    T = link7_pose(q)
    return T[:3, 3] + T[:3, 2] * GRIP_SITE_OFFSET


def build_observation(q, dq, target_xy_base=None):
    """24 inputs (banging), or 26 with a target given in the robot base frame (hammering)."""
    q = np.asarray(q, dtype=np.float64)
    parts = [np.cos(q), np.sin(q), np.asarray(dq, dtype=np.float64), grip_site_base(q) + SIM_BASE]
    if target_xy_base is not None:
        parts.append(np.asarray(target_xy_base, dtype=np.float64) + SIM_BASE[:2])
    return np.concatenate(parts).astype(np.float32)
