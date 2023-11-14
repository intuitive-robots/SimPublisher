def mj2unity_pos(pos):
    return [-pos[1], pos[2], pos[0]]


def mj2unity_quat(quat):
    # note that the order is "[x, y, z, w]"
    return [quat[2], -quat[3], -quat[1], quat[0]]


def unity2mj_pos(pos):
    return [-pos[1], pos[2], pos[0]]


def unity2mj_quat(quat):
    # note that the order is "[x, y, z, w]"
    return [quat[2], -quat[3], -quat[1], quat[0]]