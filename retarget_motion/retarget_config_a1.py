import numpy as np

URDF_FILENAME = "a1/a1.urdf"

REF_POS_SCALE = 0.825
INIT_POS = np.array([0, 0, 0.30])
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [
    5,  # right hand
    15,  # right foot
    10,  # left hand
    20,  # left foot
]

SIM_HIP_JOINT_IDS = [1, 11, 6, 16]


# SIM_TOE_JOINT_IDS = [
#     7,12,17,22
# ]

SIM_TOE_JOINT_IDS = [
    7,17,12,22
]

SIM_HIP_JOINT_IDS = [3,13,8,18]

SIM_ROOT_OFFSET = np.array([0, 0, -0.06])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, -0.05, 0.0]),
    np.array([0, -0.05, 0.01]),
    np.array([0, 0.05, 0.0]),
    np.array([0, 0.05, 0.01])
]

# DEFAULT_JOINT_POSE = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
# DEFAULT_JOINT_POSE = np.array([0.0,0.0,0.0, 0.1, 0.1, -0.1, -0.1, 0.8, 1.0, 0.8, 1.0, -1.5, -1.5, -1.5,-1.5])

default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

joints= ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
DEFAULT_JOINT_POSE = [default_joint_angles[joint_name] if joint_name in default_joint_angles else 0.0 for joint_name in joints]

JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])
