"""Run from motion_imitation/retarget_motion to find data correctly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import time

import tensorflow as tf
import numpy as np

from utils.utilities import pose3d
from pybullet_utils import transformations
import pybullet
import pybullet_data as pd
from utils.utilities import motion_util

import retarget_motion.retarget_config_a1 as config
import math
import glob
import os
import torch
# import retarget_config_laikago as config
# import retarget_config_vision60 as config

POS_SIZE = 3
ROT_SIZE = 4
DEFAULT_ROT = np.array([0, 0, 0, 1])
FORWARD_DIR = np.array([1, 0, 0])

GROUND_URDF_FILENAME = "plane_implicit.urdf"



# reference motion
FRAME_DURATION = 0.01667
REF_COORD_ROT = transformations.quaternion_from_euler(0.5 * np.pi, 0, 0)
REF_POS_OFFSET = np.array([0, 0, 0])
REF_ROOT_ROT = transformations.quaternion_from_euler(0, 0, 0.47 * np.pi)

REF_PELVIS_JOINT_ID = 0
REF_NECK_JOINT_ID = 3
REF_HIP_JOINT_IDS = [6, 16, 11, 20]
REF_TOE_JOINT_IDS = [10, 19, 15, 23]

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

def get_position_from_q(q):
    q = torch.Tensor(q)
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)


    angle = angle * torch.sum(axis, dim=-1)
    # angle = normalize_angle(angle)
 
    return angle.item()

def list_all_joint_names(robot_id, p):
    """
    List all joint names in a PyBullet robot model.

    :param robot_id: The ID of the robot model in the PyBullet simulation.
    :type robot_id: int
    :return: A list of joint names.
    :rtype: list
    """
    num_joints = p.getNumJoints(robot_id)
    joint_names = []

    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        joint_name = joint_info[1].decode("utf-8")  # Extract and decode joint name
        joint_names.append(joint_name)

    return joint_names
  

def set_pose(robot, root_pos, root_rot, qs):
  num_joints = pybullet.getNumJoints(robot)
  pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

  i = 0
  for j in range(num_joints):

    j_info = pybullet.getJointInfo(robot, j)
    j_state = pybullet.getJointStateMultiDof(robot, j)

    j_pose_idx = j_info[3]
    j_pose_size = len(j_state[0])
    j_vel_size = len(j_state[1])

    joint_name = j_info[12].decode("utf-8") + '_joint'
    

    if not joint_name in ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FR_foot_joint' ,
                          'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FL_foot_joint',
                        'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RR_foot_joint',
                          'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RL_foot_joint']:
       print('------------------------', joint_name)
       continue
    

    j_pose = np.array([get_position_from_q(qs[i])])
    print(j_pose, joint_name)
    i += 1

    j_vel = np.zeros(j_vel_size)
    pybullet.resetJointStateMultiDof(robot, j, j_pose, j_vel)

    j_info = pybullet.getJointInfo(robot, j)
    j_state = pybullet.getJointStateMultiDof(robot, j)
    print('post update:', j_info[12].decode("utf-8") + '_joint', j_state[0])
    print()

#   input()

  return

def update_camera(robot):
  base_pos = np.array(pybullet.getBasePositionAndOrientation(robot)[0])
  [yaw, pitch, dist] = pybullet.getDebugVisualizerCamera()[8:11]
  pybullet.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
  return

def paint_red_circle_on_contact(p, robot_id, ground_plane_id, circle_visual_shape_id):
    """
    Paints a red circle on the ground plane when the robot's foot makes contact.

    Args:
    - robot_id: The unique ID of the robot in the PyBullet simulation.
    - ground_plane_id: The unique ID of the ground plane in the PyBullet simulation.
    - circle_visual_shape_id: The ID of the visual shape for the red circle.
    """
    # Check for contacts between the robot and the ground plane

    out = []
    contact_points = p.getContactPoints(robot_id, ground_plane_id)

    for contact in contact_points:
        foot_link_id = contact[3]  # Index 3 contains the link index of the robot
        contact_position = contact[5]  # Index 5 contains the contact position

        # Add a visual marker (red circle) at the contact position
        marker_id = p.createMultiBody(baseMass=0, 
                          baseVisualShapeIndex=circle_visual_shape_id, 
                          basePosition=contact_position)
        out.append(marker_id)
    return out
        
def delete_all_circles(p, markers):
    """
    Deletes all the circle markers created on the ground plane.
    """

    for marker_id in markers:
        p.removeBody(marker_id)


def convert_quaternion_between_handness(quat):
    """
    Convert a quaternion between right-hand and left-hand coordinate systems
    where Z-axis is up in both, but the X-axis is mirrored.

    Args:
    - quat (array-like): The quaternion to convert, represented as [x, y, z, w].

    Returns:
    - array-like: The converted quaternion.
    """
    x, y, z, w = quat
    return np.array([x, -y, z, -w])
  
def rotate_quaternion_90_deg_z_xyzw(quaternion):
    """
    Rotates the given quaternion by 90 degrees around the Z-axis in XYZW notation.
    
    Parameters:
    quaternion (tuple): A tuple (x, y, z, w) representing the original quaternion.
    
    Returns:
    tuple: Rotated quaternion in XYZW notation.
    """
    # Rotation quaternion for 90 degrees around the Z-axis
    theta = math.pi / 4
    q_rot = (0, 0, math.sin(theta), math.cos(theta))

    x1, y1, z1, w1 = quaternion
    x2, y2, z2, w2 = q_rot

    # Quaternion multiplication
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x, y, z, w])

def main(argv):

  
  p = pybullet
  p.connect(p.GUI, options="--width=1920 --height=1080 --mp4=\"test.mp4\" --mp4fps=60")
  p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)

  pybullet.setAdditionalSearchPath(pd.getDataPath())


  print("PyBullet Data Directory:", pd.getDataPath())

  # rotations = torch.Tensor(np.load('./rotations-v5.npy'))
  # fr = rotations[:, [1,2,3,4]]
  # fl = rotations[:, [5,6,7,8]]
  # rr = rotations[:, [9,10,11,12]]
  # rl = rotations[:, [13,14,15,16]]
  # base = rotations[:, [0]]

  # rotations = np.concatenate((base,fl,fr,rl,rr), axis=1)

  # print(rotations.shape)
  # input()
  # positions = torch.Tensor(np.load('./positions-v5.npy'))

  rotations = torch.Tensor(np.load('./rotations-v6-rev.npy'))
  positions = torch.Tensor(np.load('./positions-v6-rev.npy'))

  while True:
    
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, 0)

    ground = pybullet.loadURDF(GROUND_URDF_FILENAME)
    # robot = pybullet.loadURDF('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/data/assets/parkour/a1/urdf/a1.urdf', config.INIT_POS, config.INIT_ROT)
    robot = pybullet.loadURDF('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/data/assets/parkour/a1/urdf/a1.urdf')

    circle_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])

    markers = []
    for i in range(len(rotations)):
      delete_all_circles(p, markers)
      position = positions[i]
      rotation = rotations[i][0]
      qs = rotations[i][1:]

      set_pose(robot, position, rotation, qs)
      markers = paint_red_circle_on_contact(p, robot, ground, circle_visual_shape_id)



      update_camera(robot)
      p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
 
      p.stepSimulation()
      time.sleep(0.1)


  pybullet.disconnect()

  return


if __name__ == "__main__":
  tf.app.run(main)
