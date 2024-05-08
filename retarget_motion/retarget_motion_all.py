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

import torch

from utils.utilities import pose3d
from pybullet_utils import transformations
import pybullet
import pybullet_data as pd
from utils.utilities import motion_util

import retarget_config_a1 as config
import math
import glob
import os
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

SEQUENCE_LEN = 240 #seconds

directory_path = 'data/'

# Pattern to match files ending with 'pos.txt'
pattern = os.path.join(directory_path, '*pos.txt')

# Get list of files matching the pattern
files = glob.glob(pattern)

# Print the list of files
for file in files:
    print(file)

print(len(files), ': files')


def extract_string(input_str):
    # Split the string after the first '/'
    first_split = input_str.split('/')[1]

    # Split the result after the second '_'
    second_split = first_split.split('_')

    # Return the required part of the string
    return second_split[0] + '_' + second_split[1]

mocap_motions = [[extract_string(file), file, 0,-1] for i, file in enumerate(files) ]


mocap_motions = [
  ['dog_run02', 'data/dog_run02_joint_pos.txt', 0, -1],
  ['dog_run04', 'data/dog_run04_joint_pos.txt', 500, 500+240], 
  ['dog_walk01', 'data/dog_walk01_joint_pos.txt', 300, -1], 
  ['dog_walk09', 'data/dog_walk09_joint_pos.txt', 0, 2600], 
  ['dog_walk09', 'data/dog_walk09_joint_pos.txt', 2600+200, -1], 
  ['dog_walk05', 'data/dog_walk05_joint_pos.txt', 200, 200+350],  
  ['dog_turn00', 'data/dog_turn00_joint_pos.txt', 330, 330+430], 
  ['dog_walk03', 'data/dog_walk03_joint_pos.txt', 0, -1], 

  ['dog_run00', 'data/dog_run00_joint_pos.txt', 400, -1], 
  ['dog_walk02', 'data/dog_walk02_joint_pos.txt', 410, -1], 
  ['dog_walk06', 'data/dog_walk06_joint_pos.txt', 175, 175+250], 

  ['dog_run01', 'data/dog_run01_joint_pos.txt', 0, -1], 
  ['dog_trot', 'data/dog_trot_joint_pos.txt', 0, -1], 
  ['dog_walk04', 'data/dog_walk04_joint_pos.txt', 240, -1], 
  ['dog_walk00', 'data/dog_walk00_joint_pos.txt', 110, -1]
  ]





  
def build_markers(num_markers):
  marker_radius = 0.02

  markers = []
  for i in range(num_markers):
    if (i == REF_NECK_JOINT_ID) or (i == REF_PELVIS_JOINT_ID)\
        or (i in REF_HIP_JOINT_IDS):
      col = [0, 0, 1, 1]
    elif (i in REF_TOE_JOINT_IDS):
      col = [1, 0, 0, 1]
    else:
      col = [0, 1, 0, 1]

    virtual_shape_id = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                  radius=marker_radius,
                                                  rgbaColor=col)
    body_id =  pybullet.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=-1,
                                  baseVisualShapeIndex=virtual_shape_id,
                                  basePosition=[0,0,0],
                                  useMaximalCoordinates=True)
    markers.append(body_id)

  return markers

def get_joint_limits(robot):
  num_joints = pybullet.getNumJoints(robot)
  joint_limit_low = []
  joint_limit_high = []

  for i in range(num_joints):
    joint_info = pybullet.getJointInfo(robot, i)
    joint_type = joint_info[2]

    if (joint_type == pybullet.JOINT_PRISMATIC or joint_type == pybullet.JOINT_REVOLUTE):
      joint_limit_low.append(joint_info[8])
      joint_limit_high.append(joint_info[9])

  return joint_limit_low, joint_limit_high

def get_root_pos(pose):
  return pose[0:POS_SIZE]

def get_root_rot(pose):
  return pose[POS_SIZE:(POS_SIZE + ROT_SIZE)]

def get_joint_pose(pose):
  return pose[(POS_SIZE + ROT_SIZE):]

def set_root_pos(root_pos, pose):
  pose[0:POS_SIZE] = root_pos
  return

def set_root_rot(root_rot, pose):
  pose[POS_SIZE:(POS_SIZE + ROT_SIZE)] = root_rot
  return

def set_joint_pose(joint_pose, pose):
  pose[(POS_SIZE + ROT_SIZE):] = joint_pose
  return

def set_pose(robot, pose):
  num_joints = pybullet.getNumJoints(robot)
  root_pos = get_root_pos(pose)
  root_rot = get_root_rot(pose)
  pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

  q =[]

  for j in range(num_joints):
    j_info = pybullet.getJointInfo(robot, j)
    j_state = pybullet.getJointStateMultiDof(robot, j)


    j_pose_idx = j_info[3]
    j_pose_size = len(j_state[0])
    j_vel_size = len(j_state[1])

    if j_pose_idx >= 0:
      q.append(j_info[12].decode("utf-8")+'_joint')

    if (j_pose_size > 0):
      j_pose = pose[j_pose_idx:(j_pose_idx + j_pose_size)]
      j_vel = np.zeros(j_vel_size)
      pybullet.resetJointStateMultiDof(robot, j, j_pose, j_vel)



  return

def set_maker_pos(marker_pos, marker_ids):
  num_markers = len(marker_ids)
  assert(num_markers == marker_pos.shape[0])

  for i in range(num_markers):
    curr_id = marker_ids[i]
    curr_pos = marker_pos[i]

    pybullet.resetBasePositionAndOrientation(curr_id, curr_pos, DEFAULT_ROT)

  return

def process_ref_joint_pos_data(joint_pos):
  proc_pos = joint_pos.copy()
  num_pos = joint_pos.shape[0]

  for i in range(num_pos):
    curr_pos = proc_pos[i]
    curr_pos = pose3d.QuaternionRotatePoint(curr_pos, REF_COORD_ROT)
    curr_pos = pose3d.QuaternionRotatePoint(curr_pos, REF_ROOT_ROT)
    curr_pos = curr_pos * config.REF_POS_SCALE + REF_POS_OFFSET
    proc_pos[i] = curr_pos

  return proc_pos

def retarget_root_pose(ref_joint_pos):
  pelvis_pos = ref_joint_pos[REF_PELVIS_JOINT_ID]
  neck_pos = ref_joint_pos[REF_NECK_JOINT_ID]

  left_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[0]]
  right_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[2]]
  left_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[1]]
  right_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[3]]

  forward_dir = neck_pos - pelvis_pos
  forward_dir += config.FORWARD_DIR_OFFSET
  forward_dir = forward_dir / np.linalg.norm(forward_dir)

  delta_shoulder = left_shoulder_pos - right_shoulder_pos
  delta_hip = left_hip_pos - right_hip_pos
  dir_shoulder = delta_shoulder / np.linalg.norm(delta_shoulder)
  dir_hip = delta_hip / np.linalg.norm(delta_hip)

  left_dir = 0.5 * (dir_shoulder + dir_hip)

  up_dir = np.cross(forward_dir, left_dir)
  up_dir = up_dir / np.linalg.norm(up_dir)

  left_dir = np.cross(up_dir, forward_dir)
  left_dir[2] = 0.0 # make the base more stable
  left_dir = left_dir / np.linalg.norm(left_dir)

  rot_mat = np.array([[forward_dir[0], left_dir[0], up_dir[0], 0],
                      [forward_dir[1], left_dir[1], up_dir[1], 0],
                      [forward_dir[2], left_dir[2], up_dir[2], 0],
                      [0, 0, 0, 1]])

  root_pos = 0.5 * (pelvis_pos + neck_pos)
  #root_pos = 0.25 * (left_shoulder_pos + right_shoulder_pos + left_hip_pos + right_hip_pos)
  root_rot = transformations.quaternion_from_matrix(rot_mat)
  root_rot = transformations.quaternion_multiply(root_rot, config.INIT_ROT)
  root_rot = root_rot / np.linalg.norm(root_rot)

  return root_pos, root_rot

def retarget_pose(robot, default_pose, ref_joint_pos):
  joint_lim_low, joint_lim_high = get_joint_limits(robot)

  root_pos, root_rot = retarget_root_pose(ref_joint_pos)
  root_pos += config.SIM_ROOT_OFFSET

  pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

  inv_init_rot = transformations.quaternion_inverse(config.INIT_ROT)
  heading_rot = motion_util.calc_heading_rot(transformations.quaternion_multiply(root_rot, inv_init_rot))

  tar_toe_pos = []
  for i in range(len(REF_TOE_JOINT_IDS)):
    ref_toe_id = REF_TOE_JOINT_IDS[i]
    ref_hip_id = REF_HIP_JOINT_IDS[i]
    sim_hip_id = config.SIM_HIP_JOINT_IDS[i]
    toe_offset_local = config.SIM_TOE_OFFSET_LOCAL[i]

    ref_toe_pos = ref_joint_pos[ref_toe_id]
    ref_hip_pos = ref_joint_pos[ref_hip_id]

    hip_link_state = pybullet.getLinkState(robot, sim_hip_id, computeForwardKinematics=True)
    sim_hip_pos = np.array(hip_link_state[4])

    toe_offset_world = pose3d.QuaternionRotatePoint(toe_offset_local, heading_rot)

    ref_hip_toe_delta = ref_toe_pos - ref_hip_pos
    sim_tar_toe_pos = sim_hip_pos + ref_hip_toe_delta
    sim_tar_toe_pos[2] = ref_toe_pos[2]
    sim_tar_toe_pos += toe_offset_world

    tar_toe_pos.append(sim_tar_toe_pos)

  joint_pose = pybullet.calculateInverseKinematics2(robot, config.SIM_TOE_JOINT_IDS,
                                                    tar_toe_pos,
                                                    jointDamping=config.JOINT_DAMPING,
                                                    lowerLimits=joint_lim_low,
                                                    upperLimits=joint_lim_high,
                                                    restPoses=default_pose)
  joint_pose = np.array(joint_pose)

  pose = np.concatenate([root_pos, root_rot, joint_pose])


  return pose

def update_camera(robot):
  base_pos = np.array(pybullet.getBasePositionAndOrientation(robot)[0])
  [yaw, pitch, dist] = pybullet.getDebugVisualizerCamera()[8:11]
  pybullet.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
  return

def load_ref_data(JOINT_POS_FILENAME, FRAME_START, FRAME_END):
  joint_pos_data = np.loadtxt(JOINT_POS_FILENAME, delimiter=",")

  start_frame = 0 if (FRAME_START is None) else FRAME_START
  end_frame = joint_pos_data.shape[0] if (FRAME_END is None) else FRAME_END
  joint_pos_data = joint_pos_data[start_frame:end_frame]

  return joint_pos_data

def retarget_motion(robot, joint_pos_data):

  num_frames = joint_pos_data.shape[0]

  for f in range(num_frames):
    ref_joint_pos = joint_pos_data[f]
    ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
    ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)

    curr_pose = retarget_pose(robot, config.DEFAULT_JOINT_POSE, ref_joint_pos)
    set_pose(robot, curr_pose)

    if f == 0:
      pose_size = curr_pose.shape[-1]
      new_frames = np.zeros([num_frames, pose_size])

    new_frames[f] = curr_pose

  new_frames[:, 0:2] -= new_frames[0, 0:2]

  return new_frames

def output_motion(frames, out_filename):
  with open(out_filename, "w") as f:
    f.write("{\n")
    f.write("\"LoopMode\": \"Wrap\",\n")
    f.write("\"FrameDuration\": " + str(FRAME_DURATION) + ",\n")
    f.write("\"EnableCycleOffsetPosition\": true,\n")
    f.write("\"EnableCycleOffsetRotation\": true,\n")
    f.write("\n")

    f.write("\"Frames\":\n")

    f.write("[")
    for i in range(frames.shape[0]):
      curr_frame = frames[i]

      if i != 0:
        f.write(",")
      f.write("\n  [")

      for j in range(frames.shape[1]):
        curr_val = curr_frame[j]
        if j != 0:
          f.write(", ")
        f.write("%.5f" % curr_val)

      f.write("]")

    f.write("\n]")
    f.write("\n}")

  return

def axis_angle_to_quaternion(axis, angle_rad):
    # Normalize the axis vector
    if axis == (0,0,0):
      return [0,0,0,1]
    axis_length = math.sqrt(sum([x**2 for x in axis]))
    normalized_axis = [x/axis_length for x in axis]


    # Calculate the quaternion components
    qx = normalized_axis[0] * math.sin(angle_rad / 2)
    qy = normalized_axis[1] * math.sin(angle_rad / 2)
    qz = normalized_axis[2] * math.sin(angle_rad / 2)
    qw = math.cos(angle_rad / 2)

    return [qx, qy, qz, qw]

def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def quat_unit(a):
    return normalize(a)

def quat_from_angle_axis( axis, angle):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))

def get_frame_local_roations(robot, p, log=False):
  joints = list_all_joint_names(robot, p)

  # Get joint angles by name
  joint_angles = {}

  out = []
  rads = []
  pos = []
  base_position, base_orn = p.getBasePositionAndOrientation(robot)

  for joint_name in joints:
      if 'fixed' in joint_name and 'hip' in joint_name:
        continue

      if 'floating' in joint_name:
        continue

      if 'camera' in joint_name:
        continue
        
      # Find the joint index by name
      joint_index = p.getJointInfo(robot, joints.index(joint_name))[0]
      
      if joint_index != -1:  # Check if the joint name exists in the model
          joint_state = p.getJointState(robot, joint_index)
          joint_info = p.getJointInfo(robot, joint_index)

          joint_position = joint_state[0]
          joint_angles[joint_name] = joint_position

          joint_q =  axis_angle_to_quaternion(joint_info[13], joint_position)

      
          joint_q_v2 = quat_from_angle_axis(torch.Tensor(joint_info[13]), torch.Tensor([joint_position]))

    
          rads.append(joint_position)
          if 'imu' in joint_name:
            joint_q = list(base_orn)
            link_state = p.getLinkState(robot, joint_index)
            pos.append(list(link_state[0]))
            
          out.append(joint_q)
      else:
          raise 2
          print(f"Joint '{joint_name}' not found in the robot model.")

 
  # print(out[0])
  return out, pos, rads

def get_frame_root_translation(robot,p):
  
  # Get the position (translation) of the base link
  base_position, _ = p.getBasePositionAndOrientation(robot)
  

  return list(base_position)

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

def print_all_body_names(robot_id, p):
    """
    Print all body names in a PyBullet robot model.

    :param robot_id: The ID of the robot model in the PyBullet simulation.
    :type robot_id: int
    """
    num_bodies = p.getNumBodies()
    
    for body_id in range(num_bodies):
        body_info = p.getBodyInfo(body_id)
        body_name = body_info[1].decode("utf-8")  # Extract and decode body name



def save_positions_rotations(file_name, segment, positions, rotations):

  fr = rotations[:, [1,2,3,4]]
  fl = rotations[:, [5,6,7,8]]
  rr = rotations[:, [9,10,11,12]]
  rl = rotations[:, [13,14,15,16]]
  base = rotations[:, [0]]
  rotations = np.concatenate((base,fl,fr,rl,rr), axis=1)

  np.savez(f'./data/processed/{file_name}_{segment}', pos=positions, rot=rotations)
  return



def main(argv):

  
  p = pybullet
  p.connect(p.GUI, options="--width=1920 --height=1080 --mp4=\"test.mp4\" --mp4fps=60")
  p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
  pybullet.setAdditionalSearchPath(pd.getDataPath())

 

    
    
  for mocap_motion in mocap_motions:
    store_rots = []
    store_pos = []
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, 0)
  
    ground = pybullet.loadURDF(GROUND_URDF_FILENAME)
    # robot = pybullet.loadURDF('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/data/assets/parkour/a1/urdf/a1.urdf', config.INIT_POS, config.INIT_ROT)
    robot = pybullet.loadURDF('/home/milo/Documents/cdt-1/examples/ASE-Atlas/ase/data/assets/parkour/a1/urdf/a1.urdf')
    # Set robot to default pose to bias knees in the right direction.
    set_pose(robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))

    p.removeAllUserDebugItems()
    joint_pos_data = load_ref_data(mocap_motion[1],mocap_motion[2],mocap_motion[3])
  
    num_markers = joint_pos_data.shape[-1] // POS_SIZE
    marker_ids = build_markers(num_markers)
  
    retarget_frames = retarget_motion(robot, joint_pos_data)

    f = 0
    seg = 0
    num_frames = joint_pos_data.shape[0]

    
    set_pose(robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))

    for frame in range (num_frames):
      print(frame)
      if (frame % SEQUENCE_LEN == 0 and not frame == 0) or frame == num_frames-1:
        save_positions_rotations(mocap_motion[0], seg, np.array(store_pos), np.array(store_rots))
        store_pos = []
        store_rots =[]
        seg += 1

    
      time_start = time.time()
  
      f_idx = f % num_frames
  
      ref_joint_pos = joint_pos_data[f_idx]
      ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
      ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)
  
      pose = retarget_frames[f_idx]
      
      set_pose(robot, pose)
      set_maker_pos(ref_joint_pos, marker_ids)
  
      rots, imu_pos, rads = get_frame_local_roations(robot, p, False)
      
      pos = get_frame_root_translation(robot, p) #- config.SIM_ROOT_OFFSET

      store_rots.append(rots)
      store_pos.append(pos)

      update_camera(robot)
      p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
      f += 1
  
      time_end = time.time()
      sleep_dur = FRAME_DURATION - (time_end - time_start)
      sleep_dur = max(0, sleep_dur)
  
      time.sleep(sleep_dur)

    time.sleep(0.5) # jp hack
    for m in marker_ids:
      p.removeBody(m)
    marker_ids = []

  pybullet.disconnect()

  return


if __name__ == "__main__":
  tf.app.run(main)

