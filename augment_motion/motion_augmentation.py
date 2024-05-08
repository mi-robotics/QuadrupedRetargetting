import numpy as np
import torch
import math

def swap_left_right_legs(rotations):
    # default legs rotations order fl,fr,rl,rr

    fl = rotations[:, [1,2,3,4]]
    fr = rotations[:, [5,6,7,8]]
    rl = rotations[:, [9,10,11,12]]
    rr = rotations[:, [13,14,15,16]]
    base = rotations[:, [0]]

    rotations = np.concatenate((base, fr,fl,rr,rl ), axis=1)

    return rotations

def invert_quart(rotations, x=False, y=False, z=False):
    inv_x = rotations[:, :, [0]]
    inv_y = rotations[:, :, [1]]
    inv_z = rotations[:, :, [2]]
    w = rotations[:, :, [3]]

    if x:
        inv_x = -inv_x
    if y:
        inv_y = -inv_y
    if z:
        inv_z = -inv_z

    rotations = np.concatenate((inv_x, inv_y, inv_z, w), axis=-1)

    return rotations

def invert_position(positions, x=False, y=False, z=False):
    inv_x = positions[:, [0]]
    inv_y = positions[:, [1]]
    inv_z = positions[:, [2]]

    if x:
        inv_x = -inv_x
    if y:
        inv_y = -inv_y
    if z:
        inv_z = -inv_z
    
    positions = np.concatenate((inv_x, inv_y, inv_z), axis=-1)

    return positions

def reflect_xy_plane(positions, rotations):
    """
    Relfect the robot along XY plane 
    Steps:
        - swap left and right legs
        - inverts x and z rotations
        - invert z and x orientations
        - invert y position
    """
    rotations = swap_left_right_legs(rotations)
    rotations = invert_quart(rotations, x=True, z=True)
    positions = invert_position(positions, y=True)

    return positions, rotations


def reverse_time(positions, rotations):
    positions = np.flip(positions, axis=0)
    rotations = np.flip(rotations, axis=0)
    return positions, rotations


def reverse_and_reflect(positions, rotations):

    positions, rotations = reflect_xy_plane(positions, rotations)
    positions, rotations = reverse_time(positions, rotations)

    return positions, rotations


def add_feet_rotations(rot):
    rotations = torch.Tensor(rot)
    toes = torch.Tensor([[0., 0., 0., 1.]]).unsqueeze(0).repeat(rotations.size(0),1,1)

    r0 = rotations[:, :4, :]
    r1 = rotations[:, 4:7, :]
    r2 = rotations[:, 7:10, :]
    r3 = rotations[:, 10:, :]


    rotations = torch.cat((r0, toes, r1, toes, r2, toes, r3, toes), dim=1)

    return rotations.numpy()


def resize_motion_clip(pos, rot, seq_len, file_name, dir):

    def split_array(arr, sub_array_length):
        n = arr.shape[0]
        indices = list(range(sub_array_length, n, sub_array_length))
        return np.split(arr, indices)

    s_pos = split_array(pos,seq_len) 
    s_rot = split_array(rot,seq_len)

    for i in range(len(s_pos)):
        _pos = s_pos[i]
        _rot = s_rot[i]
        np.savez(f'{dir}/{file_name}_{i}', pos=_pos, rot=_rot)
    return


if __name__ == '__main__':
    import os

    # RESIZING INTO 2 SECOND CLIPS -----------------------------------------------------------------------------------
    a1_inv_dir = './data/a1_captureV3/inv'
    a1_dir = './data/a1_captureV3/feet'

    save_inv_dir = './data/a1_captureV3/short/inv'
    save_dir = './data/a1_captureV3/short'

    inv_files = os.listdir(a1_inv_dir)
    files = os.listdir(a1_dir)

    for f_name in inv_files:
        save_name = f_name.split('.npz')[0]
        f_path = a1_inv_dir + '/' + f_name
        if os.path.isfile(f_path):
            data = np.load(f_path)
            pos = data['pos']
            rot = data['rot']
            resize_motion_clip(pos, rot, 200, save_name, save_inv_dir )


    for f_name in files:
        save_name = f_name.split('.npz')[0]
        f_path = a1_dir + '/' + f_name
        if os.path.isfile(f_path):
            data = np.load(f_path)
            pos = data['pos']
            rot = data['rot']
            resize_motion_clip(pos, rot, 200, save_name, save_dir )


    # AUGMENT DATA ------------------------------------------------------------------------------------------------
    # Directory to list files from
    # mocap_dir = "./data/processed"
    a1_dir = './data/a1_captureV3'

    # List all files in the directory
    # mocap_files = os.listdir(mocap_dir)
    a1_files = os.listdir(a1_dir)

    # for file_name in mocap_files:
    #     file_path = mocap_dir + '/' + file_name
    #     if os.path.isfile(file_path):

    #         data = np.load(file_path)

    #         # Extracting the positions and rotations
    #         pos = data['pos']
    #         rot = data['rot']
    
    #         pos, rot = reflect_xy_plane(positions=pos, rotations=rot)

    #         np.savez(f'{mocap_dir}/inv/{file_name}', pos=pos, rot=rot)

    # for file_name in a1_files:
    #     file_path = a1_dir + '/' + file_name
    #     if os.path.isfile(file_path):
    #         data = np.load(file_path)

    #         # Extracting the positions and rotations
    #         pos = data['pos']
    #         rot = data['rot']
    #         rot = add_feet_rotations(rot)
    #         pos, rot = reflect_xy_plane(positions=pos, rotations=rot)

    #         np.savez(f'{a1_dir}/inv/{file_name}', pos=pos, rot=rot)


    # for file_name in a1_files:
    #     file_path = a1_dir + '/' + file_name
    #     if os.path.isfile(file_path):
    #         data = np.load(file_path)

    #         # Extracting the positions and rotations
    #         pos = data['pos']
    #         rot = data['rot']
    #         rot = add_feet_rotations(rot)
    #         # pos, rot = reflect_xy_plane(positions=pos, rotations=rot)

    #         np.savez(f'{a1_dir}/feet/{file_name}', pos=pos, rot=rot)
