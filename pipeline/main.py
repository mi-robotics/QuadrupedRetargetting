import os
import sys
import numpy as np
import torch 

# Get the directory of the script or environment you are currently in
current_directory = os.getcwd()

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

sys.path.append(parent_directory)


from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from augment_motion.motion_augmentation import resize_motion_clip, reflect_xy_plane


RAW_DIR = '/home/mcarroll/Documents/cdt-1/QuadrupedRetargetting/data/raw'
AUGMENTED_DIR = '/home/mcarroll/Documents/cdt-1/QuadrupedRetargetting/data/augmented'
PROCESSED_DIR = '/home/mcarroll/Documents/cdt-1/QuadrupedRetargetting/data/processed'

def check_dir(dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return


def invert_files(data_dir):

    data_files = os.listdir(RAW_DIR+'/'+data_dir)
    for file_name in data_files:
        file_path = RAW_DIR + '/' + data_dir + '/' + file_name
        
        if os.path.isfile(file_path):
            data = np.load(file_path)
            # Extracting the positions and rotations
            pos = data['pos']
            rot = data['rot']
    
            pos, rot = reflect_xy_plane(positions=pos, rotations=rot)
         
            check_dir(f'{AUGMENTED_DIR}/{data_dir}/inv/')
         
            np.savez(f'{AUGMENTED_DIR}/{data_dir}/inv/{file_name}', pos=pos, rot=rot)
    return



def resize_files(data_dir):
    data_files = os.listdir(RAW_DIR+'/'+data_dir)
    inv_data_files = os.listdir(AUGMENTED_DIR+'/'+data_dir+'/inv/')

    for f_name in data_files:
        save_name = f_name.split('.npz')[0]
        f_path = RAW_DIR + '/' + data_dir + '/' +f_name
        if os.path.isfile(f_path):
            data = np.load(f_path)
            pos = data['pos']
            rot = data['rot']
        
            check_dir( f'{AUGMENTED_DIR}/{data_dir}/short/')

            resize_motion_clip(pos, rot, 200, save_name, f'{AUGMENTED_DIR}/{data_dir}/short/' )

    for f_name in inv_data_files:
       
        save_name = f_name.split('.npz')[0]
        f_path = AUGMENTED_DIR + '/' + data_dir + '/inv/' +f_name
        if os.path.isfile(f_path):
            data = np.load(f_path)
            pos = data['pos']
            rot = data['rot']

            check_dir(f'{AUGMENTED_DIR}/{data_dir}/short/inv/')

            resize_motion_clip(pos, rot, 200, save_name, f'{AUGMENTED_DIR}/{data_dir}/short/inv/' )
    return



def format_files(data_dir, fps=60):
    data_files = os.listdir( f'{AUGMENTED_DIR}/{data_dir}/short/')
    inv_data_files = os.listdir( f'{AUGMENTED_DIR}/{data_dir}/short/inv/')

    for f_name in data_files:
        name = f_name.split('.npz')[0]
        f_path = f'{AUGMENTED_DIR}/{data_dir}/short/' + f_name

        if os.path.isfile(f_path):
            try:
                data = np.load(f_path)
                pos = torch.Tensor(data['pos'])
                rot = torch.Tensor(data['rot'])
                
                if len(pos) > 50:

                    a1_skeleton = SkeletonState.from_file('/home/mcarroll/Documents/cdt-1/QuadrupedRetargetting/poselib/data/a1_tpose_v2.npy').skeleton_tree

                    a1_state = SkeletonState.from_rotation_and_root_translation(
                                a1_skeleton, r=rot, t=pos, is_local=True
                            )
                    motion = SkeletonMotion.from_skeleton_state(a1_state, fps=50)
                    # plot_skeleton_motion_interactive(motion)

                    check_dir(f'{PROCESSED_DIR}/{data_dir}/')

                    motion.to_file(f'{PROCESSED_DIR}/{data_dir}/{name}.npy')
            except:
                raise Exception('Data Formatting Error')
            
    for f_name in inv_data_files:
        name = f_name.split('.npz')[0]
        f_path = f'{AUGMENTED_DIR}/{data_dir}/short/inv/' + f_name

        if os.path.isfile(f_path):
            try:
                data = np.load(f_path)
                pos = torch.Tensor(data['pos'])
                rot = torch.Tensor(data['rot'])
                
                if len(pos) > 50:

                    a1_skeleton = SkeletonState.from_file('/home/mcarroll/Documents/cdt-1/QuadrupedRetargetting/poselib/data/a1_tpose_v2.npy').skeleton_tree

                    a1_state = SkeletonState.from_rotation_and_root_translation(
                                a1_skeleton, r=rot, t=pos, is_local=True
                            )
                    motion = SkeletonMotion.from_skeleton_state(a1_state, fps=50)
                    # plot_skeleton_motion_interactive(motion)

                    check_dir(f'{PROCESSED_DIR}/{data_dir}/inv')

                    motion.to_file(f'{PROCESSED_DIR}/{data_dir}/inv/{name}.npy')
            except:
                raise Exception('Data Formatting Error')
    return 



def main():

    target_dirs = ['stand']

    for target_dir in target_dirs:
        invert_files(target_dir)
        print('INVERTED')
        resize_files(target_dir)
        print('RESIZED')
        format_files(target_dir)
        print('FORMATED')

    return 

if __name__ == '__main__':
    main()