"""
This file is used to simulate linear and random motion.

@author: Haoan Xu, ZJU BME, 2021/9

"""

import os
import numpy as np
import SimpleITK as sitk
from MotionSimulation.motion_simulator import interleaved_motion_simulation, linear_motion_simulation
from MotionSimulation.motion_simulator import motion_free_simulation
# from MotionSimulation.motion_generator import MotionGenerator
from util import save_data


if __name__ == '__main__':
    motion_type = 'linear'
    input_path = './Data/Volumes/'

    name_list = os.listdir(input_path)
    for name in name_list:
        print('****** Processing', name, '******')
        img = sitk.ReadImage(input_path + name + '/volume.nii.gz')
        nda = sitk.GetArrayFromImage(img)
        nda = nda.transpose((2, 1, 0))

        pixdim_recon = 0.8
        s_thickness = 4
        st_ratio = s_thickness / pixdim_recon
        num_slice = 24
        orts = ['axi', 'cor', 'sag']

        if motion_type == 'random':
            output_path = './Data/RandomMotionStacks/'
            print('Simulating random motion')
            output_stack, output_transform, motion_free_index = interleaved_motion_simulation(nda, num_slice, st_ratio, orts)

            if not os.path.isdir(output_path + name):
                os.mkdir(output_path + name)
            for index_ort, ort in enumerate(orts):
                save_data(output_stack[index_ort], output_path + name + '/', name=ort, ort=ort,
                          pixdim=[pixdim_recon, pixdim_recon, s_thickness])
                print(ort, 'save done!')
            np.save(output_path + name + '/transform_gt.npy', output_transform)
            print('motion parameters save done!')
            with open(output_path + name + '/motion_free_index_gt.txt', 'w') as f_gt:
                f_gt.write(orts[motion_free_index] + '.nii.gz')
        elif motion_type == 'linear':
            output_path = './Data/LinearMotionStacks/'
            displacement = 0.4
            rotation = 2
            print('Simulating linear motion with displacement of', displacement, 'and rotation of', rotation)
            motion_path = 'D_' + str(round(displacement, 1)) + '_R_' + str(round(rotation, 1)) + '/'
            if not os.path.isdir(output_path + motion_path):
                os.mkdir(output_path + motion_path)
            output_stack, output_transform = linear_motion_simulation(nda, displacement, rotation, num_slice, st_ratio,
                                                                      orts)
            if not os.path.isdir(output_path + motion_path + name):
                os.mkdir(output_path + motion_path + name)
            for index_ort, ort in enumerate(orts):
                save_data(output_stack[index_ort], output_path + motion_path + name + '/', name=ort, ort=ort,
                          pixdim=[pixdim_recon, pixdim_recon, s_thickness])
                print(ort, 'save done!')
            np.save(output_path + motion_path + name + '/transform_gt.npy', output_transform)
            print('motion parameters save done!')
        elif motion_type == 'motion_free':
            output_path = './Data/MotionFreeStacks/'
            print('Simulating motion-free stacks with')
            output_stack, output_transform = motion_free_simulation(nda, num_slice, st_ratio, orts)
            if not os.path.isdir(output_path + name):
                os.mkdir(output_path + name)
            for index_ort, ort in enumerate(orts):
                save_data(output_stack[index_ort], output_path + name + '/', name=ort, ort=ort,
                          pixdim=[pixdim_recon, pixdim_recon, s_thickness])
                print(ort, 'save done!')
            np.save(output_path + name + '/transform_gt.npy', output_transform)
            print('motion parameters save done!')
        else:
            raise ValueError('Invalid motion type!')
