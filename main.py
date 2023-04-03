"""
This file is the main function for reference stack selection, and is mainly used to assess random motion and
determine the stack with minimum motion. Stacks with motions (list of .nii images) are preformed and the stack with
minimum motion indicator is selected as the reference.

@author: Haoan Xu, ZJU BME, 2022/5

"""

import os
import SimpleITK as sitk
from methods import CP, SVD_FS, SVD_RSS

if __name__ == '__main__':
    method_type = 'CP'
    path = './Data/RandomMotionStacks/'

    orts = ['axi', 'cor', 'sag']
    names = os.listdir(path)
    nb_slice = 24

    for name in names:
        print('****** Processing', name, '******')
        motion_indicator = []
        for i in range(len(orts)):
            img = sitk.ReadImage(path + name + '/' + orts[i] + '.nii.gz')
            if method_type == 'CP':
                motion_indicator_temp = CP(img=img, spacing=[2, 2, 2])
            elif method_type == 'SVD_RSS':
                motion_indicator_temp = SVD_RSS(img=img, spacing=[2, 2, 2], size=80)
            elif method_type == 'SVD_FS':
                motion_indicator_temp = SVD_FS(img=img)
            else:
                raise ValueError('Invalid method type!')
            motion_indicator.append(motion_indicator_temp)

        reference_index = motion_indicator.index(min(motion_indicator))
        reference = orts[reference_index] + '.nii.gz'
        with open(path + name + '/motion_free_index.txt', 'w') as f:
            f.write(reference)
        print('The selected reference stack by ' + method_type + ' is: ' + reference)
        if os.path.exists(path + name + '/motion_free_index_gt.txt'):
            with open(path + name + '/motion_free_index_gt.txt', 'r') as f_gt:
                reference_gt = f_gt.read()
            print('The simulated motion-free stack is: ' + reference_gt)
            if reference == reference_gt:
                print('Successful assessment!')
            else:
                print('Failed assessment!')
