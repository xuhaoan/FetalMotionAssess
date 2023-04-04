"""
This file is used to evaluate relative motion indicator (RMI) for different methods, corresponding to Linear
Motion Experiment in the manuscript. Motion-free and motion-corrupted stacks (list of .nii images) are processed and RMI
is calculated through MI(mc)/MI(mf).

@author: Haoan Xu, ZJU BME, 2022/5

"""

import os
import SimpleITK as sitk
from methods import CP, SVD_FS, SVD_RSS

if __name__ == '__main__':
    method_type = 'SVD_FS'
    MC_path = 'data/linear_motion_stacks/D_0.4_R_2/'
    MF_path = 'data/motion_free_stacks/'

    orts = ['axi', 'cor', 'sag']
    names = os.listdir(MC_path)
    nb_slice = 24

    for name in names:
        print('****** Processing', name, '******')
        relative_motion_indicator = []
        for i in range(len(orts)):
            img_MC = sitk.ReadImage(MC_path + name + '/' + orts[i] + '.nii.gz')
            img_MF = sitk.ReadImage(MF_path + name + '/' + orts[i] + '.nii.gz')
            if method_type == 'CP':
                motion_indicator_MC = CP(img=img_MC, spacing=[2, 2, 2])
                motion_indicator_MF = CP(img=img_MF, spacing=[2, 2, 2])
                RMI_temp = motion_indicator_MC / motion_indicator_MF
            elif method_type == 'SVD_RSS':
                motion_indicator_MC = SVD_RSS(img=img_MC, spacing=[2, 2, 2])
                motion_indicator_MF = SVD_RSS(img=img_MF, spacing=[2, 2, 2])
                RMI_temp = motion_indicator_MC / motion_indicator_MF
            elif method_type == 'SVD_FS':
                motion_indicator_MC = SVD_FS(img=img_MC)
                motion_indicator_MF = SVD_FS(img=img_MF)
                RMI_temp = motion_indicator_MC / motion_indicator_MF
            else:
                raise ValueError('Invalid method type!')
            relative_motion_indicator.append(RMI_temp)

        print('Relative Motion Indicator in', method_type)
        print('RMI =', relative_motion_indicator)