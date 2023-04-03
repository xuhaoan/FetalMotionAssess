"""
This file contains three motion assessment methods. CP is the recommended method considering accuracy and time
complexity. SVD_RSS is a modified version of SVD_FS, which extract motion information along all axes based on
re-slicing stacks. SVD_FS was proposed in NeuroImage Ebner2020 and TMI Kainz2015, which preforms SVD in the flatten
stacks.

@author: Haoan Xu, ZJU BME, 2021/11

"""

import os
import numpy as np
import SimpleITK as sitk
import tensorly as tl
from tensorly.decomposition import parafac
from util import image_resample, image_crop_pad


def CP(img, spacing, rank=25):
    """
    Motion assessment method based on CP decomposition. After interpolation, the volume is decomposed by  CP into
    r rank-one tensors, and the difference between the original tensor and rank-one tensors is considered as
    motion indicator.

    :param img: input image of stack in .nii
    :param spacing: spacing of interpolated stack
    :param rank: the number of rank-one tensors
    :return: motion indicator
    """
    img_resampled = image_resample(img, spacing)

    volume = sitk.GetArrayFromImage(img_resampled)
    volume = (volume - np.amin(volume)) / (np.amax(volume) - np.amin(volume))
    v = len(np.where(volume != 0)[0])

    weights, factors = parafac(volume, rank=rank)
    low_rank_volume = tl.kruskal_to_tensor((weights, factors))

    motion_indicator = np.mean((volume - low_rank_volume) ** 2) / v * 1000000000

    return motion_indicator


def SVD_RSS(img, spacing, size=None, rank=5):
    """
    Motion assessment method based on singular value decomposition, which is performed on every re-sliced stacks.
    After interpolation and cropping, low-rank components of every re-sliced images along three axes are extracted
    based SVD, and the sum of difference between the original matrix and its low-rank component is considered as
    motion indicator.

    :param img: input image of stack in .nii
    :param spacing: spacing of interpolated stack
    :param rank: the rank of low-rank component
    :return: motion indicator
    """
    img_resampled = image_resample(img, spacing)

    volume = sitk.GetArrayFromImage(img_resampled)
    num_slice = np.shape(volume)[1]
    if size:
        volume = image_crop_pad(volume, [size, size, size])
    else:
        volume = image_crop_pad(volume, [num_slice, num_slice, num_slice])
    volume = (volume - np.amin(volume)) / (np.amax(volume) - np.amin(volume))

    motion_indicator = 0
    for index_slice in range(np.shape(volume)[0]):
        for index_dimension in range(len(np.shape(volume))):
            if index_dimension == 0:
                slice_data = volume[index_slice, :, :]
            elif index_dimension == 1:
                slice_data = volume[:, index_slice, :]
            else:
                slice_data = volume[:, :, index_slice]
            s = len(np.where(slice_data != 0)[0])
            if np.linalg.norm(slice_data, ord='fro') > 0:
                U, S, VT = np.linalg.svd(slice_data)

                U1 = U[:, 0:rank]
                S1 = S[0:rank]
                S1 = np.diag(S1)
                VT1 = VT[0:rank, :]

                D1 = np.matmul(U1, S1)
                D1 = np.matmul(D1, VT1)
                motion_indicator_temp = np.linalg.norm(slice_data - D1, ord='fro') / np.linalg.norm(slice_data,
                                                                                                    ord='fro') / s * 100000
                motion_indicator = motion_indicator + motion_indicator_temp

    return motion_indicator


def SVD_FS(img, threshold=0.99):
    """
    Motion assessment method presented in NeuroImage Ebner2020 and TMI Kainz2015, but the version as found on the
    GitHub NiftyMIC repo. The rank of flattened matrix is considered as motion indicator.

    :param img: input image of stack in .nii
    :param threshold: error threshold
    :return: motion indicator
    """
    volume = sitk.GetArrayFromImage(img)
    volume = volume.reshape(volume.shape[0], -1).transpose()
    U, S, VT = np.linalg.svd(volume)
    S2 = np.square(S)
    A_norm = np.sum(S2)
    delta_r = np.sqrt(np.array([np.sum(S2[0:r + 1]) / A_norm for r in range(len(S2))]))
    r = np.where(delta_r < threshold)[0][-1] + 1
    motion_indicator = r * delta_r[r - 1]

    return motion_indicator

