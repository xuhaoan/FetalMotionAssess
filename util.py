"""
Utils for motion simulation

@author: Haoan Xu, ZJU BME, 2021/7

"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def rotate_matrix(theta=[0, 0, 0], ort='axi'):
    """
    Argument:
        theta: three rotation angle [x,y,z]
        ort: orientation of stack

    return 4*4 rotation matrix Rx-Ry-Rz-location

    """
    theta_x = np.deg2rad(theta[0])
    cx = np.cos(theta_x)
    sx = np.sin(theta_x)

    theta_y = np.deg2rad(theta[1])
    cy = np.cos(theta_y)
    sy = np.sin(theta_y)

    theta_z = np.deg2rad(theta[2])
    cz = np.cos(theta_z)
    sz = np.sin(theta_z)

    Rx = [[1, 0, 0, 0],
          [0, cx, -sx, 0],
          [0, sx, cx, 0, ],
          [0, 0, 0, 1]]

    Ry = [[cy, 0, sy, 0],
          [0, 1, 0, 0],
          [-sy, 0, cy, 0],
          [0, 0, 0, 1]]

    Rz = [[cz, -sz, 0, 0],
          [sz, cz, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]

    rot = np.matmul(Rx, Ry)
    rot = np.matmul(rot, Rz)

    # define three standard orientations
    if ort == 'axi':
        ort_rot = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]]

    elif ort == 'cor':
        ort_rot = [[1, 0, 0, 0],
                   [0, 0, -1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]]

    elif ort == 'sag':
        ort_rot = [[0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [-1, 0, 0, 0],
                   [0, 0, 0, 1]]

    else:
        raise ValueError('Orientation notation is invalid! Should be axi or sag or cor. Got {}'.format(ort))

    rot = np.matmul(rot, ort_rot)

    return rot


def displacement_matrix(disp=[0, 0, 0], pixrecon=0.8):
    """
    Compute displacement matrix (addictive)

    """

    disp_matrix = np.zeros((4, 4))
    disp_matrix[0, 3] += disp[0] * pixrecon
    disp_matrix[1, 3] += disp[1] * pixrecon
    disp_matrix[2, 3] += disp[2] * pixrecon

    return disp_matrix


def nii_affine(pixdim, pixrecon, vol_shape,
               ort='axi', params=[0, 0, 0, 0, 0, 0]):
    """
    Export the Nifty affine information and transformation info.

    """
    # scale matrix
    scale_matrix = [[pixdim[0], 0, 0, 0],
                    [0, pixdim[1], 0, 0],
                    [0, 0, pixdim[2], 0],
                    [0, 0, 0, 1]]

    # displacement
    standard_matrix = [[1, 0, 0, -(vol_shape[0] - 1) / 2],
                       [0, 1, 0, -(vol_shape[1] - 1) / 2],
                       [0, 0, 1, -(vol_shape[2] - 1) / 2],
                       [0, 0, 0, 1]]

    st_crd = np.matmul(scale_matrix, standard_matrix)
    rot = rotate_matrix(params[3:], ort=ort)
    disp = displacement_matrix(params[:3], pixrecon)

    affine = disp + np.matmul(rot, st_crd)

    return affine


def motion_affine(motion_corrupt=[0, 0, 0, 0, 0, 0], global_offset=[0, 0, 0, 0, 0, 0], ort='axi', pixrecon=0.8):
    """
    Calculate the motion transformation parameter

    Note: there is a coordinate transform between SimpleITK and the
          physical coordinate

    """
    # coordinate transform as sitk follow [z,y,x]
    A = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    rot_all = rotate_matrix(motion_corrupt[3:])
    disp_all = displacement_matrix(motion_corrupt[:3], pixrecon)
    glb_all = disp_all + rot_all
    glb_all = np.matmul(A, glb_all)

    rot_offset = rotate_matrix(global_offset[3:])
    disp_offset = displacement_matrix(global_offset[:3], pixrecon)
    glb_offset = disp_offset + rot_offset
    glb_offset = np.matmul(A, glb_offset)

    rigid_motion = np.matmul(glb_all, np.linalg.inv(glb_offset))
    motion_rot = rigid_motion[:3, :3]
    motion_disp = rigid_motion[:3, 3]

    return motion_rot, motion_disp


def save_data(data, save_path, name,
              params=[0., 0., 0., 0., 0., 0., ],
              ort='axi', mask=True,
              pixdim=[0.8, 0.8, 4.], verbose=False):
    """
    Save motion-corrupted stacks as .nii.gz
    Data should be normalized from 0 to 1

    Arguments:
        - params [theta_x, theta_y, theta_z,
                  disp_x,  disp_y,  disp_z]

    """
    vol_shape = data.shape
    mask = np.zeros(vol_shape)
    mask[data > 0] = 1

    assert len(vol_shape) == 3, 'input should be one 3D volume!'

    img_affine = nii_affine(pixdim, pixrecon=0.8, vol_shape=vol_shape,
                            ort=ort, params=params)

    empty_header = nib.Nifti1Header()
    img = nib.Nifti1Image(data, img_affine, empty_header)

    # set scanner coordinate
    img.header['sform_code'] = 1

    # define the standard space as 1*1*1 mm3
    img.header['xyzt_units'] = 10

    # set pixel dim
    img.header['pixdim'] = [-1.] + pixdim + [0., 0., 0., 0.]

    mask = nib.Nifti1Image(mask, img.affine, img.header)

    if verbose:
        print(img.affine)

    nib.save(img, save_path + name + '.nii.gz')

    if mask:
        nib.save(mask, save_path + name + '_mask.nii.gz')

        return True

    else:

        return True


def save_mc(path, rigid_transform, global_params, ort, slice_idx, pixdim_recon=0.8):
    """
    Save motion correction parameters (rigid/affine)

    a small interface with regards with NiftyMIC reconstruction

    rigid_transform [6,]
    global_params list 6 [dis,dis,dis,rot,rot,rot]


    """
    motion_rot, motion_disp = motion_affine(rigid_transform, global_params,
                                            pixrecon=pixdim_recon, ort='axi')

    motion_rot = np.ndarray.flatten(motion_rot)
    rigid_transform_sitk = sitk.AffineTransform(3)
    rigid_transform_sitk.SetMatrix(motion_rot)
    rigid_transform_sitk.SetTranslation(motion_disp)
    path_transform = os.path.join(path, '%s_slice%d.tfm' % (ort, slice_idx))
    sitk.WriteTransform(rigid_transform_sitk, path_transform)

    return True

def image_resample(img, spacing):
    """
    An image resample method presented in simplereg Ebner.

    :param img: input image in .nii
    :param spacing: output spacing, a list with length of image dimension
    :return: resampled image
    """
    spacing_in = np.array(img.GetSpacing())
    size_in = np.array(img.GetSize()).astype(int)
    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection())

    spacing_out = np.atleast_1d(spacing).astype(np.float64)
    size = size_in * spacing_in / spacing_out
    size = np.round(size).astype(int)
    size = [int(i) for i in size]

    resampled_img = sitk.Resample(img, size, getattr(sitk, "Euler%dDTransform" % img.GetDimension())(),
                                  sitk.sitkLinear, origin,
                                  spacing_out,
                                  direction,
                                  0,
                                  img.GetPixelIDValue()
                                  )

    return resampled_img


def image_crop_pad(x, target_size, shift=[0, 0, 0]):
    """
    Crop or zero-pad the 3D .numpy volume to the target size

    :param x: input 3D volume in .numpy
    :param target_size: output size of volume
    :param shift:
    :return: cropped or padded volume with target size
    """
    x = np.asarray(x)
    current_size = x.shape
    pad_size = [0, 0, 0]
    for dim in range(3):
        if current_size[dim] > target_size[dim]:
            pad_size[dim] = 0
        else:
            pad_size[dim] = int(np.ceil((target_size[dim] - current_size[dim]) / 2.0))  # np.ceil向上取整
    x1 = np.pad(x, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [pad_size[2], pad_size[2]]], 'constant',
                constant_values=0)
    start_pos = np.ceil(
        (np.asarray(x1.shape) - np.asarray(target_size)) / 2.0)
    start_pos = start_pos.astype(int)
    y = x1[(shift[0] + start_pos[0]):(shift[0] + start_pos[0] + target_size[0]),
        (shift[1] + start_pos[1]):(shift[1] + start_pos[1] + target_size[1]),
        (shift[2] + start_pos[2]):(shift[2] + start_pos[2] + target_size[2])]

    return y