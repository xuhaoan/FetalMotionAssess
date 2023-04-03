"""
This file contains InterSliceMotionGenerator for simulation and motion assessment

@author: Haoan Xu, ZJU BME, 2021/7

"""

import SimpleITK as sitk
import numpy as np
import random
from scipy.interpolate import InterpolatedUnivariateSpline
from util import rotate_matrix


class MotionGenerator(object):
    """
    This is a generator for inter-slice motion simulation. It takes an isotropic volume as input and can output
    simulated motion and the motion-corrupted stack.

    """

    def __init__(self,
                 num_slice,
                 num_point,
                 st_ratio,
                 c_point=[3],
                 motion_type='Euler',
                 g_angle=5,
                 g_disp_xy=1,
                 g_disp_z=1,
                 l_angle=5,
                 l_disp_xy=1,
                 l_disp_z=1,
                 trs_angle=20,
                 trs_disp_xy=5,
                 trs_disp_z=5):
        """

        Argument:
            num_slice: num of slices
            num_point: num of control points
            st_ratio: slice thickness/recon isotropic resolution
            c_point: control point setting
            motion_type: rigid motion parameterization type, currently only support 'Euler'
            g_angle: global rotation offset range (degree)
            g_disp_xy: global in-plane displacement offset range (unit: voxel)
            g_disp_z: global through-plane displacement offset range (unit:voxel)
            l_angle: local rotation offset range (degree)
            l_disp_xy: local in-plane displacement offset range (unit: voxel)
            l_disp_z: local through-plane displacement offset range (unit:voxel)
            trs_angle: threshold of the maximum rotation angle (degree)
            trs_disp_xy: threshold of the maximum in-plane displacement
            trs_disp_z: threshold of the maximum through-plane displacement


        """

        self.num_slice = num_slice
        self.num_point = num_point
        self.st_ratio = st_ratio
        self.c_point = c_point
        self.motion_type = motion_type
        self.g_angle = g_angle
        self.g_disp_xy = g_disp_xy
        self.g_disp_z = g_disp_z
        self.l_angle = l_angle
        self.l_disp_xy = l_disp_xy
        self.l_disp_z = l_disp_z
        self.trs_angle = trs_angle
        self.trs_disp_xy = trs_disp_xy
        self.trs_disp_z = trs_disp_z

        assert self.motion_type in ['Euler', 'Quaternion'], ValueError('Motion type selection is not valid!')

    def _set_angle(self, scope_angle):
        """
        Generate the trajectory of angle
        :param scope_angle: control the average angle/sec

        """

        idx = [0]
        angle = [0]
        c_point = random.choice(self.c_point)

        while True:
            step = np.random.randint(2, c_point + 1)
            temp = idx[-1] + step
            if temp < self.num_point - step:
                idx += [temp]
                angle += [angle[-1] + np.random.uniform(-scope_angle, scope_angle)]
            else:
                break

        idx += [self.num_point - 1]
        angle += [angle[-1] + np.random.uniform(-scope_angle, scope_angle)]
        angle -= np.mean(angle)

        return idx, angle

    def _set_disp(self, scope_disp):
        """
        Generate the trajectory of displacement
        :param scope_disp: control the average disp/sec

        """
        idx = [0]
        disp = [0]
        c_point = random.choice(self.c_point)

        while True:
            step = np.random.randint(2, c_point + 1)
            temp = idx[-1] + step
            if temp < self.num_point - step:
                idx += [temp]
                disp += [disp[-1] + np.random.uniform(-scope_disp, scope_disp)]
            else:
                break

        idx += [self.num_point - 1]
        disp += [disp[-1] + np.random.uniform(-scope_disp, scope_disp)]
        disp -= np.mean(disp)

        return idx, disp

    def _get_motion(self):

        """

        This function is used to generate random motion trajectory with six rigid transformation parameters in Euler
        representation.

        default TR = 800ms

        """

        idx_slice = np.linspace(0, self.num_point - 1, self.num_point)

        global_rotation = np.random.uniform(-self.g_angle, self.g_angle, 3)
        global_displacement_xy = np.random.uniform(-self.g_disp_xy, self.g_disp_xy, 2)
        global_displacement_z = np.random.uniform(-self.g_disp_z, self.g_disp_z, 1)
        global_displacement = np.concatenate((global_displacement_xy, global_displacement_z))

        ax, ay, az, dx, dy, dz = [1000], [1000], [1000], [1000], [1000], [1000]
        while np.any(np.abs(ax + global_rotation[0]) >= self.trs_angle): idx_ax, ax = self._set_angle(self.l_angle)
        while np.any(np.abs(ay + global_rotation[1]) >= self.trs_angle): idx_ay, ay = self._set_angle(self.l_angle)
        while np.any(np.abs(az + global_rotation[2]) >= self.trs_angle): idx_az, az = self._set_angle(self.l_angle)
        while np.any(np.abs(dx + global_displacement[0]) >= self.trs_disp_xy): idx_dx, dx = self._set_disp(
            self.l_disp_xy)
        while np.any(np.abs(dy + global_displacement[1]) >= self.trs_disp_xy): idx_dy, dy = self._set_disp(
            self.l_disp_xy)
        while np.any(np.abs(dz + global_displacement[2]) >= self.trs_disp_z): idx_dz, dz = self._set_disp(
            self.l_disp_z)

        # Spline interpolator
        f_x_angle = InterpolatedUnivariateSpline(idx_ax, ax)
        f_y_angle = InterpolatedUnivariateSpline(idx_ay, ay)
        f_z_angle = InterpolatedUnivariateSpline(idx_az, az)
        f_x_disp = InterpolatedUnivariateSpline(idx_dx, dx)
        f_y_disp = InterpolatedUnivariateSpline(idx_dy, dy)
        f_z_disp = InterpolatedUnivariateSpline(idx_dz, dz)

        x_disp = f_x_disp(idx_slice).reshape((1, -1, 1)) + global_displacement[0]
        y_disp = f_y_disp(idx_slice).reshape((1, -1, 1)) + global_displacement[1]
        z_disp = f_z_disp(idx_slice).reshape((1, -1, 1)) + global_displacement[2]

        x_angle = f_x_angle(idx_slice).reshape((1, -1, 1)) + global_rotation[0]
        y_angle = f_y_angle(idx_slice).reshape((1, -1, 1)) + global_rotation[1]
        z_angle = f_z_angle(idx_slice).reshape((1, -1, 1)) + global_rotation[2]

        displacement = np.concatenate((x_disp, y_disp), axis=-1)
        displacement = np.concatenate((displacement, z_disp), axis=-1)

        rotation = np.concatenate((x_angle, y_angle), axis=-1)
        rotation = np.concatenate((rotation, z_angle), axis=-1)

        # generate images according to the motion trajectory
        transform = np.concatenate((displacement, rotation), axis=-1)

        return transform

    def _get_motion_corrupted_slice(self,
                                    volume,
                                    rigid_transform,
                                    vol_shape,
                                    ort='axi'):
        """
        Simulate inter-slice motion-corrupted images

        :param volume: input volume in numpy
        :param rigid_transform: motion trajectory with shape of [num_slice, 6]
        :param vol_shape: shape of input volume
        :param ort: orientation of output stack

        Note: the order of xyz in itk format may be a little bit different
        """

        affine = sitk.AffineTransform(3)

        psf = [1]
        w_size = len(psf)

        # initialize the data in sitk
        volume = volume.transpose((2,1,0))
        data = sitk.GetImageFromArray(volume)
        data.SetOrigin([-(vol_shape[0] - 1) / 2, -(vol_shape[1] - 1) / 2, -(vol_shape[2] - 1) / 2])
        data.SetSpacing([1, 1, 1])
        data.SetDirection(np.eye(3).flatten())

        corrupted_slice = np.zeros((*vol_shape[:2], self.num_slice))
        idx = int((vol_shape[2] - (self.num_slice - 1) * self.st_ratio + 1) - w_size) / 2

        if idx < 0 or idx >= vol_shape[2]:
            raise ValueError('st_ratio set too large! Got idx equals to {}.'.format(idx))

        for i in range(self.num_slice):
            # rigid trans params
            motion_rot = rotate_matrix(rigid_transform[0, i, 3:], ort=ort)[:3, :3]
            motion_disp = rigid_transform[0, i, :3]
            affine.SetMatrix(motion_rot.ravel())
            affine.SetTranslation(motion_disp)
            mt_stack = sitk.Resample(data, data, affine, sitk.sitkLinear, 0, data.GetPixelIDValue())
            idx_slice = int(idx + i * self.st_ratio)
            slab = sitk.GetArrayFromImage(mt_stack)
            slab = slab.transpose((2, 1, 0))
            corrupted_slice[..., i] = np.sum(np.multiply(slab[..., idx_slice:int(idx_slice + w_size)], psf),
                                                   axis=-1)

        return corrupted_slice

