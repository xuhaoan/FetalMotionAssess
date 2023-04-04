"""
This file contains simulation of linear and random motion. For linear motion, the magnitude of displacement and
rotation needs to be set in advance. Relative motion indicator (RMI) can be calculated to assess the motion
sensitivity of the different methods. For random motion, motion-corrupted stacks are simulated with random motion
trajectory to match real-world situation. Success rate can be calculated based on this simulation. Interleaved
acquisition has been considered.

@author: Haoan Xu, ZJU BME, 2021/9

"""

import random
import numpy as np
from motion_simulation.motion_generator import MotionGenerator


def random_motion_simulation(motion_generator, volume, transform, parity, ort, num_slice):
    """
    Simulate random motion with a input transformation matrix. Considering interleaved acquisition, the first half
    and the second half of the transformation matrix are the motion parameters of the odd and even slices
    respectively.

    :param motion_generator: MotionGenerator
    :param volume: input 3D isotropic volume in .numpy
    :param transform: transformation matrix
    :param parity: for interleaved acquisition
    :param ort: orientation
    :param num_slice: number of slices
    :return: simulated stack with interleaved random motion
    """
    if parity == 'odd':
        transform_stack = transform[:, :num_slice, :]
    else:
        transform_stack = transform[:, num_slice:, :]
    simulated_stack = motion_generator._get_motion_corrupted_slice(volume=volume, rigid_transform=transform_stack,
                                                                   vol_shape=np.shape(volume), ort=ort)

    return simulated_stack


def interleaved_motion_simulation(volume, num_slice, st_ratio, orts):
    """
    Simulate random motion in an interleaved manner.

    :param volume: input 3D isotropic volume in .numpy
    :param num_slice: number of slices in simulated stacks
    :param st_ratio: slice thickness/recon isotropic resolution
    :param orts: orientations
    :return: simulated stacks and transformation matrix
    """
    nb_stack = len(orts)
    motion_generator = MotionGenerator(num_slice=num_slice, num_point=2 * num_slice, st_ratio=st_ratio)
    output_transform = np.zeros((1, int(num_slice * nb_stack), 6))
    output_stacks = []
    motion_free_index = random.randint(0,2)
    for index_ort, ort in enumerate(orts):
        transform = motion_generator._get_motion()
        if motion_free_index == index_ort:
            transform *= 0.05
        odd_stack = random_motion_simulation(motion_generator, volume, transform, 'odd', ort, num_slice)
        even_stack = random_motion_simulation(motion_generator, volume, transform, 'even', ort, num_slice)
        output_stack_temp = np.zeros([*np.shape(volume)[0:2], num_slice])
        for i in range(num_slice):
            if i % 2 == 0:
                output_stack_temp[:, :, i] = odd_stack[:, :, i]
                output_transform[0, num_slice * index_ort + i, :] = transform[0, i, :]
            else:
                output_stack_temp[:, :, i] = even_stack[:, :, i]
                output_transform[0, num_slice * index_ort + i, :] = transform[0, num_slice + i, :]
        output_stacks.append(output_stack_temp)

    return output_stacks, output_transform, motion_free_index


def linear_motion_simulation(volume, displacement, rotation, num_slice, st_ratio, orts):
    """
    Simulate linear motion. In-plane motion simulation is recommended to control the variables in the analysis of
    different methods, while not using large motion magnitude with through-plane motion simulation.

    :param volume: input 3D isotropic volume in .numpy
    :param displacement: magnitude of displacement
    :param rotation: magnitude of rotation
    :param num_slice: number of slices
    :param st_ratio: slice thickness/recon isotropic resolution
    :param orts: orientations
    :return: simulated stacks and transformation matrix
    """
    nb_stack = len(orts)
    motion_generator = MotionGenerator(num_slice=num_slice, num_point=num_slice, st_ratio=st_ratio)
    output_transform = np.zeros((1, int(num_slice * nb_stack), 6))
    output_stacks = []

    for index_ort, ort in enumerate(orts):
        transform_stack = in_plane_simulation(displacement, rotation, num_slice, ort)
        # transform_stack = through_plane_simulation(displacement, rotation, num_slice)

        output_transform[0, num_slice * index_ort:num_slice * (index_ort + 1), :] = transform_stack
        output_stack_temp = motion_generator._get_motion_corrupted_slice(volume=volume, rigid_transform=transform_stack,
                                                                         vol_shape=np.shape(volume), ort=ort)
        output_stacks.append(output_stack_temp)

    return output_stacks, output_transform


def in_plane_simulation(displacement, rotation, num_slice, ort):
    """
    Linear in-plane motion simulation, which is recommended. For a stack in one direction (e.g. axi), displacement
    along this axis (z) and rotations not about this axis (x/y) are considered through-plane motions and are both set
    to zero.

    :param displacement: magnitude of displacement
    :param rotation: magnitude of rotation
    :param num_slice: number of slices
    :param ort: orientation
    :return: transformation matrix of the input stack
    """
    transform_base = np.reshape(np.array(range(num_slice)) + 1, [num_slice, 1])
    transform_zero = np.zeros([num_slice, 1])
    if ort == 'axi':
        transform_T = displacement * np.concatenate([transform_base, transform_base, transform_zero], axis=-1)
        transform_R = rotation * np.concatenate([transform_zero, transform_zero, transform_base], axis=-1)
    elif ort == 'cor':
        transform_T = displacement * np.concatenate([transform_base, transform_zero, transform_base], axis=-1)
        transform_R = rotation * np.concatenate([transform_zero, transform_base, transform_zero], axis=-1)
    else:
        transform_T = displacement * np.concatenate([transform_zero, transform_base, transform_base], axis=-1)
        transform_R = rotation * np.concatenate([transform_base, transform_zero, transform_zero], axis=-1)

    transform_stack = np.concatenate([transform_T, transform_R], axis=-1)
    transform_stack = transform_stack[np.newaxis, :, :]

    return transform_stack


def through_plane_simulation(displacement, rotation, num_slice):
    """
    Linear through-plane motion simulation.

    :param displacement: magnitude of displacement
    :param rotation: magnitude of rotation
    :param num_slice: number of slices
    :return: transformation matrix of the input stack
    """
    assert displacement >= 1, ValueError('Excessive displacement')
    transform_base = np.reshape(np.array(range(num_slice)) + 1, [num_slice, 1])
    transform_T = displacement * np.concatenate([transform_base, transform_base, transform_base], axis=-1)
    transform_R = rotation * np.concatenate([transform_base, transform_base, transform_base], axis=-1)

    transform_stack = np.concatenate([transform_T, transform_R], axis=-1)
    transform_stack = transform_stack[np.newaxis, :, :]

    return transform_stack


def motion_free_simulation(volume, num_slice, st_ratio, orts):
    """
    Simulation of motion-free stacks for baseline motion indicator calculation.

    :param volume: input 3D isotropic volume in .numpy
    :param num_slice: number of slices
    :param st_ratio: slice thickness/recon isotropic resolution
    :param orts: orientations
    :return: transformation matrix of the input stack
    """
    transform_stack = np.zeros([1, num_slice, 6])
    output_stacks = []
    output_transform = np.zeros([1, num_slice * len(orts), 6])
    motion_generator = MotionGenerator(num_slice=num_slice, num_point=num_slice, st_ratio=st_ratio)
    for index_ort, ort in enumerate(orts):
        output_stack_temp = motion_generator._get_motion_corrupted_slice(volume=volume, rigid_transform=transform_stack,
                                                                         vol_shape=np.shape(volume), ort=ort)
        output_stacks.append(output_stack_temp)
    return output_stacks, output_transform
