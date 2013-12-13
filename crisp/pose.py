# -*- coding: utf-8 -*-
"""
Relative pose calibration module
"""

__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2013, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

import logging
logger = logging.getLogger()

import numpy as np
import matplotlib.pyplot as plt
import cv2

from . import timesync
from . import tracking
from . import rotations

def estimate_pose(image_sequences, imu_sequences, K):
    """Estimate sync between IMU and camera based on gyro readings and optical flow.
    
    The user should first create at least two sequences of corresponding image and 
    gyroscope data.
    From each sequence we calculate the rotation axis (one from images, one from IMU/gyro).
    The final set of len(image_sequences) corresponding rotation axes are then used to calculate
    the relative pose between the IMU and camera.
    
    The returned rotation is such that it transfers vectors in the gyroscope coordinate
    frame to the camera coordinate frame:
    
        X_camera = R * X_gyro
    
    
    Parameters
    ------------
    image_sequences : list of list of ndarrays 
            List of image sequences (list of ndarrays) to use. Must have at least two sequences.
    imu_sequences : list of (3, N) ndarray
            Sequence of gyroscope measurements (angular velocities).
    K : (3,3) ndarray
            Camera calibration matrix
            
    Returns
    -----------
    R : (3,3) ndarray
            The relative pose (gyro-to-camera) such that X_camera = R * X_gyro
    """
    assert len(image_sequences) == len(imu_sequences)
    assert len(image_sequences) >= 2
    # Note: list(image_sequence) here makes sure any generator type input is expanded to an actual list
    sync_correspondences = [_get_point_correspondences(list(image_sequence)) for image_sequence in image_sequences]
    
    # ) Procrustes on corresponding pairs
    PROCRUSTES_MAX_POINTS = 15 # Number of tracks/points to use for procrustes
    logger.debug("Running procrustes on track-retrack results")
    image_rotation_axes = []
    for i, points in enumerate(sync_correspondences):
        if points.size < 1:
            logger.error('Shape of points are %s', str(points.shape))
            raise Exception("Did not get enough points when tracking")
        num_points_to_use = min(PROCRUSTES_MAX_POINTS, points.shape[0])
        logger.debug("Using %d tracks to calculate procrustes", num_points_to_use)
        idxs_to_use = np.random.permutation(points.shape[0])[:num_points_to_use]
        assert points.shape[-1] == 2
        x = points[idxs_to_use,0,:].T.reshape(2,-1)
        y = points[idxs_to_use,-1,:].T.reshape(2,-1)

        x = np.vstack((x, np.ones((1, x.shape[1]))))
        y = np.vstack((y, np.ones((1, y.shape[1]))))

        K_inv = np.linalg.inv(K)
        X = K_inv.dot(x)
        Y = K_inv.dot(y)

        # Depth must be positive
        (R, t) = rotations.procrustes(X, Y, remove_mean=False) # X = R * Y + t
        (v, theta) = rotations.rotation_matrix_to_axis_angle(R)
        image_rotation_axes.append(v) # Save rotation axis
        
        # Check the quality via the mean reprojection error
        mean_error = np.mean(np.sqrt(np.sum((X - R.dot(Y))**2, axis=0)))
        MEAN_ERROR_LIMIT = 0.1 # Arbitrarily chosen limit (in meters)
        logger.debug('Image sequence %d: Rotation axis %s, degrees %.2f, mean error %.3f',
            i, v, np.rad2deg(theta), mean_error)
        if mean_error > MEAN_ERROR_LIMIT: 
            logger.warning("Procrustes solution mean error %.3f > %.3f", mean_error, MEAN_ERROR_LIMIT)

    # ) Gyro principal rotation axis
    gyro_rotation_axes = []
    for i, gyro_seq in enumerate(imu_sequences):
        assert gyro_seq.shape[0] == 3
        v = principal_rotation_axis(gyro_seq)
        logger.debug('Gyro sequence %d: Rotation axis %s', i, v)
        gyro_rotation_axes.append(v)
        
    # ) Procrustes to get rotation between coordinate frames
    X = np.vstack(image_rotation_axes).T
    Y = np.vstack(gyro_rotation_axes).T
    (R,t) = rotations.procrustes(X, Y, remove_mean=False)

    return (R, t)

#--------------------------------------------------------------------------

def pick_manual(image_sequence, imu_gyro, num_sequences=2):
    """Select N matching sequences and return data indices.
    
    Parameters
    ---------------
    image_sequence : list_like
            A list, or generator, of image data
    imu_gyro : (3, N) ndarray
            Gyroscope data (angular velocities)
    num_sequences : int
            The number of matching sequences to pick
    
    Returns
    ----------------
    sync_sequences : list
            List of (frame_pair, gyro_pair) tuples where each pair contains
            (a, b) which are indices of the (inclusive) range [a, b] that was chosen
    """
    assert num_sequences >= 2    
    # Create optical flow for user to select parts in
    logger.info("Calculating optical flow")
    flow = tracking.optical_flow_magnitude(image_sequence)
    
    # ) Prompt user for sync slices
    logger.debug("Prompting user for %d sequences" % num_sequences)
    imu_fake_timestamps = np.linspace(0,1,num=imu_gyro.shape[1])
    sync_sequences = [timesync.manual_sync_pick(flow, imu_fake_timestamps, imu_gyro) for i in range(num_sequences)]

    return sync_sequences

#--------------------------------------------------------------------------

def principal_rotation_axis(gyro_data):
    """Get the principal rotation axis of angular velocity measurements.    
    
    Parameters
    -------------
    gyro_data : (3, N) ndarray
            Angular velocity measurements
           
    Returns
    -------------
    v : (3,1) ndarray
            The principal rotation axis for the chosen sequence
    """
    N = np.zeros((3,3))
    for x in gyro_data.T: # Transpose because samples are stored as columns
        y = x.reshape(3,1)
        N += y.dot(y.T)
        
    (eig_val, eig_vec) = np.linalg.eig(N)
    i = np.argmax(eig_val)
    v = eig_vec[:,i]
    
    # Make sure v has correct sign
    s = 0
    for x in gyro_data.T: # Transpose because samples are stored as columns
        s += v.T.dot(x.reshape(3,1))
        
    v *= np.sign(s)
    
    return v
    
#--------------------------------------------------------------------------

def _get_point_correspondences(image_list, max_corners=200, min_distance=5, quality_level=0.07):
    max_retrack_distance = 0.5
    initial_points = cv2.goodFeaturesToTrack(image_list[0], max_corners, quality_level, min_distance)
    (tracks, status) = tracking.track_retrack(image_list, initial_points=initial_points, max_retrack_distance=max_retrack_distance) # Status is ignored
    return tracks[:,(0,-1),:] # First and last frame only