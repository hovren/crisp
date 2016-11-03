# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

"""
Video slice module
"""

__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2015, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

import logging
logger = logging.getLogger(__name__)

import cv2
import numpy as np

from . import rotations
from . import tracking
        
class Slice(object):
    def __init__(self, start, end, points):
        self.points = points
        self.start = start
        self.end = end
        self.axis = None
        self.angle = None
        self.inliers = []
        
    def estimate_rotation(self, camera, ransac_threshold=7.0):
        """Estimate the rotation between first and last frame

        It uses RANSAC where the error metric is the reprojection error of the points
        from the last frame to the first frame.

        Parameters
        -----------------
        camera : CameraModel
            Camera model
        ransac_threshold : float
            Distance threshold (in pixels) for a reprojected point to count as an inlier
        """
        if self.axis is None:
            x = self.points[:, 0, :].T
            y = self.points[:, -1, :].T
            inlier_ratio = 0.5
            R, t, dist, idx = rotations.estimate_rotation_procrustes_ransac(x, y,
                                                                     camera, 
                                                                     ransac_threshold,
                                                                     inlier_ratio=inlier_ratio,
                                                                     do_translation=False)
            
            if R is not None:                                      
                self.axis, self.angle = rotations.rotation_matrix_to_axis_angle(R)
                if self.angle < 0: # Constrain to positive angles
                    self.angle = -self.angle
                    self.axis = -self.axis
                self.inliers = idx
                                                          
        return self.axis is not None

    @staticmethod
    def from_stream_randomly(video_stream, step_bounds=(5, 15), length_bounds=(2, 15), max_start=None, min_distance=10, min_slice_points=10):
        """Create slices from a video stream using random sampling

        Parameters
        -----------------
        video_stream : VideoStream
            A video stream
        step_bounds : tuple
            Range bounds (inclusive) of possible step lengths
        length_bounds : tuple
            Range bounds (inclusive) of possible slice lengths
        max_start : int
            Maximum frame number to start from
        min_distance : float
            Minimum (initial) distance between tracked points
        min_slice_points : int
            Minimum number of points to keep a slice

        Returns
        -------------------
        list of Slice
            List of slices
        """
        new_step = lambda: int(np.random.uniform(low=step_bounds[0], high=step_bounds[1]))
        new_length = lambda: int(np.random.uniform(low=length_bounds[0], high=length_bounds[1]))
        
        seq_frames = []
        slices = []
        seq_start_points = None
        next_seq_start = new_step() if max_start is None else min(new_step(), max_start)
        next_seq_length = new_length()
        for i, im in enumerate(video_stream):            
            if next_seq_start <= i < next_seq_start + next_seq_length:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                seq_frames.append(im)
                
                if len(seq_frames) == 1:
                    max_corners = 400
                    quality_level = 0.07
                    seq_start_points = cv2.goodFeaturesToTrack(im, max_corners, quality_level, min_distance)
                    
                elif len(seq_frames) == next_seq_length:                               
                    points, status = tracking.track_retrack(seq_frames, seq_start_points)
                    if points.shape[0] >= min_slice_points:
                        s = Slice(next_seq_start, i, points)
                        slices.append(s)
                        logger.debug('{0:4d} {1:3d} {2:5d} {3:>5d}-{4:<5d}'.format(len(slices)-1, points.shape[1], points.shape[0], next_seq_start, i))
                    seq_frames = []
                    next_seq_start = i + new_step()
                    next_seq_length = new_length()
        
        return slices

def fill_sampling(slice_list, N):
    """Given a list of slices, draw N samples such that each slice contributes as much as possible

    Parameters
    --------------------------
    slice_list : list of Slice
        List of slices
    N : int
        Number of samples to draw
    """
    A = [len(s.inliers) for s in slice_list]
    N_max = np.sum(A)
    if N > N_max:
        raise ValueError("Tried to draw {:d} samples from a pool of only {:d} items".format(N, N_max))
    
    samples_from = np.zeros((len(A),), dtype='int') # Number of samples to draw from each group

    remaining = N
    while remaining > 0:
        remaining_groups = np.flatnonzero(samples_from - np.array(A))
        
        if remaining < len(remaining_groups):
            np.random.shuffle(remaining_groups)
            for g in remaining_groups[:remaining]:
                samples_from[g] += 1
        else:
            # Give each group the allowed number of samples. Constrain to their max size.
            to_each = max(1, int(remaining / len(remaining_groups)))
            samples_from = np.min(np.vstack((samples_from + to_each, A)), axis=0)
        
        # Update remaining count
        remaining = int(N - np.sum(samples_from))
    if not remaining == 0:
        raise ValueError("Still {:d} samples left! This is an error in the selection.")

    # Construct index list of selected samples
    samples = []
    for s, a, n in zip(slice_list, A, samples_from):
        if a == n:
            samples.append(np.array(s.inliers)) # all
        elif a == 0:
            samples.append(np.arange([]))
        else:
            chosen = np.random.choice(s.inliers, n, replace=False)
            samples.append(np.array(chosen))
    return samples
