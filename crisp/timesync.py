# -*- coding: utf-8 -*-
"""
Time synchronization module
"""

__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2013, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

#--------------------------------------------------------------------------
# Includes
#--------------------------------------------------------------------------
import logging
logger = logging.getLogger()
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssig
import scipy.optimize
from matplotlib.mlab import normpdf

from . import tracking
from .imu import IMU
from . import rotations
from . import znccpyr

#--------------------------------------------------------------------------
# Public functions
#--------------------------------------------------------------------------

def sync_camera_gyro(image_sequence, image_timestamps, gyro_data, gyro_timestamps, levels=6, full_output=False):
    """Get time offset that aligns image timestamps with gyro timestamps.
    
    Given an image sequence, and gyroscope data, with their respective timestamps,
    calculate the offset that aligns the image data with the gyro data.
    The timestamps must only differ by an offset, not a scale factor.

    This function finds an approximation of the offset *d* that makes this transformation

        t_gyro = t_camera + d
        
    i.e. your new image timestamps should be
    
        image_timestamps_aligned = image_timestamps + d
        
    The offset is calculated using zero-mean cross correlation of the gyroscope data magnitude
    and the optical flow magnitude, calculated from the image sequence.
    ZNCC is performed using pyramids to make it quick.
    
    The offset is accurate up to about +/- 2 frames, so you should run
    *refine_time_offset* if you need better accuracy.
    
    Parameters
    ---------------
    image_sequence : sequence of image data
            This must be either a list or generator that provides a stream of 
            images that are used for optical flow calculations.
    image_timestamps : ndarray
            Timestamps of the images in image_sequence
    gyro_data : (3, N) ndarray
            Gyroscope measurements (angular velocity)
    gyro_timestamps : ndarray
            Timestamps of data in gyro_data
    levels : int
            Number of pyramid levels
    full_output : bool
            If False, only return the offset, otherwise return extra data
            
    Returns
    --------------
    time_offset  :  float
            The time offset to add to image_timestamps to align the image data
            with the gyroscope data
    flow : ndarray
            (Only if full_output=True)
            The calculated optical flow magnitude
    """
    
    # Get rotation magnitude from gyro data
    gyro_mag = np.sum(gyro_data**2, axis=0)
    flow_org = tracking.optical_flow_magnitude(image_sequence)
    flow_timestamps = image_timestamps[:-2]

    # Resample to match highest
    rate = lambda ts: len(ts) / float(ts[-1] - ts[0])
    freq_gyro = rate(gyro_timestamps)
    freq_image = rate(flow_timestamps)
    
    if freq_gyro > freq_image:
        rel_rate = freq_gyro / freq_image
        flow_mag = znccpyr.upsample(flow_org, rel_rate)
    else:
        rel_rate = freq_image / freq_gyro
        gyro_mag = znccpyr.upsample(gyro_mag, rel_rate)
    
    ishift = znccpyr.find_shift_pyr(flow_mag, gyro_mag, levels)
    
    if freq_gyro > freq_image:
        flow_shift = int(-ishift / rel_rate)
    else:
        flow_shift = int(-ishift)
    
    time_offset = flow_timestamps[flow_shift]
    
    if full_output:
        return time_offset, flow_org # Return the orginal flow, not the upsampled version
    else:
        return time_offset

#--------------------------------------------------------------------------

def sync_camera_gyro_manual(image_sequence, image_timestamps, gyro_data, gyro_timestamps, full_output=False):
    """Get time offset that aligns image timestamps with gyro timestamps.
    
    Given an image sequence, and gyroscope data, with their respective timestamps,
    calculate the offset that aligns the image data with the gyro data.
    The timestamps must only differ by an offset, not a scale factor.

    This function finds an approximation of the offset *d* that makes this transformation

        t_gyro = t_camera + d
        
    i.e. your new image timestamps should be
    
        image_timestamps_aligned = image_timestamps + d
        
    The offset is calculated using correlation. The parts of the signals to use are
    chosen by the user by picking points in a plot window.

    The offset is accurate up to about +/- 2 frames, so you should run
    *refine_time_offset* if you need better accuracy.
    
    Parameters
    ---------------
    image_sequence : sequence of image data
            This must be either a list or generator that provides a stream of 
            images that are used for optical flow calculations.
    image_timestamps : ndarray
            Timestamps of the images in image_sequence
    gyro_data : (3, N) ndarray
            Gyroscope measurements (angular velocity)
    gyro_timestamps : ndarray
            Timestamps of data in gyro_data
    full_output : bool
            If False, only return the offset, otherwise return extra data
            
    Returns
    --------------
    time_offset  :  float
            The time offset to add to image_timestamps to align the image data
            with the gyroscope data
    flow : ndarray
            (Only if full_output=True)
            The calculated optical flow magnitude
    frame_pair : (int, int)
            The frame pair that was picked for synchronization
    """
    
    flow = tracking.optical_flow_magnitude(image_sequence)
    flow_timestamps = image_timestamps[:-2]    
    
    # Let user select points in both pieces of data
    (frame_pair, gyro_idx) = manual_sync_pick(flow, gyro_timestamps, gyro_data)
    
    # Normalize data
    gyro_abs_max = np.max(np.abs(gyro_data), axis=0)
    gyro_normalized = (gyro_abs_max / np.max(gyro_abs_max)).flatten()
    flow_normalized = (flow / np.max(flow)).flatten()

    rate = lambda ts: len(ts) / float(ts[-1] - ts[0])

    # Resample to match highest
    freq_gyro = rate(gyro_timestamps)
    freq_image = rate(flow_timestamps)
    logger.debug("Gyro sampling frequency: %.2f Hz, Image sampling frequency: %.2f Hz", freq_gyro, freq_image)
    
    gyro_part = gyro_normalized[gyro_idx[0]:gyro_idx[1]+1] # only largest
    flow_part = flow_normalized[frame_pair[0]:frame_pair[1]+1]
    
    N = flow_part.size * freq_gyro / freq_image
    flow_part_resampled = ssig.resample(flow_part, N).flatten()
    
    # ) Cross correlate the two signals and find time diff
    corr = ssig.correlate(gyro_part, flow_part_resampled, 'full') # Find the flow in gyro data
 
    i = np.argmax(corr)
    
    t_0_f = flow_timestamps[frame_pair[0]]
    t_1_f = flow_timestamps[frame_pair[1]]
    
    t_off_g = gyro_timestamps[gyro_idx[0] + i]
    t_off_f = t_1_f
    time_offset = t_off_g - t_off_f
    
    if full_output:
        return time_offset, flow, frame_pair
    else:
        return time_offset

#--------------------------------------------------------------------------

def manual_sync_pick(flow, gyro_ts, gyro_data):
    # First pick good points in flow
    plt.clf()
    plt.plot(flow)
    plt.title('Select two points')
    selected_frames = [int(round(x[0])) for x in plt.ginput(2)]

    # Now pick good points in gyro
    plt.clf()
    plt.subplot(211)
    plt.plot(flow)
    plt.plot(selected_frames, flow[selected_frames], 'ro')
    
    plt.subplot(212)
    plt.plot(gyro_ts, gyro_data.T)
    plt.title('Select corresponding sequence in gyro data')
    plt.draw()
    selected = plt.ginput(2) #[int(round(x[0])) for x in plt.ginput(2)]
    gyro_idxs = [(gyro_ts >= x[0]).nonzero()[0][0] for x in selected]
    plt.plot(gyro_ts[gyro_idxs], gyro_data[:, gyro_idxs].T, 'ro')
    plt.title('Ok, click to continue to next')
    plt.draw()
    plt.waitforbuttonpress(timeout=10.0)
    plt.close()
    
    return (tuple(selected_frames), gyro_idxs)

#--------------------------------------------------------------------------

def refine_time_offset(image_list, frame_timestamps, rotation_sequence, rotation_timestamps, camera_matrix, readout_time):
    """Refine a time offset between camera and IMU using rolling shutter aware optimization.
    
    To refine the time offset using this function, you must meet the following constraints
    
    1) The data must already be roughly aligned. Only a few image frames of error
        is allowed.
    2) The images *must* have been captured by a *rolling shutter* camera.
    
    This function finds a refined offset using optimization.
    Points are first tracked from the start to the end of the provided images.
    Then an optimization function looks at the reprojection error of the tracked points
    given the IMU-data and the refined offset.
    
    The found offset *d* is such that you want to perform the following time update
    
        new_frame_timestamps = frame_timestamps + d
    
    Parameters
    ------------
    image_list : list of ndarray
            A list of images to perform tracking on. High quality tracks are required,
            so make sure the sequence you choose is easy to track in.
    frame_timestamps : ndarray
            Timestamps of image_list
    rotation_sequence : (4, N) ndarray
            Absolute rotations as a sequence of unit quaternions (first element is scalar).
    rotation_timestamps : ndarray
            Timestamps of rotation_sequence
    camera_matrix : (3,3) ndarray
            The internal camera calibration matrix of the camera.
    readout_time : float
            The readout time of the camera.
            
    Returns
    ------------
    offset : float
            A refined offset that aligns the image data with the rotation data.
    """
    # ) Track points
    max_corners = 200
    quality_level = 0.07
    min_distance = 5
    max_tracks = 20
    initial_points = cv2.goodFeaturesToTrack(image_list[0], max_corners, quality_level, min_distance)
    (points, status) = tracking.track_retrack(image_list, initial_points)

    # Prune to at most max_tracks number of tracks, choose randomly    
    track_id_list = np.random.permutation(points.shape[0])[:max_tracks]
        
    rows, cols = image_list[0].shape[:2]
    row_delta_time = readout_time / rows            
    num_tracks, num_frames, _ = points.shape
    K = np.matrix(camera_matrix)
    
    def func_to_optimize(td, *args):
        res = 0.0
        N = 0
        for frame_idx in range(num_frames-1):            
            for track_id in track_id_list:                
                p1 = points[track_id, frame_idx, :].reshape((-1,1))
                p2 = points[track_id, frame_idx + 1, :].reshape((-1,1))
                t1 = frame_timestamps[frame_idx] + (p1[1] - 1) * row_delta_time + td
                t2 = frame_timestamps[frame_idx + 1] + (p2[1] - 1) * row_delta_time +td
                t1 = float(t1)
                t2 = float(t2)
                q1 = IMU.rotation_at_time(t1, rotation_timestamps, rotation_sequence)
                q2 = IMU.rotation_at_time(t2, rotation_timestamps, rotation_sequence)
                R1 = rotations.quat_to_rotation_matrix(q1)
                R2 = rotations.quat_to_rotation_matrix(q2)
                p1_rec = K.dot(R1.T).dot(R2).dot(K.I).dot(np.vstack((p2, 1)))
                if p1_rec[2] == 0:
                    continue
                else:
                    p1_rec /= p1_rec[2]                    
                res += np.sum((p1 - np.array(p1_rec[0:2]))**2)
                N += 1
        return res / N
    
    # Bounded Brent optimizer
    t0 = time.time()
    tolerance = 1e-4 # one tenth millisecond
    (refined_offset, fval, ierr, numfunc) = scipy.optimize.fminbound(func_to_optimize, -0.12, 0.12, xtol=tolerance, full_output=True)
    t1 = time.time()
    if ierr == 0:
        logger.info("Time offset found by brent optimizer: %.4f. Elapsed: %.2f seconds (%d function calls)", refined_offset, t1-t0, numfunc)
    else:
        logger.error("Brent optimizer did not converge. Aborting!")
        raise Exception("Brent optimizer did not converge, when trying to refine offset.")
    
    return refined_offset
    
    
def good_sequences_to_track(flow, motion_threshold=1.0):
    """Get list of good frames to do tracking in.

    Looking at the optical flow, this function chooses a span of frames
    that fulfill certain criteria.
    These include
        * not being too short or too long
        * not too low or too high mean flow magnitude
        * a low max value (avoids motion blur)
    Currently, the cost function for a sequence is hard coded. Sorry about that.
    
    Parameters
    -------------
    flow : ndarray
            The optical flow magnitude
    motion_threshold : float
            The maximum amount of motion to consider for sequence endpoints.
            
    Returns
    ------------
    sequences : list
            Sorted list of (a, b, score) elements (highest scpre first) of sequences
            where a sequence is frames with frame indices in the span [a, b].
    """
    endpoints = []
    in_low = False
    for i, val in enumerate(flow):
        if val < motion_threshold:
            if not in_low:
                endpoints.append(i)
                in_low = True
        else:
            if in_low:
                endpoints.append(i-1) # Previous was last in a low spot
            in_low = False
    
    def mean_score_func(m):
        mu = 15
        sigma = 8
        top_val = normpdf(mu, mu, sigma)
        return normpdf(m, mu, sigma) / top_val
    
    def max_score_func(m):
        mu = 40
        sigma = 8
        if m <= mu:
            return 1.
        else:
            top_val = normpdf(mu, mu, sigma)
            return normpdf(m, mu, sigma) / top_val
    
    def length_score_func(l):
        mu = 30
        sigma = 10
        top_val = normpdf(mu, mu, sigma)
        return normpdf(l, mu, sigma) / top_val
    
    min_length = 5 # frames
    sequences = []
    for k, i in enumerate(endpoints[:-1]):
        for j in endpoints[k+1:]:
            length = j - i
            if length < min_length:
                continue
            seq = flow[i:j+1]
            m_score = mean_score_func(np.mean(seq))
            mx_score = max_score_func(np.max(seq))
            l_score = length_score_func(length)
            logger.debug("%d, %d scores: (mean=%.5f, max=%.5f, length=%.5f)" % (i,j,m_score, mx_score, l_score))
            if min(m_score, mx_score, l_score) < 0.2:
                continue
            
            score = m_score + mx_score + l_score 
            sequences.append((i, j, score))

    return sorted(sequences, key=lambda x: x[2], reverse=True)