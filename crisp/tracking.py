# -*- coding: utf-8 -*-
"""
Tracking module
"""

__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2013, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

#--------------------------------------------------------------------------
# Includes
#--------------------------------------------------------------------------

import cv2
import numpy as np

#--------------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------------

GFTT_DEFAULTS = {'max_corners' : 40,
                'quality_level' : 0.07,
                'min_distance' : 10}    

#--------------------------------------------------------------------------
# Public functions
#--------------------------------------------------------------------------

def track_points(img1, img2, initial_points=None, gftt_params={}):
    """Track points between two images
    
    Parameters
    -----------------
    img1 : (M, N) ndarray
            First image
    img2 : (M, N) ndarray
            Second image
    initial_points : ndarray
            Initial points. If empty, initial points will be calculated from
            img1 using goodFeaturesToTrack in OpenCV
    gftt_params : dict
            Keyword arguments for goodFeaturesToTrack
    
    Returns
    -----------------
    points : ndarray
            Tracked points
    initial_points : ndarray
            Initial points used
    """
    params = GFTT_DEFAULTS
    if gftt_params:
        params.update(gftt_params)

    if initial_points is None:
        initial_points = cv2.goodFeaturesToTrack(img1, params['max_corners'], params['quality_level'], params['min_distance'])
    
    [_points, status, err] = cv2.calcOpticalFlowPyrLK(img1, img2, initial_points, np.array([]))

    # Filter out valid points only
    points = _points[np.nonzero(status)]
    initial_points = initial_points[np.nonzero(status)]

    return (points, initial_points)

#--------------------------------------------------------------------------

def optical_flow_magnitude(image_sequence, max_diff=60, gftt_options={}):
    """Return optical flow magnitude for the given image sequence
    
    The flow magnitude is the mean value of the total (sparse) optical flow
    between two images.
    Crude outlier detection using the max_diff parameter is used.
    
    Parameters
    ----------------
    image_sequence : sequence
            Sequence of image data (ndarrays) to calculate flow magnitude from
    max_diff : float
            Distance threshold for outlier rejection
    gftt_options : dict
            Keyword arguments to the OpenCV goodFeaturesToTrack function
            
    Returns
    ----------------
    flow : ndarray
            The optical flow magnitude
    """
    flow = []
    prev_img = None
    for img in image_sequence:
        if prev_img is None:
            prev_img = img
            continue
        (next_points, prev_points) = track_points(prev_img, img, gftt_params=gftt_options)
        distance = np.sqrt(np.sum((next_points - prev_points)**2, 1))
        distance2 = distance[np.nonzero(distance < max_diff)] # Crude outlier rejection
        dm = np.mean(distance2)
        if np.isnan(dm):
            dm = 0
        flow.append(dm)
        prev_img = img

    return np.array(flow)

#--------------------------------------------------------------------------

def track(image_list, initial_points, remove_bad=True):
    """Track points in image list
    
    Parameters
    ----------------
    image_list : list
            List of images to track in
    initial_points : ndarray
            Initial points to use (in first image in image_list)
    remove_bad : bool
            If True, then the resulting list of tracks will only contain succesfully
            tracked points. Else, it will contain all points present in initial_points.
    
    Returns
    -----------------
    tracks : (N, M, 2) ndarray
            N tracks over M images with (x,y) coordinates of points
    status : (N,) ndarray
            The status of each track. 1 means ok, while 0 means tracking failure
    """
    # Precreate track array
    tracks = np.zeros((initial_points.shape[0], len(image_list), 2), dtype='float32') # NxMx2
    tracks[:,0,:] = np.reshape(np.array(initial_points), [-1,2])
    track_status = np.ones([np.size(initial_points,0),1]) # All initial points are OK
    empty = np.array([])
    window_size = (5,5)
    for i in range(1, len(image_list)):
        img1 = image_list[i-1]
        img2 = image_list[i]
        prev_ok_track = np.flatnonzero(track_status)
        prev_points = tracks[prev_ok_track,i-1,:]
        [points, status, err] = cv2.calcOpticalFlowPyrLK(img1, img2, prev_points, empty, empty, empty, window_size)
        valid_set = np.flatnonzero(status)
        now_ok_tracks = prev_ok_track[valid_set] # Remap
        tracks[now_ok_tracks,i,:] = points[valid_set]
        track_status[prev_ok_track] = status

    if remove_bad:
        final_ok = np.flatnonzero(track_status)
        tracks = tracks[final_ok] # Only rows/tracks with nonzero status
        track_status = track_status[final_ok] 

    return (tracks, track_status)

#--------------------------------------------------------------------------

def track_retrack(image_list, initial_points, max_retrack_distance=0.5, keep_bad=False):
    """Track-retracks points in image list
    
    Using track-retrack can help in only getting point tracks of high quality.    
    
    The point is tracked forward, and then backwards in the image sequence.
    Points that end up further than max_retrack_distance from its starting point
    are marked as bad.
    
    Parameters
    ----------------
    image_list : list
            List of images to track in
    initial_points : ndarray
            Initial points to use (in first image in image_list)
    max_retrack_distance : float
            The maximum distance of the retracked point from its starting point to 
            still count as a succesful retrack.
    remove_bad : bool
            If True, then the resulting list of tracks will only contain succesfully
            tracked points. Else, it will contain all points present in initial_points.
    
    Returns
    -----------------
    tracks : (N, M, 2) ndarray
            N tracks over M images with (x,y) coordinates of points
            Note that M is the number of image in the input, and is the track in
            the forward tracking step.
    status : (N,) ndarray
            The status of each track. 1 means ok, while 0 means tracking failure
    """
    (forward_track, forward_status) = track(image_list, initial_points, remove_bad=False)
    # Reverse the order
    (backward_track, backward_status) = track(image_list[::-1], forward_track[:,-1,:], remove_bad=False)

    # Prune bad tracks
    ok_track = np.flatnonzero(forward_status * backward_status) # Only good if good in both
    forward_first = forward_track[ok_track,0,:]
    backward_last = backward_track[ok_track,-1,:]

    # Distance
    retrack_distance = np.sqrt(np.sum((forward_first - backward_last)**2, 1))

    # Allowed
    retracked_ok = np.flatnonzero(retrack_distance <= max_retrack_distance)
    final_ok = np.intersect1d(ok_track, retracked_ok)

    if keep_bad: # Let caller check status
        status = np.zeros(forward_status.shape)
        status[final_ok] = 1
        return (forward_track, status)
    else: # Remove tracks with faulty retrack
        return (forward_track[final_ok], forward_status[final_ok])
