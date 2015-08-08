# -*- coding: utf-8 -*-
"""
Camera-gyro calibration module
"""
__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2015, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

import time
import warnings
import logging
logger = logging.getLogger('crisp')

import numpy as np
import scipy.optimize

from . import videoslice, rotations, ransac, timesync, fastintegrate

PARAM_SOURCE_ORDER = ('user', 'initialized', 'calibrated') # Increasing order of importance
PARAM_ORDER = ('gyro_rate', 'time_offset', 'gbias_x', 'gbias_y', 'gbias_z', 'rot_x', 'rot_y', 'rot_z')

MAX_OPTIMIZATION_TRACKS = 1500
MAX_OPTIMIZATION_FEV = 800

class CalibrationError(Exception):
    pass

class InitializationError(Exception):
    pass

class AutoCalibrator(object):
    """Class that handles auto calibration of a camera-gyroscope system.

    This calibrator uses the method described in [1]_.

    The parameters which are calibrated for are

        * Gyroscope sample rate
        * Time offset
        * Gyroscope bias
        * Rotation between camera and gyroscope

    Notes
    ---------------------
    Given time offset d, and gyro rate F, the time relation is such that we
    calculate the corresponding gyroscope sample n from video time t as

        n = F ( t + d )

    The rotation between camera and gyroscope, R, is expressed such that it transfers points from the gyroscope
    coordinate frame to the camera coordinate frame as

        p_camera = R * p_gyro

    The bias is applied to the gyroscope measurements, w, before integration

        w_adjusted = w - bias

    References
    ----------------------
    ..  [1] Ovrén, H and Forssén, P.-E. "Gyroscope-based video stabilisation with auto-calibration."
        In 2015 IEEE International Conference on Robotics and Automation (ICRA) (pp. 2090–2097). Seattle, WA
    """
    def __init__(self, video, gyro):
        """Create calibrator

        Parameters
        ---------------
        video : VideoStream
            A video stream object that provides frames and camera information
        gyro : GyroStream
            A gyroscope stream that provides angular velocity measurements
        """
        self.video = video
        self.gyro = gyro
        
        self.slices = None

        # Parameters can be supplied from different sources, and it can be useful to track that
        self.params = {
            'user' : {}, # Supplied by the user
            'initialized' : {}, # Estimated automatically by running initialize()
            'calibrated' : {} # Final calibrated values
        }
    
    def initialize(self, gyro_rate, slices=None):
        """Prepare calibrator for calibration

        This method does three things:
        1. Create slices from the video stream, if not already provided
        2. Estimate time offset
        3. Estimate rotation between camera and gyroscope

        Parameters
        ------------------
        gyro_rate : float
            Estimated gyroscope sample rate
        slices : list of Slice, optional
            Slices to use for optimization

        Raises
        --------------------
        InitializationError
            If the initialization fails
        """
        self.params['user']['gyro_rate'] = gyro_rate

        for p in ('gbias_x', 'gbias_y', 'gbias_z'):
            self.params['initialized'][p] = 0.0

        if slices is not None:
            self.slices = slices

        if self.slices is None:
            self.slices = videoslice.Slice.from_stream_randomly(self.video)
            logger.debug("Number of slices: {:d}".format(len(self.slices)))

        time_offset = self.find_initial_offset()
        # TODO: Detect when time offset initialization fails, and raise InitializationError

        logger.debug("Initial time offset: {:.4f}".format(time_offset))
        self.params['initialized']['time_offset'] = time_offset

        R = self.find_initial_rotation()
        if R is None:
            raise InitializationError("Failed to calculate initial rotation")

        n, theta = rotations.rotation_matrix_to_axis_angle(R)
        logger.debug("Found rotation: n={} theta={};  r={}".format(n, theta, n*theta))
        logger.debug(R)
        rx, ry, rz = theta * n
        self.params['initialized']['rot_x'] = rx
        self.params['initialized']['rot_y'] = ry
        self.params['initialized']['rot_z'] = rz
        
    def video_time_to_gyro_sample(self, t):
        """Convert video time to gyroscope sample index and interpolation factor

        Parameters
        -------------------
        t : float
            Video timestamp

        Returns
        --------------------
        n : int
            Sample index that precedes t
        tau : float
            Interpolation factor [0.0-1.0]. If tau=0, then t falls on exactly n. If tau=1 then t falls exactly on n+1
        """
        f_g = self.parameter['gyro_rate']
        d_c = self.parameter['time_offset']
        n = f_g * (t + d_c)
        n0 = int(np.floor(n))
        tau = n - n0
        return n0, tau
    
    @property
    def parameter(self):
        """Return the current best value of a parameter"""
        D = {}
        for source in PARAM_SOURCE_ORDER:
            D.update(self.params[source])
        return D              
        
    def calibrate(self, max_tracks=MAX_OPTIMIZATION_TRACKS, max_eval=MAX_OPTIMIZATION_FEV):
        """Perform calibration

        Parameters
        ----------------------
        max_eval : int
            Maximum number of function evaluations

        Returns
        ---------------------
        dict
            Optimization result

        Raises
        -----------------------
        CalibrationError
            If calibration fails
        """
        x0 = np.array([self.parameter[param] for param in PARAM_ORDER])
        available_tracks = np.sum([len(s.inliers) for s in self.slices])
        if available_tracks < max_tracks:
            warnings.warn("Could not use the requested {} tracks, since only {} were available in the slice data.".format(max_tracks, available_tracks))
            max_tracks = available_tracks

        # Get subset of available tracks such that all slices are still used
        slice_sample_idxs = videoslice.fill_sampling(self.slices, max_tracks)

        func_args = (self.slices, slice_sample_idxs, self.video.camera_model, self.gyro)
        self.slice_sample_idxs = slice_sample_idxs
        logger.debug("Starting optimization on {:d} slices and {:d} tracks".format(len(self.slices), max_tracks))
        start_time = time.time()
        # TODO: Check what values of ftol and xtol are required for good results. The current setting is probably pessimistic.
        leastsq_result = scipy.optimize.leastsq(optimization_func, x0, args=func_args, full_output=True, ftol=1e-10, xtol=1e-10, maxfev=max_eval)
        elapsed = time.time() - start_time
        x, covx, infodict, mesg, ier = leastsq_result
        self.__debug_leastsq = leastsq_result
        logger.debug("Optimization completed in {:.1f} seconds and {:d} function evaluations. ier={}, mesg='{}'".format(elapsed, infodict['nfev'], ier, mesg))
        if ier in (1,2,3,4):
            for pname, val in zip(PARAM_ORDER, x):
                self.params['calibrated'][pname] = val
            return self.parameter
        else:
            raise CalibrationError(mesg)

    def find_initial_offset(self):
        """Estimate time offset

        Returns
        ---------------
        float
            Estimated time offset
        """
        flow = self.video.flow
        gyro_rate = self.parameter['gyro_rate']
        frame_times = np.arange(len(flow)) / self.video.frame_rate
        gyro_times = np.arange(self.gyro.num_samples) / gyro_rate
        time_offset = timesync.sync_camera_gyro(flow, frame_times, self.gyro.data.T, gyro_times, levels=6)
        return time_offset

    def find_initial_rotation(self):
        """Estimate rotation between camera and gyroscope

        Returns
        --------------------
        (3,3) ndarray
            Estimated rotation between camera and gyroscope
        """
        dt = 1.0 / self.parameter['gyro_rate']
        q = self.gyro.integrate(dt)
        
        video_axes = []
        gyro_axes = []
        
        for _slice in self.slices:
            # Estimate rotation here
            _slice.estimate_rotation(self.video.camera_model, ransac_threshold=7.0) # sets .axis and .angle memebers
            if _slice.axis is None:
                continue
            assert _slice.angle > 0
            
            t1 = _slice.start / float(self.video.frame_rate)
            n1, _ = self.video_time_to_gyro_sample(t1)
            t2 = _slice.end / float(self.video.frame_rate)
            n2, _ = self.video_time_to_gyro_sample(t2)
            
            try:
                qx = q[n1]
                qy = q[n2]
            except IndexError:
                continue # No gyro data -> nothing to do with this slice
                
            Rx = rotations.quat_to_rotation_matrix(qx)
            Ry = rotations.quat_to_rotation_matrix(qy)
            R = np.dot(Rx.T, Ry)
            v, theta = rotations.rotation_matrix_to_axis_angle(R)
            if theta < 0:
                v = -v
                
            gyro_axes.append(v)
            video_axes.append(_slice.axis)    

        logger.debug("Using {:d} slices (from initial {:d} for rotation estimation".format(len(gyro_axes), len(self.slices)))

        model_func = lambda data: rotations.procrustes(data[:3], data[3:6], remove_mean=False)[0]
        
        def eval_func(model, data):
            X = data[:3].reshape(3,-1)
            Y = data[3:6].reshape(3,-1)
            R = model
            Xhat = np.dot(R, Y)
            
            costheta = np.sum(Xhat*X, axis=0)
            theta = np.arccos(costheta)
            
            return theta
       
        inlier_selection_prob = 0.99999
        model_points = 2 # Set to 3 to use non-minimal case
        inlier_ratio = 0.5
        threshold = np.deg2rad(10.0)
        ransac_iterations = int(np.log(1 - inlier_selection_prob) / np.log(1-inlier_ratio**model_points))
        data = np.vstack((np.array(video_axes).T, np.array(gyro_axes).T))    
        assert data.shape == (6, len(gyro_axes))
        
        R, ransac_conseus_idx = ransac.RANSAC(model_func, eval_func, data,
                                              model_points, ransac_iterations,
                                              threshold, recalculate=True)

        return R

    def print_params(self):
        """Print the current best set of parameters"""
        print "Parameters"
        print "--------------------"
        for param in PARAM_ORDER:
            print '  {:>11s} = {}'.format(param, self.parameter[param])

def sample_at_time(t, rate):
    s = t * rate
    n = np.ceil(s)
    tau = s - n
    return n, tau

def robust_norm(r, c):
    return r / (1 + (np.abs(r)/c))


def optimization_func(x, slices, slice_sample_idxs, camera, gyro):
    # Unpack parameters and convert representations
    Fg, offset, gbias_x, gbias_y, gbias_z, rot_x, rot_y, rot_z, = x

    gyro_bias = np.array([gbias_x, gbias_y, gbias_z])

    # Construct coordinate frame rotation matrix
    v = np.array([rot_x, rot_y, rot_z])
    theta = np.linalg.norm(v)
    v /= theta
    R_g2c = rotations.axis_angle_to_rotation_matrix(v, theta)

    Tg = float(1. / Fg)
    row_delta = camera.readout / camera.rows

    errors = [] # Residual vector

    # Margin of integration is amount of gyro samples per frame
    integration_margin = int(np.ceil(Fg * camera.readout))

    for _slice, sample_idxs in zip(slices, slice_sample_idxs):
        if len(sample_idxs) < 1:
            continue

        t_start = _slice.start / camera.frame_rate + offset
        t_end = _slice.end / camera.frame_rate + offset
        slice_start, _ = sample_at_time(t_start, Fg)
        slice_end, _ = sample_at_time(t_end, Fg)
        slice_end += 1 # sample_at_time() gives first sample

        # Gyro samples to integrate within
        integration_start = slice_start
        integration_end = slice_end + integration_margin

        # Depending on the data and current set of parameters, it is quite possible that
        # we might try to integrate outside of available data.
        # To not vary the number of residuals between iterations, add zero residuals for these.
        if integration_start < 0 or integration_end >= gyro.num_samples or (integration_end - integration_start) < 1:
            logging.debug("Integration range [{:d},{:d}] is outside gyro data range [{:d},{:d}]".format(integration_start, integration_end, 0, gyro.num_samples-1))
            num_zero_residuals = 2 * 2 * len(sample_idxs) # 2 dim per point, 2 because of symmetric cost
            errors.extend(np.zeros((num_zero_residuals, )))
            continue

        # Integrate
        gyro_part = gyro.data[integration_start:integration_end+1]
        gyro_part_corrected = gyro_part + gyro_bias
        q = fastintegrate.integrate_gyro_quaternion_uniform(gyro_part_corrected, Tg)

        for track in _slice.points[sample_idxs]:
            x = track[0] # Points in first frame
            y = track[-1] # Points in last frame

            # Get row time
            tx = t_start + x[1] * row_delta
            ty = t_end + y[1] * row_delta

            # Sample index and interpolation value for point correspondences
            nx, taux = sample_at_time(tx, Fg)
            ny, tauy = sample_at_time(ty, Fg)

            # Interpolate rotation using SLERP
            a = nx - integration_start
            b = ny - integration_start
            qx = rotations.slerp(q[a], q[a+1], taux)
            qy = rotations.slerp(q[b], q[b+1], tauy)

            Rx = rotations.quat_to_rotation_matrix(qx)
            Ry = rotations.quat_to_rotation_matrix(qy)
            R1 = np.dot(Rx.T, Ry) # Note: Transpose order is "wrong", but this is because definition of Rx

            R = R_g2c.dot(R1).dot(R_g2c.T)

            Y = camera.unproject(y)
            Xhat = np.dot(R, Y)
            xhat = camera.project(Xhat)

            err = x - xhat
            errors.extend(err.flatten())

            # Symmetric errors, so let's do this again
            R1 = np.dot(Ry.T, Rx) # Note: Transpose order is "wrong", but this is because definition of Rx
            R = R_g2c.dot(R1).dot(R_g2c.T)

            X = camera.unproject(x)
            Yhat = np.dot(R, X)
            yhat = camera.project(Yhat)

            err = y - yhat
            errors.extend(err.flatten())

    if not errors:
        raise ValueError("No residuals!")

    # Apply robust norm
    c = 3.0
    robust_errors = robust_norm(np.array(errors), c)

    return robust_errors