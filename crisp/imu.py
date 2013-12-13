# -*- coding: utf-8 -*-
"""
IMU module
"""

__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2013, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

#--------------------------------------------------------------------------
# Includes
#--------------------------------------------------------------------------

import numpy as np
import re

import scipy.io

from . import rotations
 
#--------------------------------------------------------------------------
# Classes
#--------------------------------------------------------------------------

class IMU(object):
    """
    Defines an IMU (currently only gyro)
    """
    def __init__(self):
        self.integrated = []
        self.gyro_data = []
        self.timestamps = []
    
    @classmethod
    def from_mat_file(cls, matfilename):
        """Load gyro data from .mat file
        
        The MAT file should contain the following two arrays
        
        gyro : (3, N) float ndarray
                The angular velocity measurements.
        timestamps : (N, ) float ndarray
                Timestamps of the measurements.
                
        Parameters
        ---------------
        matfilename : string
                Name of the .mat file
        
        Returns
        ----------------
        A new IMU class instance
        """
        M = scipy.io.loadmat(matfilename)
        instance = cls()
        instance.gyro_data = M['gyro']
        instance.timestamps = M['timestamps']
        return instance
        
    
    @property    
    def rate(self):
        """Get the sample rate in Hz.
        
        Returns
        ---------
        rate : float
                The sample rate, in Hz, calculated from the timestamps        
        """
        N = len(self.timestamps)
        t = self.timestamps[-1] - self.timestamps[0]
        rate = 1.0 * N / t
        return rate

    def zero_level_calibrate(self, duration, t0=0.0):
        """Performs zero-level calibration from the chosen time interval.
        
        This changes the previously lodaded data in-place.
        
        Parameters
        --------------------
        duration : float
                Number of timeunits to use for calibration
        t0 : float 
                Starting time for calibration
                
        Returns
        ----------------------
        gyro_data : (3, N) float ndarray
                The calibrated data (note that it is also changed in-place!)
        """
        
        t1 = t0 + duration
        indices = np.flatnonzero((self.timestamps >= t0) & (self.timestamps <= t1))
        m = np.mean(self.gyro_data[:, indices], axis=1)
        self.gyro_data -= m.reshape(3,1)
        
        return self.gyro_data
        
    def gyro_data_corrected(self, pose_correction=np.eye(3)):
        """Get relative pose corrected data.
        
        Parameters
        -------------
        pose_correction : (3,3) ndarray, optional
                Rotation matrix that describes the relative pose between the IMU and something else (e.g. camera).
        
        Returns
        ---------------
        gyro_data : (3, N) ndarray
                The relative pose corrected data.
        """
        return pose_correction.dot(self.gyro_data)
            
    def integrate(self, pose_correction=np.eye(3)):
        """Integrate angular velocity measurements to rotations.

        Parameters
        -------------
        pose_correction : (3,3) ndarray, optional
                Rotation matrix that describes the relative pose between the IMU and something else (e.g. camera).
        
        Returns
        -------------
        rotations : (4, N) ndarray
                Rotations as unit quaternions with scalar as first element.
        """
        N = len(self.timestamps)
        integrated = np.zeros((4, N))
        integrated[:,0] = np.array([1, 0, 0, 0]) # Initial rotation (no rotation)
        
        # Iterate over all
        for i in xrange(1, len(self.timestamps)):
            w = pose_correction.dot(self.gyro_data[:, i]) # Change to correct coordinate frame
            dt = float(self.timestamps[i] - self.timestamps[i - 1])
            qprev = integrated[:, i - 1].flatten()
            
            A = np.array([[0,    -w[0],  -w[1],  -w[2]],
                         [w[0],  0,      w[2],  -w[1]],
                         [w[1], -w[2],   0,      w[0]],
                         [w[2],  w[1],  -w[0],   0]])
            qnew = (np.eye(4) + (dt/2.0) * A).dot(qprev)
            qnorm = np.sqrt(np.sum(qnew ** 2))
            qnew = qnew / qnorm if qnorm > 0 else 0
            integrated[:, i] = qnew
            #print "%d, %s, %s, %s, %s" % (i, w, dt, qprev, qnew)
        return integrated
        
    @staticmethod
    def rotation_at_time(t, timestamps, rotation_sequence):
        """Get the gyro rotation at time t using SLERP.
        
        Parameters
        -----------
        t : float
                The query timestamp.
        timestamps : array_like float
                List of all timestamps
        rotation_sequence : (4, N) ndarray
                Rotation sequence as unit quaternions with scalar part as first element.
                
        Returns
        -----------
        q : (4,) ndarray
                Unit quaternion representing the rotation at time t.
        """
        idx = np.flatnonzero(timestamps >= (t - 0.0001))[0]
        t0 = timestamps[idx - 1]
        t1 = timestamps[idx]
        tau = (t - t0) / (t1 - t0)
        
        q1 = rotation_sequence[:, idx - 1]
        q2 = rotation_sequence[:, idx]
        q = rotations.slerp(q1, q2, tau)
        return q

class ArduIMU(IMU):
    def __init__(self, filename):
        super(ArduIMU, self).__init__()
        self.filename = filename
        ts, acc, gyro = self.__load(filename)
        self.timestamps = ts
        self.gyro_data = gyro
    
    def __load(self, gyro_data_filename):
        f = open(gyro_data_filename, 'r')
        # Read header
        if not f.readline().strip().startswith('RPY'):
            raise ValueError("This is not a ArduIMU log file")
            
        data = np.array([[]], dtype='float32')
        data.shape = 0,7
        for line in f.readlines():
            row = [int(s) for s in re.findall('\d+', line)]
            if len(row) == 8:
                if row[7] != 1:
                    pass # "Checksum error, skipping"
                else:
                    data = np.append(data, [row[0:7]], axis=0)

        timestamps = data[:,0]
        timestamps -= timestamps[0] # Start at 0
        timestamps /= 1000.0 # Milliseconds -> seconds
            
        # Map from 10-bit value to voltage
        # The arduino has mapped the range [0,Vref] to [0,1023] before removing the offset
        Vref = 3.3    
        data[:,1:] *= (Vref / 1023.0);

        accelerometer = data[:,1:4]
        gyroscope = data[:,4:7]
        
        # Scale gyro output
        gyro_scale = 3.33 / 1000 # (V/(degrees/s)) From datasheet (not ratiometric)
        gyroscope = np.deg2rad(gyroscope / gyro_scale) #  rad / s
        
        # Scale accelerometer
        gravity = 9.81
        gravity_scale = 0.330 # V/g
        accelerometer /= gravity_scale; # Scale to acceleration in g's
        accelerometer *= gravity # Scale to acceleration in m/s2

        return (timestamps, accelerometer.T, gyroscope.T)
