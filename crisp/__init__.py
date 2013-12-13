# -*- coding: utf-8 -*-
"""
========================================
Camera-to-IMU calibration toolbox
========================================
This package solves two tasks: finding the relative pose between a camera and IMU
(gyroscope), and finding the timesynchronization for the same.
   
Relative pose estimation
========================
   estimate_pose
   
Time synchronization
========================
    sync_camera_gyro
    sync_camera_gyro_manual
    refine_time_offset

Utilities
========================    
    good_sequences_to_track
    IMU
    Camera
"""

__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2013, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

from .imu import IMU
from .camera import Camera
from .timesync import sync_camera_gyro, sync_camera_gyro_manual, refine_time_offset, good_sequences_to_track
from .pose import estimate_pose