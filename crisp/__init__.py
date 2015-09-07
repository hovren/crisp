# -*- coding: utf-8 -*-
"""
========================================
Camera-to-IMU calibration toolbox
========================================
This package solves the task of finding the parameters
that relate gyroscope data with video data.

To run, please see the README or the class AutoCalibrator.
"""
from __future__ import absolute_import

__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2015, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

from .camera import CameraModel, AtanCameraModel, OpenCVCameraModel
from .stream import GyroStream, VideoStream, OpenCvVideoStream
from .calibration import AutoCalibrator, CalibrationError, InitializationError