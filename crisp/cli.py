# -*- coding: utf-8 -*-
"""
Command line interface helpers
"""

__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2013, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

import os
import csv
import logging
logger = logging.getLogger()

from scipy.io import loadmat

from .imu import ArduIMU, IMU

def load_imu_from_file(imu_file):
    try:
        imu = ArduIMU(imu_file)
        logger.debug("Loaded IMU data from ArduIMU logfile %s" % imu_file)
        return imu
    except (IOError, ValueError):
        logger.debug("%s did not load as ArduIMU file" % imu_file)
        
    try:
        imu = IMU.from_mat_file(imu_file)
        logger.debug("Loaded IMU data from .mat-file %s" % imu_file)
        return imu
    except IOError:
        logger.debug("%s did not load as MAT file" % imu_file)
    
    return None

#--------------------------------------------------------------------------

def load_vars_from_mat(filename, var_dict):
    result_dict = {}
    M = loadmat(filename)
    for var_name, possible_names in var_dict.items():
        val = None
        for key in possible_names:
            logger.debug("Trying %s for variable %s in %s" % (key, var_name, filename))
            try:
                val = M[key]
                break
            except KeyError:
                pass
        if val is None:
            raise ValueError("Could not find a candidate for requested variable %s" % var_name)
        result_dict[var_name]= val
            
    return result_dict
    
#--------------------------------------------------------------------------

def load_images_timestamps_from_csv(image_csv):
    (root, _) = os.path.split(image_csv)
    timestamps = []
    files = []
    with open(image_csv, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            filename, timestamp = row[:2]
            _, filename = os.path.split(filename)
            timestamp = float(timestamp)
            image_path = os.path.join(root, filename)
            timestamps.append(timestamp)
            files.append(image_path)
    return files, timestamps