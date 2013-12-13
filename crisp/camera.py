# -*- coding: utf-8 -*-
"""
Camera module
"""
__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2013, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

import os
import glob
import logging
logger = logging.getLogger()

import numpy as np
import cv2
import scipy.interpolate

from . import remove_slp

class Camera(object):
    def __init__(self):
        self.K = None
        self.readout_time = 0.0
        self.timestamps = []
        self.files = [] # Filenames, same index corresponds to timestamps list
        self._images = []
    
    def load_image(self, filename):
        return cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    @property
    def images(self):
        if len(self._images) < 1:
            self._images = [self.load_image(f) for f in self.files]
        return self._images
        
    def image_sequence(self, first=0, last=-1):
        file_slice = self.files[first:] if last == -1 else self.files[first:last+1]
        for filename in file_slice:
            img = self.load_image(filename)
            if img is None:
                raise IOError("Failed to load file %s" % filename)
            yield(img)
  
class DepthCamera(Camera):
    pass
   
class Kinect(object):
    DEFAULT_DEPTH_NIR_SHIFT = (1.5, -3.5)
    DEFAULT_OPARS = [0, 0]
    DEFAULT_NIR_K = np.array([[ 582.67750309,    0.        ,  314.96824757],
                              [   0.        ,  584.65055308,  248.16240365],
                              [   0.        ,    0.        ,    1.        ]])
    DEFAULT_RGB_K = np.array([[ 519.83879135,    0.        ,  313.55797842],
                              [   0.        ,  520.71387   ,  267.59027502],
                              [   0.        ,    0.        ,    1.        ]])
    
    class NirCamera(Camera):
        def load_image(self, filename):
            img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_UNCHANGED).astype('float32')
            img = remove_slp.remove_slp(img)
            img = Kinect.NirCamera.convert_nbit_float32_to_uint8(img, 10)
            return img
        
        @staticmethod
        def convert_nbit_float32_to_uint8(img, nbit):
            "Converts a float32 image to a uint8 under the assumption that the float32 image has a bit depth of nbit. Will copy input buffer before making changes to it"
            img = img.copy()
            if nbit != 8:
                img *= 255.0 / (2**nbit - 1)
            return img.astype('uint8')
    
    def __init__(self, depth_camera, video_camera, mode):
        self.depth_camera = depth_camera
        self.video_camera = video_camera
        self.video_mode = mode
        self.set_default_params()
        
    def set_default_params(self):
        self.opars =  Kinect.DEFAULT_OPARS # Depth conversion parameters
        self.depth_nir_shift = Kinect.DEFAULT_DEPTH_NIR_SHIFT # NIR to depth shift params
        self.depth_camera.K = Kinect.DEFAULT_NIR_K
        self.video_camera.K = Kinect.DEFAULT_RGB_K if self.video_mode == 'rgb' else Kinect.DEFAULT_NIR_K
    
    @classmethod
    def from_directory(cls, datadir, video_mode='any'):        
        # ) Load list of NIR files
        nir_file_list = glob.glob(os.path.join(datadir, 'i-*.pgm'))    
        nir_file_list.sort()

        # ) Load list of RGB files
        rgb_file_list = glob.glob(os.path.join(datadir, 'r-*.ppm'))
        rgb_file_list.sort()

        if video_mode == 'any':
            video_mode = 'rgb' if len(rgb_file_list) > len(nir_file_list) else 'nir'
                
        video_files = rgb_file_list if video_mode == 'rgb' else nir_file_list
        
        depth_camera = DepthCamera()
        # FIXME: KinectNirCamera only handles 10-bit NIR right now
        video_camera = Kinect.NirCamera() if video_mode == 'nir' else Camera()
        
        #) Load list of depth files
        depth_files = glob.glob(os.path.join(datadir, 'd-*.pgm'))
        depth_files.sort()
                
        # Get a consistent set of files
        video_files = Kinect.purge_bad_timestamp_files(video_files)
        depth_files = Kinect.purge_bad_timestamp_files(depth_files)

        if video_mode == 'nir':
            (video_files, depth_files, _, _) =  Kinect.find_nir_file_with_missing_depth(video_files, depth_files)
                    
        depth_timestamps = Kinect.timestamps_from_file_list(depth_files)
        video_timestamps = Kinect.timestamps_from_file_list(video_files)
        
        depth_camera.timestamps = depth_timestamps
        depth_camera.files = depth_files
        video_camera.timestamps = video_timestamps
        video_camera.files = video_files
        
        instance = cls(depth_camera, video_camera, video_mode)
                    
        return instance
    
    @staticmethod
    def timestamp_from_filename(fname):
        "Extract timestamp from filename"    
        ts = int(fname.split('-')[-1].split('.')[0])
        return ts
    
    @staticmethod
    def timestamps_from_file_list(file_list):
        "Take list of Kinect filenames (without path) and extracts timestamps while accounting for timestamp overflow (returns linear timestamps)."
        timestamps = np.array([Kinect.timestamp_from_filename(fname) for fname in file_list])

        # Handle overflow
        diff = np.diff(timestamps)
        idxs = np.flatnonzero(diff < 0)
        ITEM_SIZE = 2**32
        for i in idxs:
            timestamps[i+1:] += ITEM_SIZE

        return timestamps.flatten()

    @staticmethod
    def detect_bad_timestamps(ts_list):
        EXPECTED_DELTA = 2002155 # Expected time between IR frames
        MAX_DIFF = EXPECTED_DELTA / 4
        bad_list = []
        for frame_num in xrange(1, len(ts_list)):
            diff = ts_list[frame_num] - ts_list[frame_num-1]
            if abs(diff - EXPECTED_DELTA) > MAX_DIFF:
                bad_list.append(frame_num)

        return bad_list

    @staticmethod
    def purge_bad_timestamp_files(file_list):
        "Given a list of image files, find bad frames, remove them and modify file_list"
        MAX_INITIAL_BAD_FRAMES = 15
        bad_ts = Kinect.detect_bad_timestamps(Kinect.timestamps_from_file_list(file_list))
        
        # Trivial case
        if not bad_ts:
            return file_list

        # No bad frames after the initial allowed
        last_bad = max(bad_ts)
        if last_bad >= MAX_INITIAL_BAD_FRAMES:
            raise Exception('Only 15 initial bad frames are allowed, but last bad frame is %d' % last_bad)

        # Remove all frames up to the last bad frame
        for i in range(last_bad + 1):
            os.remove(file_list[i])

        # Purge from the list
        file_list = file_list[last_bad+1:]

        return file_list # Not strictly needed since Python will overwrite the list

    @staticmethod
    def depth_file_for_nir_file(video_filename, depth_file_list):
        """Returns the corresponding depth filename given a NIR filename"""
        (root, filename) = os.path.split(video_filename)
        needle_ts = int(filename.split('-')[2].split('.')[0])
        haystack_ts_list = np.array(Kinect.timestamps_from_file_list(depth_file_list))
        haystack_idx = np.flatnonzero(haystack_ts_list == needle_ts)[0]
        depth_filename = depth_file_list[haystack_idx]
        return depth_filename
        
    @staticmethod 
    def depth_file_for_rgb_file(rgb_filename, rgb_file_list, depth_file_list):
        """Returns the *closest* depth file from an RGB filename"""
        (root, filename) = os.path.split(rgb_filename)
        rgb_timestamps = np.array(Kinect.timestamps_from_file_list(rgb_file_list))
        depth_timestamps = np.array(Kinect.timestamps_from_file_list(depth_file_list))
        needle_ts = rgb_timestamps[rgb_file_list.index(rgb_filename)]
        haystack_idx = np.argmin(np.abs(depth_timestamps - needle_ts))
        depth_filename = depth_file_list[haystack_idx]
        return depth_filename

    @staticmethod
    def find_nir_file_with_missing_depth(video_file_list, depth_file_list):
        "Remove all files without its own counterpart. Returns new lists of files"
        new_video_list = []
        new_depth_list = []
        for fname in video_file_list:
            try:
                depth_file = Kinect.depth_file_for_nir_file(fname, depth_file_list)                
                new_video_list.append(fname)
                new_depth_list.append(depth_file)
            except IndexError: # Missing file
                pass
                
        # Purge bad files
        bad_nir = [f for f in video_file_list if f not in new_video_list]
        bad_depth = [f for f in depth_file_list if f not in new_depth_list]
        
        return (new_video_list, new_depth_list, bad_nir, bad_depth)
    
    def disparity_image_to_distance(self, dval_img):
        "Convert image of Kinect disparity values to distance (linear method)"
        dist_img = dval_img / 2048.0
        dist_img = 1 / (self.opars[0]*dist_img + self.opars[1])
        return dist_img
        
    def align_depth_to_nir(self, depth_img):
        vpad = np.zeros((4,640))
        depth_new = np.vstack((vpad, depth_img, vpad))
        x, y = np.mgrid[0:np.size(depth_new,1), 0:np.size(depth_new,0)]
        xs = x + self.depth_nir_shift[1]
        ys = y + self.depth_nir_shift[0]
    
        points = np.dstack((x,y)).reshape([-1,2])
    
        depth_new = scipy.interpolate.griddata(points, depth_new[points[:,1],points[:,0]].flatten(), (xs.T, ys.T), method='nearest')
        return depth_new 
        
    def depthmap_for_nir(self, nir_filename):
        if not self.video_mode == 'nir':
            raise Exception("Tried to get depth map from NIR, but capture used RGB")
        depth_filename = Kinect.depth_file_for_nir_file(nir_filename, self.video_camera.files, self.depth_camera.files)
        depth_img = cv2.imread(depth_filename, cv2.CV_LOAD_IMAGE_UNCHANGED)
        depth_img = self.disparity_image_to_distance(depth_img)
        depth_img = self.align_depth_to_nir(depth_img)
        return depth_img