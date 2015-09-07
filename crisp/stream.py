# -*- coding: utf-8 -*-
from __future__ import division

"""
Input streams module
"""

__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2015, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

import os
import logging
logger = logging.getLogger('crisp')

import cv2
import numpy as np

from . import fastintegrate, tracking

class GyroStream(object):
    def __init__(self):
        self.__last_dt = None
        self.__last_q = None
        self.data = None # Arranged as Nx3 where N is number of samples

    @classmethod
    def from_csv(cls, filename):
        """Create gyro stream from CSV data

        Load data from a CSV file.
        The data must be formatted with three values per line: (x, y, z)
        where x, y, z is the measured angular velocity (in radians) of the specified axis.

        Parameters
        -------------------
        filename : str
            Path to the CSV file

        Returns
        ---------------------
        GyroStream
            A gyroscope stream
        """
        instance = cls()
        instance.data = np.loadtxt(filename, delimiter=',')
        return instance

    @classmethod
    def from_data(cls, data):
        """Create gyroscope stream from data array

        Parameters
        -------------------
        data : (N, 3) ndarray
            Data array of angular velocities (rad/s)

        Returns
        -------------------
        GyroStream
            Stream object
        """
        if not data.shape[1] == 3:
            raise ValueError("Gyroscope data must have shape (N, 3)")

        instance = cls()
        instance.data = data
        return instance


    @property
    def num_samples(self):
        return self.data.shape[0]

    def integrate(self, dt):
        """Integrate gyro measurements to orientation using a uniform sample rate.

        Parameters
        -------------------
        dt : float
            Sample distance in seconds

        Returns
        ----------------
        orientation : (4, N) ndarray
                    Gyroscope orientation in quaternion form (s, q1, q2, q3)
        """
        if not dt == self.__last_dt:
            self.__last_q = fastintegrate.integrate_gyro_quaternion_uniform(self.data, dt)
            self.__last_dt = dt
        return self.__last_q

class VideoStream(object):
    """Video stream representation

    This is the base class for all video streams, and should normally not be used directly.
    Instead you should use a VideoStream subclass that can work on the data you have.

    The concept of a video stream can be summarized as something that
    "provides frames of video data captured by a single camera".

    A VideoStream object is iterable to allow reading frames easily::

        stream = VideoStreamSubClass(SOME_PARAMETER)
        for frame in stream:
            do_stuff(frame)

    """
    def __init__(self, camera_model):
        """Create a VideoStream object

        Parameters
        ----------------
        camera_model : CameraModel
                     Camera model used by this stream
        """
        self._flow = None
        self.camera_model = camera_model

    def __iter__(self):
        return self._frames()

    def _frames(self):
        raise NotImplementedError("{} does not implement the _frames() method used to extract frames".format(self.__class__.__name__))

    @classmethod
    def from_file(cls, camera_model, filename):
        """Create stream automatically from filename.

        Note
        --------------------
        This currently only works with video files that are readable by OpenCV

        Parameters
        --------------------
        camera_model : CameraModel
            Camera model to use with this stream
        filename : str
            The filename to load the stream data from

        Returns
        --------------------
        VideoStream
            Video stream of a suitable sub class
        """
        # TODO: Other subclasses
        return OpenCvVideoStream(camera_model, filename)

    def project(self, points):
        """Project 3D points to image coordinates.

        This projects 3D points expressed in the camera coordinate system to image points.

        Parameters
        --------------------
        points : (3, N) ndarray
            3D points

        Returns
        --------------------
        image_points : (2, N) ndarray
            The world points projected to the image plane of the camera used by the stream
        """
        return self.camera_model.project(points)

    def unproject(self, image_points):
        """Find (up to scale) 3D coordinate of an image point

        This is the inverse of the `project` function.
        The resulting 3D points are only valid up to an unknown scale.

        Parameters
        ----------------------
        image_points : (2, N) ndarray
            Image points

        Returns
        ----------------------
        points : (3, N) ndarray
            3D coordinates (valid up to scale)
        """
        return self.camera_model.unproject(image_points)

    @property
    def frame_rate(self):
        return self.camera_model.frame_rate
    
    @property
    def flow(self):
        if self._flow is None:
            logger.debug("Generating optical flow magnitude. This can take minutes depending on video length")
            self._flow = tracking.optical_flow_magnitude(self)
            #self.__generate_flow()
        return self._flow
    
    def __generate_flow(self):
        logger.debug("Generating flow")
        flow_list = []
        prev_im = None
        prev_pts = None
        for i, im in enumerate(self):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if prev_im is None:
                prev_im = im
            else:
                pts, initial_pts = tracking.track_points(prev_im, im)
                dist = np.sqrt(np.sum((pts - initial_pts)**2, axis=1))
                flow_list.append(np.mean(dist))
        self._flow = np.array(flow_list)

class OpenCvVideoStream(VideoStream):
    """Video stream that uses OpenCV to extract image data.

    This stream class uses the OpenCV VideoCapture class and can thus handle any
    video type that is supported by the installed version of OpenCV.
    It can only handle video files, and not live streams.
    """
    def __init__(self, camera_model, filename, start_time=0.0, duration=None):
        """Create video stream

        Parameters
        ---------------
        camera_model : CameraModel
            Camera model
        filename : str
            Path to the video file
        start_time : float
            The time in seconds where to start capturing (USE WITH CAUTION)
        duration : float
            Duration in seconds to capture (USE WITH CAUTION)

        Notes
        -------------------
        You can specify the start time and duration you want to use for the capture.
        However, be advised that this may or may not work depending on the type of video data
        and your installation of OpenCV. Use with caution!
        """
        super(OpenCvVideoStream, self).__init__(camera_model)
        self.filename = filename
        self.start_time = start_time
        self.duration = duration
        self.step = 1

    def _frames(self):
        vc = cv2.VideoCapture(self.filename)
        if not vc.isOpened():
            raise IOError("Failed to open '{}'. Either there is something wrong with the file or OpenCV does not have the correct codec".format(self.filename))
        # OpenCV does something really stupid: to set the frame we need to set it twice and query in between
        t = self.start_time * 1000. # turn to milliseconds
        t2 = t + self.duration*1000.0 if self.duration is not None else None

        for i in range(2): # Sometimes needed for setting to stick
            vc.set(cv2.cv.CV_CAP_PROP_POS_MSEC, t)
            vc.read()
        t = vc.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
        counter = 0
        retval = True
        while retval and (t2 is None or (t2 is not None and t < t2)):
            retval, im = vc.read()
            if retval:
                if np.mod(counter, self.step) == 0:
                    yield im
            elif t2 is not None:
                raise IOError("Failed to get frame at time %.2f" % t)
            else:
                pass # Loop will end normally
            t = vc.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
            counter += 1
