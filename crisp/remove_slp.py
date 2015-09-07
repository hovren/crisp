#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

"""
Kinect NIR Structured Light Pattern removal
"""

__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2013, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

# Adapted from MATLAB code written by Per-Erik Forssén (perfo@isy.liu.se)

#--------------------------------------------------------------------------
# Includes
#--------------------------------------------------------------------------

import cv2
import numpy as np

#--------------------------------------------------------------------------
# Default parameters
#--------------------------------------------------------------------------

GSTD1 = 0.575
GSTD2 = 2.3
GSTD3 = 3.4
W = 9.0
KSIZE = 19 # MATLAB equivalent of -9:9
EPS = 2.2204E-16

#--------------------------------------------------------------------------
# Public functions
#--------------------------------------------------------------------------
def remove_slp(img, gstd1=GSTD1, gstd2=GSTD2, gstd3=GSTD3, ksize=KSIZE, w=W):
    """Remove the SLP from kinect IR image
    
    The input image should be a float32 numpy array, and should NOT be a square root image
    Parameters
    ------------------
    img : (M, N) float ndarray
            Kinect NIR image with SLP pattern
    gstd1 : float
            Standard deviation of gaussian kernel 1
    gstd2 : float
            Standard deviation of gaussian kernel 2
    gstd3 : float
            Standard deviation of gaussian kernel 3
    ksize : int
            Size of kernel (should be odd)
    w   : float
            Weighting factor

    Returns
    ------------------
    img_noslp : (M,N) float ndarray
            Input image with SLP removed
    """
    gf1 = cv2.getGaussianKernel(ksize, gstd1)
    gf2 = cv2.getGaussianKernel(ksize, gstd2)
    gf3 = cv2.getGaussianKernel(ksize, gstd3)
    sqrtimg = cv2.sqrt(img)
    p1 = cv2.sepFilter2D(sqrtimg, -1, gf1, gf1)
    p2 = cv2.sepFilter2D(sqrtimg, -1, gf2, gf2)
    maxarr = np.maximum(0, (p1 - p2) / p2)
    minarr = np.minimum(w * maxarr, 1)
    p = 1 - minarr
    nc = cv2.sepFilter2D(p, -1, gf3, gf3) + EPS
    output = cv2.sepFilter2D(p*sqrtimg, -1, gf3, gf3)
    output = (output / nc) ** 2 # Since input is sqrted
    
    return output
