# -*- coding: utf-8 -*-
"""
ZNCC using Pyramids
"""

__author__ = "Per-Erik Forssén"
__copyright__ = "Copyright 2013, Per-Erik Forssén"
__license__ = "GPL"
__email__ = "perfo@isy.liu.se"

import logging
logger = logging.getLogger()

import numpy as np

def gaussian_kernel(gstd):
    """Generate odd sized truncated Gaussian

    The generated filter kernel has a cutoff at $3\sigma$
    and is normalized to sum to 1

    Parameters
    -------------
    gstd : float
            Standard deviation of filter

    Returns
    -------------
    g : ndarray
            Array with kernel coefficients
    """
    Nc = np.ceil(gstd*3)*2+1
    x = np.linspace(-(Nc-1)/2,(Nc-1)/2,Nc,endpoint=True)
    g = np.exp(-.5*((x/gstd)**2))
    g = g/np.sum(g)

    return g

def subsample(time_series, downsample_factor):
    """Subsample with Gaussian prefilter

    The prefilter will have the filter size $\sigma_g=.5*ssfactor$

    Parameters
    --------------
    time_series : ndarray
            Input signal
    downsample_factor : float
            Downsampling factor
       
    Returns
    --------------
       ts_out : ndarray
            The downsampled signal
    """
    Ns = np.int(np.floor(np.size(time_series)/downsample_factor))
    g = gaussian_kernel(0.5*downsample_factor)
    ts_blur = np.convolve(time_series,g,'same')
    ts_out = np.zeros((Ns,1), dtype='float64')
    for k in range(0,Ns):
        cpos  = (k+.5)*downsample_factor-.5
        cfrac = cpos-np.floor(cpos)
        cind  = np.floor(cpos)
        if cfrac>0:
            ts_out[k]=ts_blur[cind]*(1-cfrac)+ts_blur[cind+1]*cfrac
        else:
            ts_out[k]=ts_blur[cind]
            
    return ts_out
    
def upsample(time_series, scaling_factor):
    """Upsample using linear interpolation

    The function uses replication of the value at edges

    Parameters
    --------------
    time_series : ndarray
            Input signal
    scaling_factor : float
            The factor to upsample with
    
    Returns
    --------------
    ts_out  : ndarray
            The upsampled signal
    """
    Ns0 = np.size(time_series)
    Ns  = np.int(np.floor(np.size(time_series)*scaling_factor))
    ts_out = np.zeros((Ns,1), dtype='float64')
    for k in range(0,Ns):
        cpos  = np.min([Ns0-1,np.max([0.,(k+0.5)/scaling_factor-0.5])])
        cfrac = cpos-np.floor(cpos)
        cind  = np.floor(cpos)
        #print "cpos=%f cfrac=%f cind=%d", (cpos,cfrac,cind)
        if cfrac>0:
            ts_out[k]=time_series[cind]*(1-cfrac)+time_series[cind+1]*cfrac
        else:
            ts_out[k]=time_series[cind]
        
    return ts_out

def do_binning(time_series,factor):
    Ns = np.size(time_series)/factor    
    ts_out = np.zeros((Ns,1), dtype='float64')
    for k in range(0,Ns):
        ts_out[k]=0
        for l in range(0,factor):
            ts_out[k] += time_series[k*factor+l]
        ts_out[k] /= factor
    
    return ts_out

def create_pyramid(time_series,octaves):
    pyr_out = [time_series ]
    for k in range(0,octaves):
        pyr_out.append(do_binning(pyr_out[-1],2))
        
    return pyr_out

def zncc(ts1,ts2):
    """Zero mean normalised cross-correlation (ZNCC)

    This function does ZNCC of two signals, ts1 and ts2
    Normalisation by very small values is avoided by doing
    max(nmin,nvalue)

    Parameters
    --------------
    ts1 : ndarray
            Input signal 1 to be aligned with
    ts2 : ndarray
            Input signal 2

    Returns
    --------------
    best_shift : float
            The best shift of *ts1* to align it with *ts2*
    ts_out : ndarray
            The correlation result
    """
    # Output is the same size as ts1
    Ns1 = np.size(ts1)
    Ns2 = np.size(ts2)
    ts_out = np.zeros((Ns1,1), dtype='float64')

    ishift = np.floor(Ns2/2) # origin of ts2

    t1m = np.mean(ts1)
    t2m = np.mean(ts2)
            
    for k in range(0,Ns1):
        lstart = np.int(ishift-k)
        if lstart<0 :
            lstart=0
        lend = np.int(ishift-k+Ns2)
        imax = np.int(np.min([Ns2,Ns1-k+ishift]))
        if lend>imax :
            lend=imax
        csum = 0
        ts1sum = 0
        ts1sum2 = 0
        ts2sum = 0
        ts2sum2 = 0
        
        Nterms = lend-lstart        
        for l in range(lstart,lend):
            csum    += ts1[k+l-ishift]*ts2[l]
            ts1sum  += ts1[k+l-ishift]
            ts1sum2 += ts1[k+l-ishift]*ts1[k+l-ishift]
            ts2sum  += ts2[l]
            ts2sum2 += ts2[l]*ts2[l]
        ts1sum2 = np.max([t1m*t1m*100,ts1sum2])-ts1sum*ts1sum/Nterms
        ts2sum2 = np.max([t2m*t2m*100,ts2sum2])-ts2sum*ts2sum/Nterms
        #ts_out[k]=csum/np.sqrt(ts1sum2*ts2sum2)
        ts_out[k]=(csum-2.0*ts1sum*ts2sum/Nterms+ts1sum*ts2sum/Nterms/Nterms)/np.sqrt(ts1sum2*ts2sum2)
    best_shift = np.argmax(ts_out)-ishift
    return best_shift, ts_out

def refine_correlation(ts1,ts2,shift_guess):
    """Refine a rough guess of shift by evaluating ZNCC for similar values

    Shifts of *ts1* are tested in the range [-2:2]
    Refine a rough guess of shift, by trying neighbouring ZNCC values
    in the range [-2:2]

    Parameters
    ----------------
    ts1 : list_like
            The first timeseries
    ts2 : list_like
            The seconds timeseries
    shift_guess : float
            The guess to start from

    Returns
    ---------------
    best_shift : float
            The best shift of those tested
    ts_out : ndarray
            Computed correlation values
    """    
    Ns1 = np.size(ts1)
    Ns2 = np.size(ts2)
    ts_out = np.zeros((5,1))

    ishift = np.floor(Ns2/2) # origin of ts2
    k_offset = shift_guess-2+ishift # Try shifts starting with this one

    t1m = np.mean(ts1)
    t2m = np.mean(ts2)

    for k in range(0,5):
        km = k+k_offset
        lstart = np.int(ishift-km)
        if lstart<0 :
            lstart=0
        lend = np.int(ishift-km+Ns2)
        imax = np.int(np.min([Ns2,Ns1-km+ishift]))
        if lend>imax :
            lend=imax
        csum = 0
        ts1sum = 0
        ts1sum2 = 0
        ts2sum = 0
        ts2sum2 = 0
        
        Nterms = lend-lstart        
        for l in range(lstart,lend):
            csum    += ts1[km+l-ishift]*ts2[l]
            ts1sum  += ts1[km+l-ishift]
            ts1sum2 += ts1[km+l-ishift]*ts1[km+l-ishift]
            ts2sum  += ts2[l]
            ts2sum2 += ts2[l]*ts2[l]
        ts1sum2 = np.max([t1m*t1m*100,ts1sum2])-ts1sum*ts1sum/Nterms
        ts2sum2 = np.max([t2m*t2m*100,ts2sum2])-ts2sum*ts2sum/Nterms
        #ts_out[k]=csum/np.sqrt(ts1sum2*ts2sum2)
        ts_out[k]=(csum-2.0*ts1sum*ts2sum/Nterms+ts1sum*ts2sum/Nterms/Nterms)/np.sqrt(ts1sum2*ts2sum2)

    best_shift = np.argmax(ts_out)+k_offset-ishift    
    return best_shift, ts_out    

def find_shift_pyr(ts1,ts2,nlevels):
    """
    Find shift that best aligns two time series

    The shift that aligns the timeseries ts1 with ts2.
    This is sought using zero mean normalized cross correlation (ZNCC) in a coarse to fine search with an octave pyramid on nlevels levels.

    Parameters
    ----------------
    ts1 : list_like
            The first timeseries
    ts2 : list_like
            The seconds timeseries
    nlevels : int
            Number of levels in pyramid

    Returns
    ----------------
       ts1_shift : float
               How many samples to shift ts1 to align with ts2
    """
    pyr1 = create_pyramid(ts1,nlevels)
    pyr2 = create_pyramid(ts2,nlevels)
    
    logger.debug("pyramid size = %d" % len(pyr1))
    logger.debug("size of first element %d " % np.size(pyr1[0]))
    logger.debug("size of last element %d " % np.size(pyr1[-1]))

    ishift, corrfn = zncc(pyr1[-1],pyr2[-1])

    for k in range(1,nlevels+1):
        ishift, corrfn = refine_correlation(pyr1[-k-1],pyr2[-k-1],ishift*2)

    return ishift