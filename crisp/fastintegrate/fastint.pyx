# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:10:40 2014

@author: hannes
"""
cimport cython

import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double sqrt(double x)

DTYPE = np.float
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
def integrate_gyro_quaternion_uniform(np.ndarray[DTYPE_t,ndim=2] gyro_data, np.float dt):
    #NB: Quaternion q = [a, n1, n2, n3], scalar first
    cdef unsigned int N = gyro_data.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] q_list = np.empty((N, 4)) # Nx4 quaternion list
    
    # Iterate over all (except first)
    cdef unsigned int i, j
    cdef DTYPE_t wx, wy, wz
    cdef DTYPE_t q0, q1, q2, q3
        
    #cdef np.ndarray[DTYPE_t, ndim=1] qnew = np.zeros((4,))
    cdef DTYPE_t qnorm
    cdef DTYPE_t dt_half = dt / 2.0
    
    # Initial rotation
    q0 = 1.0
    q1 = q2 = q3 = 0.0    
    
    for i in range(N):
        wx = gyro_data[i,0]
        wy = gyro_data[i,1]
        wz = gyro_data[i,2]
                
        q_list[i, 0] = q0 + dt_half * (-wx*q1 -wy*q2 -wz*q3)
        q_list[i, 1] = q1 + dt_half * (q0*wx + q2*wz - wy*q3)
        q_list[i, 2] = q2 + dt_half * (wy*q0 -wz*q1 + wx*q3)
        q_list[i, 3] = q3 + dt_half * (wz*q0 + wy*q1 -wx*q2)

        # Normalize
        qnorm = sqrt(q_list[i, 0]**2 + q_list[i, 1]**2 + q_list[i, 2]**2 + q_list[i, 3]**2)
        for j in range(4):
            q_list[i, j] /= qnorm
        
        # New prev values
        q0 = q_list[i, 0]
        q1 = q_list[i, 1]
        q2 = q_list[i, 2]
        q3 = q_list[i, 3]
    return q_list