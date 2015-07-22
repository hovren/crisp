# -*- coding: utf-8 -*-
"""
Rotation handling module
"""

__author__ = "Hannes Ovrén"
__copyright__ = "Copyright 2013, Hannes Ovrén"
__license__ = "GPL"
__email__ = "hannes.ovren@liu.se"

import numpy as np

from numpy.testing import assert_almost_equal

from . import ransac

#------------------------------------------------------------------------------

def procrustes(X, Y, remove_mean=False):
    """Orthogonal procrustes problem solver
    
    The procrustes problem  finds the best rotation R, and translation t
    where
        X = R*Y + t
    
    The number of points in X and Y must be at least 2.
    For the minimal case of two points, a third point is temporarily created
    and used for the estimation.
    
    Parameters
    -----------------
    X : (3, N) ndarray
            First set of points
    Y : (3, N) ndarray
            Second set of points
    remove_mean : bool
            If true, the mean is removed from X and Y before solving the
            procrustes problem. Can yield better results in some applications.
            
    Returns
    -----------------
    R : (3,3) ndarray
            Rotation component
    t : (3,) ndarray
            Translation component (None if remove_mean is False)
 """

    assert X.shape == Y.shape
    assert X.shape[0] > 1
    
    # Minimal case, create third point using cross product
    if X.shape[0] == 2:
        X3 = np.cross(X[:,0], X[:,1], axis=0)
        X = np.hstack((X, X3 / np.linalg.norm(X3)))
        Y3 = np.cross(Y[:,0], Y[:,1], axis=0)
        Y = np.hstack((Y, Y3 / np.linalg.norm(Y3)))
        
    
    D, N = X.shape[:2]
    if remove_mean:
        mx = np.mean(X, axis=1).reshape(D, 1)
        my = np.mean(Y, axis=1).reshape(D, 1)
        Xhat = X - mx
        Yhat = Y - my
    else:
        Xhat = X
        Yhat = Y


    (U, S, V) = np.linalg.svd((Xhat).dot(Yhat.T))

    Dtmp = np.eye(Xhat.shape[0])
    Dtmp[-1,-1] = np.linalg.det(U.dot(V))

    R_est = U.dot(Dtmp).dot(V)

    # Now X=R_est*(Y-my)+mx=R_est*Y+t_est
    if remove_mean:
        t_est= mx - R_est.dot(my)
    else:
        t_est = None
    return (R_est, t_est)

#--------------------------------------------------------------------------

def rotation_matrix_to_axis_angle(R):
    """Convert a 3D rotation matrix to a 3D axis angle representation
    
    Parameters
    ---------------
    R : (3,3) array
        Rotation matrix
        
    Returns
    ----------------
    v : (3,) array
        (Unit-) rotation angle
    theta : float
        Angle of rotations, in radians
    
    Note
    --------------
    This uses the algorithm as described in Multiple View Geometry, p. 584
    """
    assert R.shape == (3,3)
    assert_almost_equal(np.linalg.det(R), 1.0, err_msg="Not a rotation matrix: determinant was not 1")
    S, V = np.linalg.eig(R)
    k = np.argmin(np.abs(S - 1.))
    s = S[k]
    assert_almost_equal(s, 1.0, err_msg="Not a rotation matrix: No eigen value s=1")
    v = np.real(V[:, k]) # Result is generally complex
    
    vhat = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
    sintheta = 0.5 * np.dot(v, vhat)
    costheta = 0.5 * (np.trace(R) - 1)
    theta = np.arctan2(sintheta, costheta)
    
    return (v, theta)

#--------------------------------------------------------------------------

def axis_angle_to_rotation_matrix(v, theta):
    """Convert rotation from axis-angle to rotation matrix
    
        Parameters
    ---------------
    v : (3,) ndarray
            Rotation axis (normalized)
    theta : float
            Rotation angle (radians)

    Returns
    ----------------
    R : (3,3) ndarray
            Rotation matrix
    """
    if np.abs(theta) < np.spacing(1):
        return np.eye(3)
    else:
        v = v.reshape(3,1)
        np.testing.assert_almost_equal(np.linalg.norm(v), 1.)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        vvt = np.dot(v, v.T)
        R = np.eye(3)*np.cos(theta) + (1 - np.cos(theta))*vvt + vx * np.sin(theta)
        return R

#--------------------------------------------------------------------------

def quat_to_rotation_matrix(q):
    """Convert unit quaternion to rotation matrix
    
    Parameters
    -------------
    q : (4,) ndarray
            Unit quaternion, scalar as first element

    Returns
    ----------------
    R : (3,3) ndarray
            Rotation matrix
    
    """
    q = q.flatten()
    assert q.size == 4
    assert_almost_equal(np.linalg.norm(q), 1.0, err_msg="Not a unit quaternion!")
    qq = q ** 2
    R = np.array([[qq[0] + qq[1] - qq[2] - qq[3], 2*q[1]*q[2] -
2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
                [2*q[1]*q[2] + 2*q[0]*q[3], qq[0] - qq[1] + qq[2] -
qq[3], 2*q[2]*q[3] - 2*q[0]*q[1]],
                [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1],
qq[0] - qq[1] - qq[2] + qq[3]]])
    return R

#--------------------------------------------------------------------------

def integrate_gyro_quaternion(gyro_ts, gyro_data):
    """Integrate angular velocities to rotations
    
    Parameters
    ---------------
    gyro_ts : ndarray
            Timestamps
    gyro_data : (3, N) ndarray
            Angular velocity measurements
    
    Returns
    ---------------
    rotations : (4, N) ndarray
            Rotation sequence as unit quaternions (first element scalar)
    
    """
    #NB: Quaternion q = [a, n1, n2, n3], scalar first
    q_list = np.zeros((gyro_ts.shape[0], 4)) # Nx4 quaternion list
    q_list[0,:] = np.array([1, 0, 0, 0]) # Initial rotation (no rotation)
    
    # Iterate over all (except first)
    for i in xrange(1, gyro_ts.size):
        w = gyro_data[i]
        dt = gyro_ts[i] - gyro_ts[i - 1]
        qprev = q_list[i - 1]
        
        A = np.array([[0,    -w[0],  -w[1],  -w[2]],
                     [w[0],  0,      w[2],  -w[1]],
                     [w[1], -w[2],   0,      w[0]],
                     [w[2],  w[1],  -w[0],   0]])
        qnew = (np.eye(4) + (dt/2.0) * A).dot(qprev)
        qnorm = np.sqrt(np.sum(qnew ** 2))
        qnew /= qnorm
        q_list[i] = qnew
         
    return q_list

#--------------------------------------------------------------------------

def slerp(q1, q2, u):
    """SLERP: Spherical linear interpolation between two unit quaternions.
    
    Parameters
    ------------
    q1 : (4, ) ndarray
            Unit quaternion (first element scalar)
    q2 : (4, ) ndarray
            Unit quaternion (first element scalar)
    u : float
            Interpolation factor in range [0,1] where 0 is first quaternion 
            and 1 is second quaternion.
            
    Returns
    -----------
    q : (4,) ndarray
            The interpolated unit quaternion
    """
    q1 = q1.flatten()
    q2 = q2.flatten()
    assert q1.shape == q2.shape
    assert q1.size == 4
    costheta = np.sqrt(np.sum(q1 * q2))    
    theta = np.arccos(costheta)
    
    f1 = np.sin((1.0 - u)*theta) / np.sin(theta)
    f2 = np.sin(u*theta) / np.sin(theta)
    
    # Shortest path is wanted, so conjugate if necessary
    if costheta < 0:
        f1 = -f1
        q = f1*q1 + f2*q2
        q = q / np.sqrt(np.sum(q**2)) # Normalize
    else:
        q = f1*q1 + f2*q2 
        q = q / np.sqrt(np.sum(q**2)) # Normalize
    return q

#--------------------------------------------------------------------------

def estimate_rotation_procrustes_ransac(x, y, camera, threshold, inlier_ratio=0.75, do_translation=False):
    """Calculate rotation between two sets of image coordinates using ransac.
    
    Inlier criteria is the reprojection error of y into image 1.

    Parameters
    -------------------------
    x : array 2xN image coordinates in image 1
    y : array 2xN image coordinates in image 2
    camera : Camera model
    threshold : float pixel distance threshold to accept as inlier
    do_translation : bool Try to estimate the translation as well

    Returns
    ------------------------
    R : array 3x3 The rotation that best fulfills X = RY
    t : array 3x1 translation if do_translation is False
    residual : array pixel distances ||x - xhat|| where xhat ~ KRY (and lens distorsion)
    inliers : array Indices of the points (in X and Y) that are RANSAC inliers
    """
    assert x.shape == y.shape
    assert x.shape[0] == 2
    
    X = camera.unproject(x)
    Y = camera.unproject(y)
    
    data = np.vstack((X, Y, x))
    assert data.shape[0] == 8
    
    model_func = lambda data: procrustes(data[:3], data[3:6], remove_mean=do_translation)
    
    def eval_func(model, data):
        Y = data[3:6].reshape(3,-1)
        x = data[6:].reshape(2,-1)
        R, t = model

        Xhat = np.dot(R, Y) if t is None else np.dot(R, Y) + t
        xhat = camera.project(Xhat)
        dist = np.sqrt(np.sum((x-xhat)**2, axis=0))

        return dist
    
    inlier_selection_prob = 0.99999
    model_points = 2
    ransac_iterations = int(np.log(1 - inlier_selection_prob) / np.log(1-inlier_ratio**model_points))
    
    model_est, ransac_consensus_idx = ransac.RANSAC(model_func, eval_func, data, model_points, ransac_iterations, threshold, recalculate=True)    
    if model_est is not None:
        (R, t) = model_est
        dist = eval_func((R, t), data)                
    else:
        dist = None
        R, t = None, None
        ransac_consensus_idx = []

    return R, t, dist, ransac_consensus_idx
