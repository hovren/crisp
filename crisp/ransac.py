from __future__ import division

import numpy as np

def RANSAC(model_func, eval_func, data, num_points, num_iter, threshold, recalculate=False):
    """Apply RANSAC.

    This RANSAC implementation will choose the best model based on the number of points in the consensus set. At evaluation time the model is created using num_points points. Then it will be recalculated using the points in the consensus set.

    Parameters
    ------------
    model_func: Takes a data parameter of size DxK where K is the number of points needed to construct the model and returns the model (Mx1 vector)
    eval_func: Takes a model parameter (Lx1) and one or more data points (DxC, C>=1) and calculates the score of the point(s) relative to the selected model
    data : array (DxN) where D is dimensionality and N number of samples
    """
    M = None
    max_consensus = 0
    all_idx = range(data.shape[1])
    final_consensus = []
    for k in xrange(num_iter):
        np.random.shuffle(all_idx)
        model_set = all_idx[:num_points]
        x = data[:, model_set]
        m = model_func(x)

        model_error = eval_func(m, data)
        assert model_error.ndim == 1
        assert model_error.size == data.shape[1]
        consensus_idx = np.flatnonzero(model_error < threshold)

        if len(consensus_idx) > max_consensus:
            M = m
            max_consensus = len(consensus_idx)
            final_consensus = consensus_idx            

    # Recalculate using current consensus set?
    if recalculate and len(final_consensus) > 0:
        final_consensus_set = data[:, final_consensus]
        M = model_func(final_consensus_set)

    return (M, final_consensus)
