import numpy as np

__all__ =['mad']

def mad(x, axis=None, no_scale=False):
    """
    median absolute deviation - scaled like a standard deviation
    
        mad = 1.4826*median(|x-median(x)|)

    options:
        no_scale - set to True to turn off scaling (default=False)

    """

    mad = np.median(np.abs(x - np.median(x,axis=axis)),axis=axis)
    if no_scale:
        return mad
    else:
        return 1.4826*mad

