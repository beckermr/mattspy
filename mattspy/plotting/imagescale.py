import numpy as np

def segbit_scale(seg):
    """
    take a map with int vals and rescale to be linear between 0 and 1
    
    Parameters
    ----------
    seg: array-like
        input image to scale
        
    Returns
    -------
    seg_scaled: array-like
        image scaled to 0 to 1 with same shape as input    
    """
    seg_new = seg.copy()
    seg_new = seg_new.astype(float)
    uvals = np.sort(np.unique(seg))
    mval = 1.0*(len(uvals)-1.0)
    if mval < 1.0:
        mval = 1.0
    for ind,uval in enumerate(uvals):
        q = np.where(seg == uval)
        seg_new[q] = float(ind)/mval
        
    return seg_new

def asinh_scale(im, nonlinear=0.075):
    """
    Scale the image using and asinh stretch
        
        I = image*f
        f = asinh(image/nonlinear)/(image/nonlinear)
    
    Output image f is clipped to [0,1].        

    Parameters
    ----------
    image:
        The image.
    nonlinear: keyword
        The non-linear scale.

    Returns
    -------
    imout: array-like
        scaled image, same shape as input image

    Authors
    -------
    Created: Erin Sheldon, BNL        
    """
    I=im.astype('f4')
    
    I *= (1./nonlinear)
    
    # make sure we don't divide by zero
    w=np.where(I <= 0)
    if w[0].size > 0:
        I[w] = 1. # value doesn't matter since images is zero
        
    f = np.arcsinh(I)/I
    
    imout = im*f
    
    imout.clip(0.0, 1.0, imout)
    
    return imout
