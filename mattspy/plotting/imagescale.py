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
    mval = 1.0 * (len(uvals) - 1.0)
    if mval < 1.0:
        mval = 1.0
    for ind, uval in enumerate(uvals):
        q = np.where(seg == uval)
        seg_new[q] = float(ind) / mval

    return seg_new


def asinh_scale(im, nonlinear=0.075):
    """Scale the image using an asinh stretch

        I = image*f
        f = asinh(image/nonlinear)/(image/nonlinear)

    Output image f is clipped to [0,1].

    Parameters
    ----------
    image: np.ndarray
        The image.
    nonlinear: float, optional
        The non-linear scale. Default is 0.075

    Returns
    -------
    imout: array-like
        scaled image, same shape as input image

    Authors
    -------
    Created: Erin Sheldon, BNL
    """
    img = im.astype("f4")

    img /= nonlinear
    msk = img <= 0
    if np.any(msk):
        img[msk] = 1.0

    img = im * np.arcsinh(img) / img

    img.clip(0.0, 1.0, img)

    return img
