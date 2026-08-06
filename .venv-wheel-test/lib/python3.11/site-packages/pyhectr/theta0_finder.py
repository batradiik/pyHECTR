import numpy as np


def pixels_to_hkl_pointwise(xpix, ypix, theta_deg, UBinv, Lambda, x0, y0, pix_size, SDD):
    """
    Convert detector pixel coordinates to point-wise HKL coordinates.

    The function converts detector pixel positions into detector angles
    ``delta`` and ``gamma``, calculates the corresponding scattering vector,
    rotates it into the sample frame using the omega/theta angle, and applies
    the inverse UB matrix to obtain fractional HKL coordinates.

    Parameters
    ----------
    xpix : float or array-like
        Detector x-pixel coordinate(s), corresponding to the delta direction.
    ypix : float or array-like
        Detector y-pixel coordinate(s), corresponding to the gamma direction.
    theta_deg : float or array-like
        Sample rotation angle in degrees. Must be broadcast-compatible with
        `xpix` and `ypix`.
    UBinv : array-like of shape (3, 3)
        Inverse UB matrix used to transform reciprocal-space vectors into HKL
        coordinates.
    Lambda : float
        X-ray wavelength. Must be positive.
    x0 : float
        Direct-beam x-pixel position.
    y0 : float
        Direct-beam y-pixel position.
    pix_size : float
        Detector pixel size, in the same length unit as `SDD`.
    SDD : float
        Sample-detector distance, in the same length unit as `pix_size`.
        Must be nonzero.

    Returns
    -------
    h : ndarray or scalar
        Calculated h coordinate.
    k : ndarray or scalar
        Calculated k coordinate.
    l : ndarray or scalar
        Calculated l coordinate.

    Raises
    ------
    ValueError
        If `Lambda` is not positive, if `SDD` is zero, or if `UBinv` does not
        have shape ``(3, 3)``.
    """
    if Lambda <= 0:
        raise ValueError("Lambda must be positive.")
    if SDD == 0:
        raise ValueError("SDD must be nonzero.")

    UBinv = np.asarray(UBinv, dtype=float)
    if UBinv.shape != (3, 3):
        raise ValueError("UBinv must have shape (3, 3).")

    xpix, ypix, theta_deg = np.broadcast_arrays(xpix, ypix, theta_deg)

    delta = np.arctan(-(xpix - x0) * pix_size / SDD)
    gamma = np.arctan(-(ypix - y0) * pix_size / SDD)
    theta = np.deg2rad(theta_deg)

    k0 = 2 * np.pi / Lambda

    cg, sg = np.cos(gamma), np.sin(gamma)
    cd, sd = np.cos(delta), np.sin(delta)
    ct, st = np.cos(theta), np.sin(theta)

    # Scattering vector in the laboratory frame:
    # q_lab = (R_delta @ R_gamma - I) @ [0, k0, 0]
    qx = -k0 * cg * sd
    qy =  k0 * (cg * cd - 1.0)
    qz =  k0 * sg

    # Rotate into sample frame by Rz(-theta).
    rx =  ct * qx + st * qy
    ry = -st * qx + ct * qy
    rz =  qz

    q = np.stack([rx, ry, rz], axis=0)

    # Safe point-wise matrix multiplication for scalar, 1-D, or N-D inputs.
    hkl = np.einsum("ij,j...->i...", UBinv, q)

    return hkl[0], hkl[1], hkl[2]