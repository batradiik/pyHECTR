import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from numpy.linalg import norm
import scipy
from itertools import product
import cv2 as cv

 
def ROI_to_angle(ROI, x0, pix_size, SDD):
    """
    Convert a one-dimensional detector ROI into detector angles.
    ``angle = arctan(-(pixel - x0) * pix_size / SDD)``.

    Parameters
    ----------
    ROI : slice
        Pixel-index interval. The function uses ``ROI.start`` and ``ROI.stop``
        and assumes a step of one pixel.
    x0 : float
        Direct-beam center position in pixels along the selected detector axis.
    pix_size : float
        Detector pixel size, in the same length unit as `SDD`.
    SDD : float
        Sample-detector distance, in the same length unit as `pix_size`.

    Returns
    -------
    angle : ndarray
        Detector angles in degrees for all pixels in the ROI.
    """
    x_= np.linspace(ROI.start, ROI.stop-1, ROI.stop - ROI.start)
    angle=np.float32(np.rad2deg(np.arctan(-(x_- x0)*pix_size/SDD)))
    return angle


def pixel_to_angle(pixel, center, pix_size, SDD):
    """
    Convert detector pixel coordinates to detector angles.
    The sign convention is ``angle = arctan(-(pixel - center) * pix_size / SDD)``.

    Parameters
    ----------
    pixel : float or array-like
        Pixel coordinate or coordinates.
    center : float
        Direct-beam center position in pixels.
    pix_size : float
        Detector pixel size, in the same length unit as `SDD`.
    SDD : float
        Sample-detector distance, in the same length unit as `pix_size`.

    Returns
    -------
    angle : float or ndarray
        Detector angle in degrees.
    """
    pixel = np.asarray(pixel, dtype=np.float64)
    return np.rad2deg(np.arctan(-(pixel - center) * pix_size / SDD))


def Q_grid2(ROIx, ROIy, pix_size, SDD, Lambda, x0, y0, incidence_ang):
    """
    Build a reciprocal-space interpolation grid for a detector ROI.

    The function maps detector pixels to grazing-incidence reciprocal-space
    coordinates and constructs a regular ``(q_r, q_z)`` grid. It also computes
    the corresponding detector-coordinate lookup arrays used by
    ``scipy.interpolate.interpn``.

    Parameters
    ----------
    ROIx : slice
        Detector x-pixel ROI.
    ROIy : slice
        Detector y-pixel ROI.
    pix_size : float
        Detector pixel size, in the same length unit as `SDD`.
    SDD : float
        Sample-detector distance, in the same length unit as `pix_size`.
    Lambda : float
        X-ray wavelength.
    x0, y0 : float
        Direct-beam center in detector pixel coordinates.
    incidence_ang : float
        Grazing-incidence angle in degrees.

    Returns
    -------
    x : ndarray
        Detector x-axis coordinates relative to `x0`, in pixels.
    y : ndarray
        Detector y-axis coordinates relative to `y0`, in pixels.
    RR_r : ndarray
        Detector x-coordinate lookup array for interpolation onto the q-grid.
    RR_z : ndarray
        Detector y-coordinate lookup array for interpolation onto the q-grid.
    q_r : ndarray
        Regular in-plane reciprocal-space axis.
    q_z : ndarray
        Regular out-of-plane reciprocal-space axis.
    """
    
    ai = np.deg2rad(incidence_ang)
    if Lambda <= 0:
        raise ValueError("Lambda must be positive.")
    if SDD <= 0:
        raise ValueError("SDD must be positive.")
    if pix_size <= 0:
        raise ValueError("pix_size must be positive.")
    if np.isclose(np.cos(ai), 0.0):
        raise ValueError("incidence_ang leads to cos(alpha_i) ≈ 0.")

    k = 2 * np.pi / Lambda

    # detector coordinates in pixel units relative to direct beam
    x = np.arange(ROIx.start, ROIx.stop, dtype=float) - x0
    y = y0 - np.arange(ROIy.start, ROIy.stop, dtype=float)
    X, Y = np.meshgrid(x, y)

    # detector coordinates in real units
    Xr = X * pix_size
    Yr = Y * pix_size

    # exact flat-detector ray direction (no small-angle approximation)
    norm = np.sqrt(SDD**2 + Xr**2 + Yr**2)
    sx = SDD / norm
    sy = Xr / norm
    sz = Yr / norm

    # exact q in laboratory frame: q = kf - ki, with ki = (k, 0, 0)
    qx_lab = k * (sx - 1.0)
    qy_lab = k * sy
    qz_lab = k * sz

    # rotate lab -> sample frame by incidence angle around y
    c = np.cos(ai)
    s = np.sin(ai)

    qx_s = c * qx_lab + s * qz_lab
    qy_s = qy_lab
    qz_s = -s * qx_lab + c * qz_lab

    # GI reciprocal-space coordinates used for the output map
    q_r_map = np.sign(qy_s) * np.sqrt(qx_s**2 + qy_s**2)
    q_z_map = qz_s

    # exact bounds from the actual pixelwise q-map
    q_r_min = np.nanmin(q_r_map)
    q_r_max = np.nanmax(q_r_map)
    q_z_min = np.nanmin(q_z_map)
    q_z_max = np.nanmax(q_z_map)

    row0 = np.argmin(np.abs(y))
    col0 = np.argmin(np.abs(x))

    dqr = np.abs(np.diff(q_r_map[row0, :]))
    dqz = np.abs(np.diff(q_z_map[:, col0]))

    dqr = np.nanmedian(dqr[dqr > 0]) if np.any(dqr > 0) else np.nan
    dqz = np.nanmedian(dqz[dqz > 0]) if np.any(dqz > 0) else np.nan

    if not np.isfinite(dqr):
        tmp = np.abs(np.diff(q_r_map, axis=1))
        dqr = np.nanmedian(tmp[tmp > 0])

    if not np.isfinite(dqz):
        tmp = np.abs(np.diff(q_z_map, axis=0))
        dqz = np.nanmedian(tmp[tmp > 0])

    dq = min(dqr, dqz)

    q_r = np.arange(np.floor(q_r_min / dq) * dq,
                    np.ceil(q_r_max / dq) * dq + 0.5 * dq, dq)
    q_z = np.arange(np.floor(q_z_min / dq) * dq,
                    np.ceil(q_z_max / dq) * dq + 0.5 * dq, dq)

    Q_r, Q_z = np.meshgrid(q_r, q_z)

    # exact inversion from (q_r, q_z) -> detector coordinates
    # Ewald-sphere constraint in sample frame:
    # (qx + k cos(ai))^2 + qy^2 + (qz - k sin(ai))^2 = k^2
    q_abs2 = Q_r**2 + Q_z**2
    qx_s_tgt = (Q_z * np.sin(ai) - q_abs2 / (2.0 * k)) / np.cos(ai)

    qy2 = Q_r**2 - qx_s_tgt**2
    valid = qy2 >= -1e-12
    # qy_s_tgt = np.sign(Q_r) * np.sqrt(np.clip(qy2, 0.0, None))
    # qy_s_tgt = np.sign(Q_r) * np.sqrt(qy2)
    # qy_s_tgt = np.zeros_like(Qpar)
    # qy_s_tgt[valid] = np.sign(Qpar[valid]) * np.sqrt(qy2[valid])

    qy_s_tgt = np.zeros_like(Q_r)
    qy_s_tgt[valid] = np.sign(Q_r[valid]) * np.sqrt(qy2[valid])


    # sample -> lab
    qx_lab_tgt = c * qx_s_tgt - s * Q_z
    qy_lab_tgt = qy_s_tgt
    qz_lab_tgt = s * qx_s_tgt + c * Q_z

    # kf_lab = ki_lab + q_lab, with ki_lab = (k, 0, 0)
    kf_x = k + qx_lab_tgt
    kf_y = qy_lab_tgt
    kf_z = qz_lab_tgt

    valid &= np.isfinite(kf_x) & (kf_x > 0)

    # initialize outside ROI so interpn(..., bounds_error=False, fill_value=0) gives 0
    RR_r = np.full_like(Q_r, x.min() - 1e9, dtype=float)
    RR_z = np.full_like(Q_z, y.min() - 1e9, dtype=float)

    RR_r[valid] = (SDD / pix_size) * (kf_y[valid] / kf_x[valid])
    RR_z[valid] = (SDD / pix_size) * (kf_z[valid] / kf_x[valid])
    # RR_r = (SDD / pix_size) * (kf_y / kf_x)
    # RR_z = (SDD / pix_size) * (kf_z / kf_x)

    return x, y, RR_r, RR_z, q_r, q_z

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x) * 180 / np.pi
    return(rho, phi)
    

def pol2cart(rho, phi_deg):
    phi = np.deg2rad(phi_deg)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def reciprocal_vec(x,y, vol):
    return np.cross(x, y)/vol
    
def triple_product(a, b, c):
    #return(np.einsum('ij, ij->i', a, np.cross(b, c)))
    return np.dot(a, np.cross(b,c))

def cos_angle_(v, w): return v.dot(w)/(norm(v)*norm(w))


def set_reciprocal_cell_5(a, b, c, transformation=None, transformation_flag=False):
    """
    Construct a reciprocal-lattice B matrix from direct lattice vectors.

    Parameters
    ----------
    a, b, c : array-like of shape (3,)
        Direct lattice vectors.
    transformation : ndarray of shape (3, 3), optional
        Optional transformation applied to the reciprocal basis vectors when
        `transformation_flag` is True.
    transformation_flag : bool, default True
        If True, apply `transformation` to the reciprocal basis before
        constructing the final B matrix.

    Returns
    -------
    bMatrix : ndarray of shape (3, 3)
        Reciprocal-lattice B matrix.
    """
    # Calculate the reciprocal lattice parameters
    pi = np.pi

    volume = triple_product(a, b, c)
    a_star = 2*pi*reciprocal_vec(b, c, volume)
    b_star = 2*pi*reciprocal_vec(c, a, volume)
    c_star = 2*pi*reciprocal_vec(a, b, volume)

    if transformation_flag and transformation is None:
        raise ValueError("Transformation must be provided when transformation_flag=True.")
    if transformation_flag: 
        star_matrix = transformation @ np.array([a_star, b_star, c_star])
        a_star, b_star, c_star  = star_matrix[0], star_matrix[1], star_matrix[2] 
        volume_star = triple_product(a_star, b_star, c_star)
        c = 2*pi*reciprocal_vec(a_star, b_star, volume_star)
        b = 2*pi*reciprocal_vec(c_star, a_star, volume_star)
        a = 2*pi*reciprocal_vec(b_star, c_star, volume_star)
    
    beta_star = np.arccos(cos_angle_(c_star, a_star))
    gamma_star = np.arccos(cos_angle_(a_star, b_star))
    alpha_star = np.arccos(cos_angle_(b_star, c_star))
    cos_alpha = cos_angle_(b, c)


    a_star = norm(a_star)
    b_star = norm(b_star)
    c_star = norm(c_star)
    c = norm(c)
    b = norm(b)

    # Calculate the BMatrix from the direct and reciprical parameters.
    # Reference: Busang and Levy (1967)
    bMatrix =np.array([
        [a_star, b_star * np.cos(gamma_star), c_star * np.cos(beta_star)],
        [0.0, b_star * np.sin(gamma_star), -c_star * np.sin(beta_star) * cos_alpha],
        [0.0, 0.0, 2*pi/c]])

    bMatrix = np.array([np.where(abs(b)>1e-9, b, 0) for b in bMatrix])
    return bMatrix



def UBinv(B, phi0, chi0, mu0):    
    """
    Compute the inverse UB matrix for a rotated crystal orientation.
    The rotation convention used by the current implementation is ``Rz(phi0) @ Ry(chi0) @ Rx(mu0)``.

    Parameters
    ----------
    B : ndarray of shape (3, 3)
        Reciprocal-lattice B matrix.
    phi0 : float
        Rotation angle around the laboratory z-axis, in degrees.
    chi0 : float
        Rotation angle around the laboratory y-axis, in degrees.
    mu0 : float
        Rotation angle around the laboratory x-axis, in degrees.

    Returns
    -------
    UB_inv : ndarray of shape (3, 3)
        Inverse of the rotated UB matrix.
    """
    
    rx = R.from_euler('x', mu0, degrees=True) 
    ry = R.from_euler('y', chi0, degrees=True) #y-axis along the x-ray beam
    rz = R.from_euler('z', phi0, degrees=True) #z-axis vertical
    rot = rz.as_matrix().dot(ry.as_matrix()).dot(rx.as_matrix())
    UB = rot.dot(B)
    UBinv = np.linalg.inv(UB)
    
    return UBinv




def hkl_calc(delta_, gamma_, theta_, UB_inv, Lambda):
    """
    Calculate HKL maps for detector angle grids and sample rotation angles.

    Parameters
    ----------
    delta_ : array-like of shape (n_delta,)
        Detector horizontal angles in degrees.
    gamma_ : array-like of shape (n_gamma,)
        Detector vertical angles in degrees.
    theta_ : array-like of shape (n_theta,)
        Sample rotation angles in degrees.
    UB_inv : ndarray of shape (3, 3)
        Inverse UB matrix.
    Lambda : float
        X-ray wavelength.

    Returns
    -------
    h, k, l : ndarray
        Arrays of shape ``(n_theta, n_gamma, n_delta)`` containing calculated
        fractional HKL coordinates.
    """
    #preparing zero matrices for faster calculation   

    A = np.zeros((3, 3), float)
    np.fill_diagonal(A, 1)

    res = np.empty((3, np.shape(gamma_)[0], np.shape(delta_)[0]), dtype='float32')
    hkl = np.empty((3, np.shape(gamma_)[0], np.shape(delta_)[0]), dtype='float32')

    h = np.empty((np.shape(theta_)[0], np.shape(gamma_)[0], np.shape(delta_)[0]), dtype='float32')
    k = np.empty((np.shape(theta_)[0], np.shape(gamma_)[0], np.shape(delta_)[0]), dtype='float32')
    l = np.empty((np.shape(theta_)[0], np.shape(gamma_)[0], np.shape(delta_)[0]), dtype='float32')

    Rgamma = np.empty(( np.shape(gamma_)[0],3,3), dtype='float32')
    Rdelta = np.empty(( np.shape(delta_)[0],3,3), dtype='float32')
    Rzthetainv = np.empty(( np.shape(theta_)[0],3,3), dtype='float32')

    rot2 = np.empty(( np.shape(gamma_)[0],np.shape(delta_)[0],3,3), dtype='float32') 

    # calculating k from Lambda
    k_vect = 2*math.pi/Lambda

    '''
    Calculating rotation matrices for each direction
    delta - horizontal detector pixel
    gamma - vertical detector pixel
    '''
    for ii in range (0, np.shape(gamma_)[0], 1):
        Rgamma[ii][:][:] = R.from_euler('x', gamma_[ii], degrees=True).as_matrix()

    for ii in range (0, np.shape(delta_)[0], 1):
        Rdelta[ii][:][:] = R.from_euler('z', delta_[ii], degrees=True).as_matrix()

    for ii in range (0,np.shape(theta_)[0], 1):   
        Rztheta = R.from_euler('z', theta_[ii], degrees=True)
        Rzthetainv[ii][:][:] = np.linalg.inv(Rztheta.as_matrix())

    #multilying rotation matrices for delta and gamma angle    
    for  jj in range (0,np.shape(delta_)[0], 1):
        for kk in range (0,np.shape(gamma_)[0], 1):
            rot2[kk][jj][:][:] = (Rdelta[jj][:][:].dot(Rgamma[kk][:][:])-A)   
    

    for ii in range (0, np.shape(theta_)[0]):

        #print(ii)
        #calculated in assumption of zeros in the Rzthetainv and k_in in form of [0,k_vect,0], see commented loops 
        res[0,:,:] = (Rzthetainv[ii,0,0]*rot2[:,:,0,1] + Rzthetainv[ii,0,1]*rot2[:,:,1,1])*k_vect
        res[1,:,:] = (Rzthetainv[ii,0,0]*rot2[:,:,1,1] - Rzthetainv[ii,0,1]*rot2[:,:,0,1])*k_vect
        res[2,:,:] = (rot2[:,:,2,1])*k_vect    

        hkl[0,:,:] = UB_inv[0][0]*res[0,:,:]+UB_inv[0][1]*res[1,:,:]+UB_inv[0][2]*res[2,:,:]
        hkl[1,:,:] = UB_inv[1][0]*res[0,:,:]+UB_inv[1][1]*res[1,:,:]+UB_inv[1][2]*res[2,:,:]
        hkl[2,:,:] = UB_inv[2][0]*res[0,:,:]+UB_inv[2][1]*res[1,:,:]+UB_inv[2][2]*res[2,:,:]

        h[ii][:][:] = (hkl[0,:,:])
        k[ii][:][:] = (hkl[1,:,:])
        l[ii][:][:] = (hkl[2,:,:])

        '''    
        for  jj in range (0,np.shape(delta_)[0], 1):
             for kk in range (0,np.shape(gamma_)[0], 1):
                mat=Rzthetainv[ii][:][:].dot(rot2[jj][kk])
                hkl=UB_inv.dot(mat).dot([0, k_vect, 0])

                h[ii][kk][jj]=(hkl[0])
                k[ii][kk][jj]=(hkl[1])
                l[ii][kk][jj]=(hkl[2])
    '''    
   # h = h[::-1]    
   # k = k[::-1]
   # l = l[::-1]
    
    return h,k,l



def find_CTR(INT, axNum, height, dist, med_kernel, flag='median'):
    """
    Detect candidate crystal truncation rod positions from an intensity volume.

    The function collapses an intensity array by taking a maximum projection
    over one axis and then averaging over the first remaining axis. A smoothed
    background is subtracted before detecting peaks with
    ``scipy.signal.find_peaks``.

    Parameters
    ----------
    INT : ndarray
        Input intensity array.
    axNum : int
        Axis over which to take the maximum projection.
    height : float
        Minimum peak height passed to ``scipy.signal.find_peaks``.
    dist : int
        Minimum peak distance passed to ``scipy.signal.find_peaks``.
    med_kernel : int
        Kernel size used for median or Wiener filtering.
    flag : {'median', 'wiener'}, default 'median'
        Background smoothing method.

    Returns
    -------
    peaks : ndarray
        Detected peak indices after local refinement.
    """
    A = np.nanmax(INT, axis=axNum)
    B = np.nanmean(A, axis=0)
    if flag == 'median':
        B = B-(scipy.signal.medfilt(B, med_kernel))
    if flag == 'wiener':
        B = B-(scipy.signal.wiener(B, med_kernel))
    peaks, _ = scipy.signal.find_peaks(B, height=height, distance=dist)

    for i in range (0, np.shape(peaks)[0]):
        window = 5
        arr = B[peaks[i] - window: peaks[i] + window]
        peaks[i] =  - window + peaks[i] + np.where(arr == np.amax(arr))
    return peaks



def mask_generator(h, k, h_peak=None, k_peak=None, threshold=0.009, flag='squared'):
    """
    Generate a binary mask around a target position in HK space.

    Parameters
    ----------
    h, k : ndarray
        HK coordinate arrays with matching shapes.
    h_peak, k_peak : float, optional
        Target HK coordinates. If one coordinate is None, the corresponding
        target is treated as zero in the current implementation.
    threshold : float, default 0.009
        Distance threshold for including pixels in the mask.
    flag : {'squared', 'abs'}, default 'squared'
        Distance mode. ``'squared'`` uses Euclidean distance in HK space.
        ``'abs'`` uses the current absolute-difference expression.

    Returns
    -------
    Mask : ndarray
        Float mask with values 1 inside the threshold and 0 outside.
    """
    if flag == 'squared':
        if (k_peak != None) and (h_peak != None):
            A = np.sqrt((h-h_peak)**2 + (k-k_peak)**2)
            Mask = np.ones(np.shape(A))
            Mask[abs(A) > threshold]=0
            return Mask
        elif (k_peak == None) and (h_peak != None):
            A = np.sqrt((h-h_peak)**2 + (k)**2)
            Mask = np.ones(np.shape(A))
            Mask[abs(A) > threshold]=0
            return Mask
        elif (k_peak != None) and (h_peak == None):
            A = np.sqrt((h)**2 + (k-k_peak)**2)
            Mask = np.ones(np.shape(A))
            Mask[abs(A) > threshold]=0
            return Mask
    if flag == 'abs':
        A = np.sqrt(abs(h-h_peak) + abs(k-k_peak))
        Mask = np.ones(np.shape(A))
        Mask[abs(A) > threshold]=0
        return Mask


def build_centers(h_peak, k_peak):
    h_arr = np.asarray(h_peak, dtype=float)
    k_arr = np.asarray(k_peak, dtype=float)

    if h_arr.size and k_arr.size:
        return np.array(list(product(h_arr, k_arr)), dtype=float)

    if h_arr.size:
        return np.column_stack((h_arr, np.zeros_like(h_arr)))

    if k_arr.size:
        return np.column_stack((np.zeros_like(k_arr), k_arr))

    return np.empty((0, 2), dtype=float)


def make_mask_fast(h, k, h_peak, k_peak, threshold=0.028):
    """
    Generate a boolean HK mask around one or more target peak centers.

    Parameters
    ----------
    h, k : ndarray
        HK coordinate arrays with matching shapes.
    h_peak, k_peak : float or array-like
        Target HK coordinates. If both contain multiple values, all Cartesian
        combinations are used as candidate centers.
    threshold : float, default 0.028
        Euclidean distance threshold in HK units.

    Returns
    -------
    mask : ndarray of bool
        Boolean mask selecting pixels within `threshold` of any target center.

    Raises
    ------
    ValueError
        If no valid peak centers are provided.
    """
    centers = build_centers(h_peak, k_peak)
    if centers.size == 0:
        raise ValueError("Peaks not found")

    mask = np.zeros(h.shape, dtype=bool)
    thr2 = threshold * threshold

    for hp, kp in centers:
        np.logical_or(mask, (h - hp)**2 + (k - kp)**2 <= thr2, out=mask)

    return mask




def generate_halo(mask, expand_koef=0.005):
    """
    Generate an outer halo around a binary mask.

    Parameters
    ----------
    mask : ndarray
        Input binary mask. Nonzero values are treated as mask pixels.
    expand_koef : float, default 0.00005
        Expansion size as a fraction of the larger image dimension.

    Returns
    -------
    halo_mask : ndarray of float32
        Mask containing the added halo region.
    """
    # Convert the mask to a binary mask
    mask_binary = np.uint8(mask)

    # Find contours of the mask
    contours, _ = cv.findContours(mask_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty mask for the halo
    halo_mask = np.zeros_like(mask, dtype=np.float32)

    # Iterate over the contours and draw them on the halo mask
    cv.drawContours(halo_mask, contours, -1, 1, thickness=cv.FILLED)

    # Expand the mask by 1%
    # kernel_size = int(np.round(expand_koef * max(mask.shape[:2])))
    kernel_size = max(1, int(np.round(expand_koef * max(mask.shape[:2]))))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    halo_mask = cv.dilate(halo_mask, kernel)

    # Subtract the original mask from the halo mask
    halo_mask = np.subtract(halo_mask, mask_binary)

    return halo_mask.astype(np.float32)




def generate_dilate(mask,
                    expand=0.00005,
                    metric='percent',
                    struct='rect',
                    return_mode='dilated'):
    """
    Dilate a binary mask and choose which output to return.

    Parameters
    ----------
    mask : ndarray
        Binary mask. Nonzero values are treated as object pixels.
    expand : float or int, default 0.00005
        Dilation size. If ``metric='percent'``, interpreted as a fraction of
        the larger image dimension. If ``metric='pixels'``, interpreted as an
        absolute number of pixels.
    metric : {'percent', 'pixels'}, default 'percent'
        Unit system for `expand`.
    struct : {'rect', 'ellipse', 'cross'}, default 'rect'
        Shape of the OpenCV structuring element.
    return_mode : {'dilated', 'ring', 'both'}, default 'dilated'
        Select what to return.
        - ``'dilated'``: return the full dilated mask.
        - ``'ring'``: return only the newly added halo/ring.
        - ``'both'``: return ``(dilated, ring)``.

    Returns
    -------
    dilated : ndarray of float32
        Full dilated mask, returned when ``return_mode='dilated'``.
    ring : ndarray of float32
        Added halo region only, returned when ``return_mode='ring'``.
    dilated, ring : tuple of ndarray
        Returned when ``return_mode='both'``.

    Raises
    ------
    ValueError
        If `metric`, `struct`, or `return_mode` is unsupported.
    """
    mask_bin = (mask > 0).astype(np.uint8)

    if metric == 'percent':
        k = int(round(expand * max(mask.shape[:2])))
    elif metric == 'pixels':
        k = int(round(expand))
    else:
        raise ValueError("metric must be 'percent' or 'pixels'.")

    k = max(1, k)

    kernel_shapes = {
        'rect': cv.MORPH_RECT,
        'ellipse': cv.MORPH_ELLIPSE,
        'cross': cv.MORPH_CROSS,
    }

    if struct not in kernel_shapes:
        raise ValueError("struct must be 'rect', 'ellipse', or 'cross'.")

    kernel = cv.getStructuringElement(kernel_shapes[struct], (k, k))

    dilated = cv.dilate(mask_bin, kernel)
    ring = dilated - mask_bin

    dilated = dilated.astype(np.float32)
    ring = ring.astype(np.float32)

    if return_mode == 'dilated':
        return dilated
    if return_mode == 'ring':
        return ring
    if return_mode == 'both':
        return dilated, ring

    raise ValueError("return_mode must be 'dilated', 'ring', or 'both'.")


def grow_or_shrink(mask,
                   expand=0.00005,
                   metric='percent',
                   struct='rect',
                   return_ring=False):
    """
    Dilate or erode a binary mask.

    Parameters
    ----------
    mask : ndarray of shape (height, width)
        Binary mask. Nonzero values are treated as object pixels.
    expand : float or int, default 0.00005
        Expansion size. Positive values grow the mask. Negative values shrink
        the mask. If ``metric='percent'``, the absolute value is interpreted as
        a fraction of the larger image dimension. If ``metric='pixels'``, the
        absolute value is interpreted as a number of pixels.
    metric : {'percent', 'pixels'}, default 'percent'
        Unit system used for `expand`.
    struct : {'rect', 'ellipse', 'cross'}, default 'rect'
        Shape of the OpenCV structuring element.
    return_ring : bool, default False
        If True, return only the added outer rim for dilation or the removed
        inner rim for erosion. If False, return the full dilated or eroded mask.

    Returns
    -------
    out : ndarray of float32
        Dilated or eroded binary mask. If `return_ring` is True, only the
        changed rim is returned.

    Raises
    ------
    ValueError
        If `mask` is not two-dimensional, if `metric` is unsupported, or if
        `struct` is unsupported.
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D array.")

    mask_bin = (mask > 0).astype(np.uint8)

    if metric == 'percent':
        k = int(round(abs(expand) * max(mask.shape[:2])))
    elif metric == 'pixels':
        k = int(round(abs(expand)))
    else:
        raise ValueError("metric must be 'percent' or 'pixels'.")

    k = max(1, k)

    shapes = {
        'rect': cv.MORPH_RECT,
        'ellipse': cv.MORPH_ELLIPSE,
        'cross': cv.MORPH_CROSS,
    }

    if struct not in shapes:
        raise ValueError("struct must be 'rect', 'ellipse', or 'cross'.")

    kernel = cv.getStructuringElement(shapes[struct], (k, k))

    if expand > 0:
        changed = cv.dilate(mask_bin, kernel)
        rim = changed - mask_bin
    elif expand < 0:
        changed = cv.erode(mask_bin, kernel)
        rim = mask_bin - changed
    else:
        changed = mask_bin
        rim = np.zeros_like(mask_bin)

    return (rim if return_ring else changed).astype(np.float32)




def check_integration(data_im,
                      data_raw,
                      data_bckgd,
                      gamma_windows,
                      delta_windows,
                      omega_values,
                      omega_windows,
                      L_values,
                      fig_size=(10, 5),
                      vmin_l=None,
                      vmax_l=None,
                      animate=False,
                      fps=2,
                      save_path=None,
                      verbose=False):
    """
    Visualize integration windows and omega profiles frame by frame.

    The function displays a detector image with the current
    ``(gamma, delta)`` integration ROI and the corresponding raw/background
    omega profiles. It can be used interactively with a slider or saved as an
    animation.

    Parameters
    ----------
    data_im : ndarray of shape (n_frames, n_gamma, n_delta)
        Image stack used for visualization.
    data_raw : sequence
        Raw omega profiles, one per frame.
    data_bckgd : sequence
        Background profiles, one per frame.
    gamma_windows : sequence of tuple[int, int]
        Gamma integration windows ``(g0, g1)`` for each frame.
    delta_windows : tuple[int, int] or sequence of tuple[int, int]
        Delta integration window. Can be one constant window or one window per
        frame.
    omega_values : ndarray
        Omega-axis values.
    omega_windows : sequence of slice or array-like
        Indices selecting the omega window for each frame.
    L_values : ndarray
        L values indexed by gamma pixel.
    fig_size : tuple of float, default (10, 5)
        Figure size.
    vmin_l, vmax_l : float, optional
        Color limits for the image display.
    animate : bool, default False
        If True, create an animation instead of an interactive slider.
    fps : int, default 2
        Animation frames per second.
    save_path : str or path-like, optional
        Output path for animation. Must end with ``.gif`` or ``.mp4`` when
        saving.
    verbose : bool, default False
        If True, print diagnostic information about input lengths and animation
        saving.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    ax : dict of matplotlib.axes.Axes
        Axes dictionary with keys ``'im'`` and ``'raw'``.

    Raises
    ------
    ValueError
        If input stacks and window lists do not have matching lengths, or if
        `save_path` has an unsupported extension.
    """
    n_frames = len(data_im)

    if verbose:
        print(
            "Input lengths:",
            f"{len(data_raw) = }",
            f"{len(data_bckgd) = }",
            f"{len(gamma_windows) = }",
            f"{len(omega_windows) = }",
            f"{n_frames = }",
        )

    if not (
        len(data_raw)
        == len(data_bckgd)
        == len(gamma_windows)
        == len(omega_windows)
        == n_frames
    ):
        raise ValueError(
            "Stacks for image, profiles and window lists must have identical length."
        )

    # Delta window handling: constant vs per-frame.
    per_frame_delta = (
        isinstance(delta_windows, (list, tuple, np.ndarray))
        and len(delta_windows) == n_frames
        and isinstance(delta_windows[0], (list, tuple, np.ndarray))
    )

    if not per_frame_delta:
        delta_windows = [delta_windows] * n_frames

    # Initial omega axis and labels.
    omega0 = omega_values[omega_windows[0]]
    delta_omega0 = np.round(omega0[0] - omega0[-1], 4)

    g0, g1 = gamma_windows[0]
    g1_l = min(g1, len(L_values) - 1)
    L0 = np.round((L_values[g0] + L_values[g1_l]) / 2, 3)

    fig, ax = plt.subplot_mosaic(
        [['im', 'raw']],
        figsize=fig_size,
        gridspec_kw={'width_ratios': [1, 1]},
    )

    im_artist = ax['im'].imshow(
        data_im[0],
        vmin=vmin_l if vmin_l is not None else np.min(data_im),
        vmax=vmax_l if vmax_l is not None else np.max(data_im),
    )

    d0, d1 = delta_windows[0]
    rect = patches.Rectangle(
        (d0, g0),
        d1 - d0,
        g1 - g0,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
    )
    ax['im'].add_patch(rect)
    ax['im'].set_title('ROI on image')
    ax['im'].axis('off')

    raw_line, = ax['raw'].plot(omega0, data_raw[0], '*', label='raw')
    bck_line, = ax['raw'].plot(omega0, data_bckgd[0], 'v', color='y', label='background')

    ax['raw'].set_title(f'Raw int, L, Δω = ({L0}, {delta_omega0})')
    ax['raw'].set_xlabel('img/$\\omega$ window')
    ax['raw'].set_ylabel('$Int$')
    ax['raw'].legend()
    ax['raw'].relim()
    ax['raw'].autoscale_view()

    def _set_frame(i):
        if data_raw[i] is None or data_bckgd[i] is None:
            raw_line.set_data([], [])
            bck_line.set_data([], [])
            ax['raw'].set_title('no signal')
            ax['im'].set_title(f'ROI frame {i}/{n_frames - 1} (empty)')
            fig.canvas.draw_idle()
            return

        g0, g1 = gamma_windows[i]
        d0, d1 = delta_windows[i]

        rect.set_xy((d0, g0))
        rect.set_width(d1 - d0)
        rect.set_height(g1 - g0)

        im_artist.set_data(data_im[i])

        omega_win = omega_values[omega_windows[i]]
        raw_line.set_data(omega_win, data_raw[i])
        bck_line.set_data(omega_win, data_bckgd[i])

        ax['raw'].relim()
        ax['raw'].autoscale_view()

        delta_omega = np.round(omega_win[0] - omega_win[-1], 4)

        g1_l = min(g1, len(L_values) - 1)
        L_val = np.round((L_values[g0] + L_values[g1_l]) / 2, 3)

        ax['raw'].set_title(f'Raw int, L, Δω = ({L_val}, {delta_omega})')
        ax['im'].set_title(f'ROI frame {i}/{n_frames - 1}')

    if not animate:
        ax_slider = plt.axes(
            [0.1, 0.03, 0.8, 0.03],
            facecolor='lightgoldenrodyellow',
        )
        slider = Slider(
            ax_slider,
            'frame',
            0,
            n_frames - 1,
            valinit=0,
            valstep=1,
        )

        def update_slider(val):
            _set_frame(int(val))
            fig.canvas.draw_idle()

        slider.on_changed(update_slider)

    else:
        def _anim(i):
            _set_frame(i)
            return im_artist, rect, raw_line, bck_line

        anim = FuncAnimation(
            fig,
            _anim,
            frames=n_frames,
            interval=1000 // fps,
        )

        if save_path:
            ext = save_path.split('.')[-1].lower()
            if ext in ('gif', 'mp4'):
                if verbose:
                    print(f"Saving animation to {save_path} ...")
                anim.save(save_path, fps=fps, dpi=150)
                if verbose:
                    print("Animation saved.")
            else:
                raise ValueError("save_path must end in .gif or .mp4")
        elif verbose:
            print("Animation created but not saved because save_path is None.")

    plt.tight_layout()
    plt.show()

    return fig, ax


