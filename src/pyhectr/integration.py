import scipy
import numpy as np
import matplotlib.pyplot as plt
from .read_plot import max_pxl_im
from scipy.signal import medfilt
from scipy.sparse.linalg import spsolve
from scipy.stats import linregress
from scipy.sparse import diags
import warnings


def _als_baseline(y: np.ndarray,
                  lam: float = 1e5,
                  p: float = 0.01,
                  n_iter: int = 10) -> np.ndarray:
    """
    Asymmetric least-squares baseline estimate.

    Parameters
    ----------
    y : 1-D ndarray
        Input intensity profile.
    lam : float, default 1e5
        Smoothness parameter. Larger values produce smoother baselines.
    p : float, default 0.01
        Asymmetry parameter. Small values make positive peaks less influential.
    n_iter : int, default 10
        Number of reweighting iterations.

    Returns
    -------
    baseline : ndarray of shape (n_points,)
        Estimated baseline profile. For input arrays shorter than three
        points, a copy of the input is returned.
    """
    
    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("ALS baseline expects a 1-D array.")
    if y.size == 0:
        raise ValueError("ALS baseline expects a non-empty array.")
    if y.size < 3:
        return y.copy()
    if lam <= 0:
        raise ValueError("lam must be positive.")
    if not (0 < p < 1):
        raise ValueError("p must be between 0 and 1.")
    if not np.all(np.isfinite(y)):
        raise ValueError("ALS baseline does not accept NaN or infinite values.")

    n = y.size

    # Second-derivative finite-difference matrix.
    diagonals = [
        np.ones(n - 2),
        -2.0 * np.ones(n - 2),
        np.ones(n - 2),
    ]
    D = scipy.sparse.diags(
        diagonals,
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )

    weights = np.ones(n)
    smoothness = lam * (D.T @ D)

    for _ in range(n_iter):
        W = diags(weights, 0, shape=(n, n), format="csc")
        baseline = spsolve(W + smoothness, weights * y)

        # Points above the baseline are likely peaks -> small weight.
        # Points below the baseline constrain the background -> large weight.
        weights = p * (y > baseline) + (1.0 - p) * (y <= baseline)

    return baseline




def line_prof_bckg_subtr(intensities: np.ndarray,
                         flag: str = 'median',
                         *,
                         med_kernel: int = 51,
                         als_lambda: float = 1e5,
                         als_p: float = 0.01) -> np.ndarray:
    """
    Estimate the background profile of a one-dimensional rocking curve.

    Depending on `flag`, the background is estimated either by fitting a
    straight line through the lower-intensity part of the profile, by applying
    a median filter, or by using asymmetric least-squares baseline smoothing.

    Parameters
    ----------
    intensities : array-like of shape (n_points,)
        Raw rocking-curve intensity profile as a function of omega or image
        index.
    flag : {'median', 'mean', 'medfilt', 'als'}, default 'median'
        Background estimation method.

        ``'median'``
            Fit a straight line through points below or equal to the median
            intensity.

        ``'mean'``
            Fit a straight line through points below or equal to the mean
            intensity.

        ``'medfilt'``
            Return a median-filtered version of the profile.

        ``'als'``
            Return an asymmetric least-squares baseline.

    med_kernel : int, default 51
        Odd median-filter window length used when ``flag='medfilt'``. Must be
        a positive odd integer.
    als_lambda : float, default 1e5
        Smoothness parameter passed to the asymmetric least-squares baseline
        estimator when ``flag='als'``.
    als_p : float, default 0.01
        Asymmetry parameter passed to the asymmetric least-squares baseline
        estimator when ``flag='als'``.

    Returns
    -------
    background : ndarray of shape (n_points,)
        Estimated background profile with the same length as `intensities`.

    Raises
    ------
    ValueError
        If `med_kernel` is not a positive odd integer, if `intensities` is not
        one-dimensional or is empty, or if `flag` is not one of the supported
        methods.

    Notes
    -----
    For ``flag='median'`` and ``flag='mean'``, the function returns the fitted
    linear background, not the background-subtracted intensity.

    For ``flag='medfilt'`` and ``flag='als'``, the function returns the
    estimated baseline directly.
    """
    if med_kernel < 1:
        raise ValueError("med_kernel must be positive.")
    if med_kernel % 2 == 0:
        raise ValueError("med_kernel must be odd.")
        
    flag = flag.lower()
    y = np.asarray(intensities, dtype=float)
    
    if y.ndim != 1:
        raise ValueError("intensities must be a 1-D array.")
    if y.size == 0:
        raise ValueError("intensities must be non-empty.")
    # if not np.all(np.isfinite(y)):
    #     raise ValueError("intensities must not contain NaN or infinite values.")

    if flag == 'mean':
        m = np.mean(y)
        mask = y <= m

    elif flag == 'median':
        m = np.median(y)
        mask = y <= m

    elif flag == 'medfilt':
        # smooth with sliding-window median
        base = medfilt(y, kernel_size=med_kernel)
        return base

    elif flag == 'als':
        # asymmetric least-squares baseline
        base = _als_baseline(y, lam=als_lambda, p=als_p)
        return base

    else:
        raise ValueError("flag must be 'mean', 'median', 'medfilt', or 'als'")

    # for mean/median: robust straight-line fit through low half
    x = np.arange(y.size)
    res = linregress(x[mask], y[mask])
    return res.intercept + res.slope * x


# def signal_direction(y_pixels):
#     y_pixels_diff = np.diff(y_pixels)
#     if (y_pixels_diff>0).all() == True:
#         signal_direction_sign = -1
#     elif (y_pixels_diff<0).all() == True:
#         signal_direction_sign = 1
#     else:
#         raise ValueError('Set sign manually!')
#     return signal_direction_sign


def signal_direction(y_pixels):
    """
    The function checks whether the supplied detector y-pixel coordinates are
    strictly increasing or strictly decreasing and returns the corresponding
    sign used by the integration workflow.

    Parameters
    ----------
    y_pixels : array-like of shape (n_points,)
        Ordered sequence of detector y-pixel coordinates.

    Returns
    -------
    signal_direction_sign : int
        Direction sign inferred from the pixel ordering.

        - ``-1`` if `y_pixels` is strictly increasing.
        - ``1`` if `y_pixels` is strictly decreasing.

    Raises
    ------
    ValueError
        If fewer than two y-pixels are provided, or if the sequence is not
        strictly monotonic.
    """
    y_pixels = np.asarray(y_pixels)
    if y_pixels.size < 2:
        raise ValueError("At least two y-pixels are required.")
    y_pixels_diff = np.diff(y_pixels)
    if np.all(y_pixels_diff > 0):
        return -1
    if np.all(y_pixels_diff < 0):
        return 1
        
    raise ValueError(f"y_pixels must be strictly monotonic.\nSet sign manually.")


def get_signal_image_range(pixel_coord_plot):
    """
    Return and plot the image indices that contain signal-mask pixels.

    Parameters
    ----------
    pixel_coord_plot : array-like of shape (n_hits, 3)
        Signal pixel coordinates. Each row is expected to contain
        ``(omega_index, gamma_pixel, delta_pixel)``.

    Returns
    -------
    images_n : ndarray
        Sorted unique omega/image indices present in `pixel_coord_plot`.

    """
    images_n = np.unique(pixel_coord_plot[:, 0])
    print("Image range: ", images_n)
    plt.figure()
    plt.title("Distribution of signal per images range")
    plt.xlabel("Image range")
    plt.ylabel("Counts")
    plt.hist(images_n, bins='auto', edgecolor='black')
    plt.legend()
    plt.show()
    return images_n


def get_omega_range(
    pixel_coord_plot: np.ndarray,
    gamma_pxl_range: int,
    plot_flag: bool = True,
    sigma_factor: float = 2.0,
) -> int:
    """
    Estimate a suitable half-width (in ω-image indices) for the
    integration window by examining, for every γ-column, how far the
    ω-indices (images) spread.

    Parameters
    ----------
    pixel_coord_plot : (N_hits, 3) array
        Each row is (omega_index, gamma_pixel, delta_pixel).
    gamma_pxl_range : int
        Number of γ-pixels on the detector (max gamma_pixel + 1).
    plot_flag : bool, default True
        If True, draw histograms for visual inspection.
    sigma_factor : float, default 2.0
        Multiplier on the global σ to define a conservative window.

    Returns
    -------
    range_omega_val : int
        Recommended ω-half-window size (images).

    Raises
    ------
    ValueError
        If no γ-column contains more than one ω-hit.
    """

    # ---------- sanity checks -------------------------------------------
    if pixel_coord_plot.size == 0:
        raise ValueError("pixel_coord_plot is empty")

    # split the columns
    omegas  = pixel_coord_plot[:, 0].astype(int)
    gammas  = pixel_coord_plot[:, 1].astype(int)

    # gather per-γ ranges
    unique_gamma = np.unique(gammas)
    omega_ranges, omega_stds = [], []

    for g in unique_gamma:
        o_vals = omegas[gammas == g]
        if o_vals.size > 1:
            omega_ranges.append(o_vals.max() - o_vals.min())
            omega_stds.append(o_vals.std())

    if not omega_ranges:
        raise ValueError("No γ column has more than one ω-hit; "
                         "cannot compute a range.")

    r = np.asarray(omega_ranges, dtype=float)
    s = np.asarray(omega_stds,  dtype=float)

    # stats
    range_median, range_std  = np.median(r),  r.std()
    std_median,  std_std     = np.median(s),  s.std()

    # window: median range + σ·median std-within-column
    range_omega_val = int(np.ceil(range_median + sigma_factor * std_median))

    # ------------- optional plots ---------------------------------------
    if plot_flag:
        fig, axs = plt.subplot_mosaic([["range", "std"]], figsize=(12, 5))

        axs["range"].hist(r, bins="auto", edgecolor="black")
        axs["range"].axvline(range_median, color="r", ls="--",
                             label=f"median={range_median:.2f}")
        axs["range"].set_title("Per-γ ω-range (max–min)")
        axs["range"].set_xlabel("ω-range (images)")
        axs["range"].set_ylabel("count")
        axs["range"].legend()

        axs["std"].hist(s, bins="auto", edgecolor="black")
        axs["std"].axvline(std_median, color="r", ls="--",
                           label=f"median={std_median:.2f}")
        axs["std"].set_title("Per-γ ω-std")
        axs["std"].set_xlabel("σ(ω) in each γ-column")
        axs["std"].set_ylabel("count")
        axs["std"].legend()

        plt.tight_layout()
        print(f"median(ω-range) = {range_median:.2f}, "
              f"median σ(ω) = {std_median:.2f}")
        print(f"Recommended ω range = {range_omega_val} images")

    return range_omega_val


def get_delta_range(pixel_coord_plot, gamma_pxl_range, plot_flag = True, sigma_factor = 2.0):
    """
    Estimate a suitable half-width (in Δ-pixels) for integration window
    by looking at how, for each γ-column, the Δ-extent of 'hits' varies.

    Parameters
    ----------
    pixel_coord_plot : (N_hits, 3) array
        Each row is (omega_index, gamma_pixel, delta_pixel).
    gamma_pxl_range : int
        Number of γ-pixels on the detector (max gamma_pixel + 1).
    plot_flag : bool, default True
        If True, show a histogram of the per-γ Δ-ranges.
    sigma_factor : float, default 2.0
        How many standard deviations above the typical range to include.
    Returns
    -------
    half_range_delta_val : int
        Recommended half-width in delta pixels.
    range_delta_val : int
        Estimated full delta range in pixels.

    """

    # sanity check
    if pixel_coord_plot.size == 0:
        raise ValueError("pixel_coord_plot is empty")

    # extract gamma and delta columns
    gammas = pixel_coord_plot[:, 1].astype(int)
    deltas = pixel_coord_plot[:, 2].astype(int)

    unique_gamma = np.unique(gammas)
    # for each gamma, mask and compute Δ-range
    delta_ranges = []
    for g in unique_gamma:
        dvals = deltas[gammas == g]
        if dvals.size > 1:
            delta_ranges.append(dvals.max() - dvals.min())
    if not delta_ranges:
        raise ValueError("No gamma column has more than one Δ-hit; cannot compute a range.")

    dr = np.array(delta_ranges, dtype=float)
    std = dr.std()
    median   = np.median(dr)
    mean     = dr.mean()

    # half-window: median + sigma_factor * std
    range_delta_val = int(np.ceil(median + sigma_factor * std))
    # half_range_delta_val = int(np.ceil(median + sigma_factor * std)/2)
    half_range_delta_val = int(np.ceil(range_delta_val / 2))

    # optional plotting
    if plot_flag:
        plt.figure()
        plt.title("Distribution of per-γ Δ-range (max–min)")
        plt.hist(dr, bins='auto', edgecolor='black',)
        plt.xlabel("Δ-range (pixels)")
        plt.ylabel("Count of γ-bins")
        plt.axvline(median, color='red', linestyle='--', label=f"median={median:.2f}")
        plt.axvline(median + sigma_factor * std, color='orange', linestyle='-.', 
                    label=f"median+{sigma_factor}σ={median + sigma_factor * std:.2f}",)
                    # label=f"median+σ={median}+{std:.2f}")
        plt.legend()
        plt.show()
        print(f"std = {std:.3f}, median = {median:.3f}, mean = {mean:.3f}")
        print(f"Recommended half Δ-range = {half_range_delta_val} pixels;\n Δ-range = {range_delta_val} pixels")

    return half_range_delta_val, range_delta_val




def rocking_scan_integration(tmp1 , pixel_coord_plot, half_omega_r, half_delta_r,
                             bin_rate, gamma_edges, gamma_centres, 
                             FLAG = 'median', medfilt_kernel = 51, SHOW_PLOT = False, 
                             PRINT_INTEGRATION_INFO = False):
    intensities_summed     = []
    int_slider_raw         = []
    int_slider_subtracted  = []
    bckgd_slider           = []
    omega_windows          = []
    gamma_windows          = []
    delta_windows          = []

    """
    Integrate rocking-curve intensities around a moving gamma/delta ROI.

    For each consecutive pair of gamma bin edges, the function finds signal
    pixels whose gamma coordinate lies inside the half-open interval
    ``[g0, g1)``. 

    Parameters
    ----------
    tmp1 : ndarray of shape (n_omega, n_gamma, n_delta)
        Image stack or corrected intensity stack to integrate. Axis 0 is the
        omega/image axis, axis 1 is gamma pixels, and axis 2 is delta pixels.
    pixel_coord_plot : array-like of shape (n_hits, 3)
        Signal pixel coordinates. Each row is expected to contain
        ``(omega_index, gamma_pixel, delta_pixel)``.
    half_omega_r : int
        Half-size parameter for the omega integration window. The current code
        uses the slice ``n_image - half_omega_r : n_image + half_omega_r``.
    half_delta_r : int
        Half-size parameter for the delta integration window. The current code
        uses the slice ``delta_mid - half_delta_r : delta_mid + half_delta_r``.
    bin_rate : int or float
        Gamma binning parameter retained for API compatibility. It is not used
        directly by the current implementation.
    gamma_edges : array-like of shape (n_bins + 1,)
        Gamma bin edges. Consecutive pairs define half-open gamma intervals
        ``[g0, g1)``.
    gamma_centres : array-like of shape (n_bins,)
        Gamma bin centres retained for API compatibility. They are not used
        directly by the current implementation.
    FLAG : {'median', 'mean', 'medfilt', 'als'}, default 'median'
        Background estimation method passed to :func:`line_prof_bckg_subtr`.
    medfilt_kernel : int, default 51
        Median-filter kernel passed to :func:`line_prof_bckg_subtr` when
        ``FLAG='medfilt'``.
    SHOW_PLOT : bool, default False
        If True, plot each background-subtracted rocking curve.
    PRINT_INTEGRATION_INFO : bool, default False
        If True, print the estimated central image, delta pixel, and gamma
        bounds for each non-empty gamma bin.

    Returns
    -------
    intensities_summed : list
        Sum of each background-subtracted rocking curve. Gamma bins without
        signal hits are stored as ``None``.
    int_slider_raw : list
        Raw omega profiles before background subtraction. Empty gamma bins are
        stored as ``None``.
    int_slider_subtracted : list
        Background-subtracted omega profiles. Empty gamma bins are stored as
        ``None``.
    bckgd_slider : list
        Estimated background profiles. Empty gamma bins are stored as ``None``.
    omega_windows : list of slice
        Omega slices used for non-empty gamma bins.
    gamma_windows : list of tuple[int, int]
        Gamma windows ``(g0, g1)`` used for non-empty gamma bins.
    delta_windows : list of tuple[int, int]
        Delta windows ``(de0, de1)`` used for non-empty gamma bins.

    """

    n_omega, gamma_pxl_range, n_delta = tmp1.shape
    n_bins                            = len(gamma_centres)

    if SHOW_PLOT:
            plt.figure()
    for g0, g1 in zip(gamma_edges[:-1], gamma_edges[1:]):
        hits_mask = (pixel_coord_plot[:, 1] >= g0) & (pixel_coord_plot[:, 1] < g1)
        if not hits_mask.any():
            # pad with None
            for lst in (int_slider_raw, int_slider_subtracted,
                        bckgd_slider, intensities_summed):
                lst.append(None)
            continue

        # median Ω and mean Δ for this γ-band
        n_image   = int(np.median(pixel_coord_plot[hits_mask, 0]))
        delta_mid = int(np.mean  (pixel_coord_plot[hits_mask, 2]))
        if PRINT_INTEGRATION_INFO:
            print(f'{n_image = }\n{delta_mid = }\n{g0 =}, {g1 =}\n\n')

        # slice bounds with clipping (avoids negative indices) 
        om0 = max(0, n_image - half_omega_r)
        om1 = min(n_omega, n_image + half_omega_r)
        de0 = max(0, delta_mid - half_delta_r)
        de1 = min(n_delta, delta_mid + half_delta_r)
        if om0 == 0:
            # print(f'Warning: {om0} = 0 \n\n')
            warnings.warn(
                f"Omega window for gamma bin ({om0}) touches lower image boundary.",
                RuntimeWarning,
            )
        if de0 == 0:
            # print(f'Warning:{de0} = 0\n\n')
            warnings.warn(
                f"Omega window for gamma bin ({de0}) touches lower image boundary.",
                RuntimeWarning,
            )
        if om1 == n_omega:
            # print(f'Warning:{om1} = {n_omega}\n\n')
            warnings.warn(
                f"Warning:{om1} = {n_omega}\n\n",
                RuntimeWarning,
            )
        if de1 == n_delta:
            print(f'Warning:{de1} = {n_delta} 0\n\n')
            warnings.warn(
                f"Warning:{de1} = {n_delta}\n\n",
                RuntimeWarning,
            )

        # rocking curve I(Ω)
        # print('integrated window\n')
        # print(f'{om0 = }, {om1 =}, {g0 = }, {g1 = }, {de0 =}, {de1 =}')
        intensities = tmp1[om0:om1, g0:g1, de0:de1].sum(axis=(1, 2)) 
        # print(f'{intensities =}')
        # bckgd       = integration.line_prof_bckg_subtr(intensities,
        #                                    flag=FLAG,
        #                                    med_kernel=medfilt_kernel)
        bckgd       = line_prof_bckg_subtr(intensities,
                                   flag=FLAG,
                                   med_kernel=medfilt_kernel)
    
        # intensities_bckgd_subtr = np.clip(intensities - bckgd, 0, None)
        intensities_bckgd_subtr = intensities - bckgd
    
        # ------------- store in the containers --------------------------
        int_slider_raw.append(intensities)
        bckgd_slider.append(bckgd)
        int_slider_subtracted.append(intensities_bckgd_subtr)
        if SHOW_PLOT:
            plt.plot(intensities_bckgd_subtr, '*')
        intensities_summed.append(intensities_bckgd_subtr.sum())
        omega_windows.append(slice(om0, om1))
        gamma_windows.append((g0, g1))
        delta_windows.append((de0, de1))
        
    if SHOW_PLOT:
            plt.show()

    return intensities_summed, int_slider_raw, int_slider_subtracted, bckgd_slider, omega_windows, gamma_windows, delta_windows


def apply_corrections2D(  delta_arr              : np.ndarray,
                          gamma_arr              : np.ndarray,
                          incidence_ang          : float  = 0.03,
                          area                   : float  = 1.0,
                          rocking                : bool   = True,
                          return_map             : bool   = False,
                          pixel_coord_plot        = None,
                          T0                      = None,
                          flat_field              = None):
    """
    Per-pixel or per-gamma correction factors for grazing-incidence X-ray rocking scans.

    Parameters
    ----------
    pixel_coord_plot : (N_hits, 3) or None
        (omega_idx, gamma_px, delta_px).  If None, no Δ-centroid averaging is used.
    delta_arr        : (N_delta,)  δ-angle [deg] of every detector column
    gamma_arr        : (N_gamma,)  γ-angle [deg] of every detector row
    incidence_ang    : grazing incidence α_i  [deg]
    area             : illuminated-area / flat-field scale (constant)
    rocking          : True → φ-scan geometry, False → stationary detector scan
    return_map       : True  → returns (Cmap, Cgamma)
                       False → returns Cgamma only   (old behaviour)
    T0               : Be-window transmission at normal incidence; None → skip
    flat_field       : (N_gamma,N_delta) pixel gain map; None → unity

    Returns
    -------
    Cmap : ndarray of shape (n_gamma, n_delta)
        Full two-dimensional correction map. Returned only when
        ``return_map=True``. Non-finite and non-positive values are replaced
        by 1 before returning.
    Cgamma : ndarray of shape (n_gamma,)
        Gamma-averaged correction vector. Returned only when
        ``return_map=False``.
    Raises
    ------
    ValueError
        If `flat_field` is provided with a shape different from
        ``(n_gamma, n_delta)``.

    """
    
    d2r     = np.pi / 180.0
    alpha_i = incidence_ang * d2r

    # ----------- full γ×δ angle grids ------------------------------------
    gamma_rad = gamma_arr * d2r                       # (Nγ,)
    delta_rad = delta_arr * d2r                       # (Nδ,)
    gamma_grid, delta_grid = np.meshgrid(
        gamma_rad, delta_rad, indexing='ij'
    )                                                 # (Nγ,Nδ)

    # ----------- exit angle β_out per pixel ------------------------------
    sin_beta = (np.cos(alpha_i) * np.sin(delta_grid)
                - np.cos(gamma_grid) * np.cos(delta_grid) * np.sin(alpha_i))
    sin_beta = np.clip(sin_beta, -1.0, 1.0)              # numeric safety
    beta_rad = np.arcsin(sin_beta)
    sin_beta_nonzero = np.where(
        np.abs(sin_beta) < 1e-12, 1e-12, sin_beta
    )  # avoid /0 later

    # ----------- Lorentz × rod factor ------------------------------------
    if rocking:
        Lrod = (1.0 / (
            np.sin(gamma_grid) * np.cos(alpha_i) * np.cos(delta_grid)
        ) * np.cos(beta_rad))
    else:
        Lrod = 1.0 / sin_beta_nonzero                # stationary scan

    # ----------- Polarisation -------------------------------------------
    P = 1.0 - (np.sin(gamma_grid) * np.cos(delta_grid))**2

    # ----------- Transmission through Be window -------------------------
    if T0 is None:
        transm = 1.0
    else:
        transm = T0**(1.0 / np.cos(beta_rad) - 1.0)

    # ----------- Optional flat-field map --------------------------------
    if flat_field is None:
        FF = 1.0
    else:
        if flat_field.shape != gamma_grid.shape:
            raise ValueError("flat_field shape must be (N_gamma,N_delta)")
        FF = flat_field

    # ----------- Total per-pixel correction -----------------------------
    Cmap = (Lrod * P * transm * area) / FF       # (Nγ,Nδ)

    # ----------- γ-averaged vector (legacy output) ----------------------
    Cgamma = np.nanmean(Cmap, axis=1)

    # If pixel_coord_plot is given, refine Cgamma using Δ-centroids -------
    if pixel_coord_plot is not None:
        # digitise each hit into its γ-index
        gamma_idx = (
            np.digitize(
                pixel_coord_plot[:, 1],
                np.arange(len(gamma_arr))
            ) - 1
        )

        # build per-γ list of Δ indices, then median
        delta_idx_median = np.full(len(gamma_arr), np.nan)
        for g in np.unique(gamma_idx):
            hits = gamma_idx == g
            delta_idx_median[g] = np.median(pixel_coord_plot[hits, 2])

        valid = ~np.isnan(delta_idx_median)
        Cgamma[valid] = Cmap[
            valid,
            delta_idx_median[valid].astype(int)
        ]

    # ---------------- return in old or new style ------------------------
    if return_map:
        Cmap[~np.isfinite(Cmap) | (Cmap <= 0)] = 1
        return Cmap
    return Cgamma


def rod_points_prep(mask_arr, y_u=None, y_b=None, 
                    x_l=None, x_r=None, im_start=None, 
                    im_end=None, show=True):
    """
    The function applies optional image, gamma, and delta crops by setting
    pixels outside the requested ranges to zero. It then returns the coordinates
    of all remaining nonzero pixels and a two-dimensional maximum projection
    for quick visualization.

    Parameters
    ----------
    mask_arr : ndarray of shape (n_images, n_gamma, n_delta)
        Three-dimensional mask stack. Nonzero values are treated as selected
        rod pixels.
    y_u : int, optional
        Upper gamma crop boundary. Pixels with gamma index smaller than `y_u`
        are set to zero.
    y_b : int, optional
        Lower gamma crop boundary. Pixels with gamma index greater than or
        equal to `y_b` are set to zero.
    x_l : int, optional
        Left delta crop boundary. Pixels with delta index smaller than `x_l`
        are set to zero.
    x_r : int, optional
        Right delta crop boundary. Pixels with delta index greater than or
        equal to `x_r` are set to zero.
    im_start : int, optional
        First image index to keep. Images before `im_start` are set to zero.
    im_end : int, optional
        First image index to discard. Images from `im_end` onward are set to
        zero, so the kept interval is ``[im_start, im_end)``.
    show : bool, default True
        If True, display the two-dimensional maximum projection.

    Returns
    -------
    pixel_coord : ndarray of shape (n_hits, 3)
        Coordinates of nonzero pixels after cropping, ordered as
        ``(image_index, gamma_pixel, delta_pixel)``.
    mask_new : ndarray of shape (n_gamma, n_delta)
        Two-dimensional maximum projection of the cropped mask stack.
    """
    mask_ = mask_arr.copy()

    # --- apply crops (only if not None) ---
    if x_l is not None:
        mask_[:, :, :x_l] = 0
    if x_r is not None:
        mask_[:, :, x_r:] = 0

    if y_u is not None:
        mask_[:, :y_u, :] = 0
    if y_b is not None:
        mask_[:, y_b:, :] = 0

    if im_start is not None:
        mask_[:im_start, :, :] = 0
    if im_end is not None:
        mask_[im_end:, :, :] = 0  # keeps [im_start:im_end]

    # --- extract coordinates ---
    # pixel_coord = np.argwhere(mask_ == 1)
    pixel_coord = np.argwhere(mask_ > 0)

    # --- make 2D view ---
    try:
        mask_new = max_pxl_im(mask_)
    except NameError:
        mask_new = mask_.max(axis=0)

    if show:
        plt.figure()
        plt.imshow(mask_new, vmin=0, vmax=1)
        plt.show()

    del mask_
    return pixel_coord, mask_new


# def find_closest_value_index(arr, value):
#     """
#     Return the index of the finite array element closest to `value`.
#     """
#     arr = np.asarray(arr, dtype=float)

#     if arr.size == 0:
#         raise ValueError("arr must be non-empty.")
#     if not np.isfinite(value):
#         raise ValueError("value must be finite.")
#     if not np.any(np.isfinite(arr)):
#         raise ValueError("arr must contain at least one finite value.")

#     return int(np.nanargmin(np.abs(arr - value)))
    
def find_closest_value_index(arr, value):
    """
    Find the index of the element in the array `arr` that is closest to the given `value`.
    
    Parameters:
        arr (numpy.ndarray): The input array.
        value (float): The value to find the closest index to.
        
    Returns:
        int: The index of the element closest to the given value.
    """
    # Calculate the absolute differences between each element and the given value
    absolute_diff = np.abs(arr - value)
    
    # Find the index of the element with the minimum absolute difference
    closest_index = np.argmin(absolute_diff)
    
    return closest_index
