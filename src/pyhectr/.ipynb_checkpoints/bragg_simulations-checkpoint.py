import ast, math, time
import numpy as np
import pandas as pd
from xrayutilities import utilities
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm


def are_perpendicular(a, b, tolerance=1e-8):
    """
    Check whether two vectors are perpendicular within a numerical tolerance.

    Parameters
    ----------
    a, b : array-like
        Input vectors. They must be compatible with `numpy.dot`.
    tolerance : float, default 1e-8
        Absolute tolerance used in the perpendicularity check.

    Returns
    -------
    is_perpendicular : bool or ndarray of bool
        Result of checking whether the dot product is numerically close to
        zero. The exact return shape follows `numpy.isclose`.

    """
    dot_product = np.dot(a, b)
    return np.isclose(dot_product, np.zeros_like(dot_product, dtype=np.float64), atol=tolerance)

def are_collinear(a, b, tolerance=1e-8):
    """
    Check whether two vectors are collinear within a numerical tolerance.

    Parameters
    ----------
    a, b : array-like
        Input vectors. They must be compatible with `numpy.cross`.
    tolerance : float, default 1e-8
        Absolute tolerance used when checking whether the cross product is
        numerically close to zero.

    Returns
    -------
    is_collinear : bool
        True if the cross product is close to zero; otherwise False.
    """
    cross = np.cross(a, b)
    return np.allclose(cross, np.zeros_like(cross, dtype=np.float64), atol=tolerance)



def peak_dict_to_records(peak_data):
    """
    Convert a peak data dictionary into a list of row records.

    Parameters
    ----------
    peak_data : dict
        Dictionary containing peak information. The current implementation
        expects the following keys:

        - ``'x'`` : plotted x-coordinates.
        - ``'y'`` : plotted y-coordinates.
        - ``'hkl'`` : Miller indices for each peak.
        - ``'qvec'`` : reciprocal-space vectors for each peak.
        - ``'angles'`` : calculated experimental angles for each peak.

    Returns
    -------
    records : list of dict
        One dictionary per peak. Each record contains ``'x'``, ``'y'``,
        ``'theoretical_peak'``, ``'hkl'``, ``'qvec'``, and ``'angles'``.

    """
    records = []
    n = len(peak_data['x'])
    
    for i in range(n):
        records.append({
            'x': float(peak_data['x'][i]),
            'y': float(peak_data['y'][i]),
            'theoretical_peak': (float(peak_data['x'][i]), float(peak_data['y'][i])),
            'hkl': tuple(int(v) for v in peak_data['hkl'][i]),
            'qvec': tuple(float(v) for v in peak_data['qvec'][i]),
            'angles': tuple(float(v) for v in peak_data['angles'][i]),
        })
    return records



def filter_peaks_in_plot_range(peak_data, xlim=(-5,5), ylim=(-0.1,7.8)):
    """
    Keep only the peaks whose (x,y) = (qy, qz) are inside the bounding box.
    Filter peak-data arrays by plotted x/y limits.

    Parameters
    ----------
    peak_data : dict
        Dictionary containing at least ``'x'`` and ``'y'`` arrays. All values
        in the dictionary are converted to NumPy arrays and filtered with the
        same boolean mask.
    xlim : tuple of float, default (-5, 5)
        Inclusive lower and upper limits for the plotted x-coordinate.
    ylim : tuple of float, default (-0.1, 7.8)
        Inclusive lower and upper limits for the plotted y-coordinate.

    Returns
    -------
    filtered : dict
        New dictionary with the same keys as `peak_data`, where each value is
        filtered to peaks satisfying both coordinate limits.
    """
    x = peak_data["x"]
    y = peak_data["y"]

    mask = ((x >= xlim[0]) & (x <= xlim[1]) &
            (y >= ylim[0]) & (y <= ylim[1]))

    filtered = {}
    for key, arr in peak_data.items():
        filtered[key] = np.asarray(arr)[mask]
        # filtered[key] = arr[mask]
    return filtered




def plot_peak_data_with_labels(data, ax=None, projection='perpendicular', 
                               maxqout=None, scalef=200, color=None,
                               annotate_offset=(0, 5), 
                               xlabel=None, ylabel=None, title=None, 
                               x_lim = (-6.05, 6.05), y_lim = (-0.01, 8.01),
                               fig_size = (5, 5),
                               d_spaces = None,
                               **kwargs):
    """
    Plots peak data (in q-space) and labels each peak with its hkl value.
    
    The function accepts `data` in one of two formats:
      • A structured array with fields 'qx', 'qy', 'qz', 'r' and 'hkl'.
      • A dictionary with at least keys 'x', 'y', and 'hkl', and optionally 'r'
        (if 'r' is missing, a constant marker size is used).
        
    Parameters
    ----------
    data : structured array or dict
        If a structured array, the function will use the 'qx', 'qy', etc. fields.
        If a dictionary, it should contain 'x', 'y', and 'hkl'.
    ax : matplotlib Axes, optional
        Axes object on which to plot. If None, a new figure and axes are created.
    projection : str, default 'perpendicular'
        If the data contains qx/qy/qz, and projection=='perpendicular',
        the x coordinate is taken from 'qy' and y from 'qz'. Otherwise,
        x is computed as sign(qy)*sqrt(qx**2+qy**2) and y is from 'qz'.
        If the input data already provides 'x' and 'y', these are used directly.
    maxqout : float, optional
        If provided and the data contains a 'qx' field, only peaks satisfying
        |qx| < maxqout will be plotted.
    scalef : float or callable, default 100
        If the data has a field 'r', marker sizes are computed as
        (data['r'] * scalef) or via scalef(val) for each intensity. Otherwise,
        a constant marker size equal to scalef is used.
    color : color spec, optional
        Color for the markers.
    annotate_offset : tuple of two floats, default (0, 5)
        Offset (dx, dy) in points for the hkl annotation relative to each marker.
    xlabel : str, optional
        Label for the x-axis; if None, a default label is used.
    ylabel : str, optional
        Label for the y-axis; if None, a default label is used.
    title : str, optional
        Title for the plot.
    **kwargs : dict
        Additional keyword arguments passed to ax.scatter.
    
    Returns
    -------
    ax : matplotlib Axes
        The Axes object with the plotted data.
    scatter_handle : matplotlib PathCollection
        The scatter plot handle.
    """
    # Create or use the provided axes.
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
        plt.sca(ax)
    
    # Decide whether the input data contains 'qx' (for vectorized q data)
    # or is already provided as (x, y) coordinates.
    if isinstance(data, dict) and ('qx' in data or 'qy' in data or 'qz' in data):
        # Data in q-space (structured-like) available.
        if maxqout is not None and 'qx' in data:
            mask = np.abs(data['qx']) < maxqout
        else:
            mask = np.ones(len(data['qy']), dtype=bool)
        
        if projection == 'perpendicular':
            # Use y coordinate from 'qy' and vertical coordinate from 'qz'
            x = data['qy'][mask]
        else:
            # Alternative projection: combine qx and qy to a single in-plane value.
            x = np.sign(data['qy'][mask]) * np.sqrt(data['qx'][mask]**2 + data['qy'][mask]**2)
        # y always taken from 'qz'
        y = data['qz'][mask]
        
        # For marker sizing: if 'r' exists use it, otherwise use constant size.
        if 'r' in data:
            if callable(scalef):
                s = np.array([scalef(val) for val in data['r'][mask]])
            else:
                s = data['r'][mask] * scalef
        else:
            s = np.full(x.shape, scalef)
        
        # hkl labels. Expecting field 'hkl'
        # hkl_labels = data['hkl']
        hkl_labels = data["hkl"][mask]
    elif isinstance(data, dict) and ('x' in data and 'y' in data):
        # Data already contains x and y.
        x = data['x']
        y = data['y']
        mask = np.ones(len(x), dtype=bool)
        if 'r' in data:
            if callable(scalef):
                s = np.array([scalef(val) for val in data['r']])
            else:
                s = data['r'] * scalef
        else:
            s = np.full(x.shape, scalef)
        # hkl_labels = data['hkl']  # Expecting an array of labels.
        hkl_labels = data["hkl"][mask]
    else:
        raise KeyError("Input data must contain either ('qx', 'qy', 'qz', 'hkl') or ('x', 'y', 'hkl').")
    
    # Use keyword arguments for scatter.
    kwargs.setdefault("s", s)
    kwargs.setdefault("c", color)
    scatter_handle = ax.scatter(x, y, **kwargs)
    
    # Set default axis labels if none given.
    if xlabel is None:
        xlabel = r'$Q_{in-plane}$ ($\mathrm{\AA^{-1}}$)' if ('qx' in data) else r'$x$ ($\mathrm{\AA^{-1}}$)'
    if ylabel is None:
        ylabel = r'$Q_{out-of-plane}$ ($\mathrm{\AA^{-1}}$)' if ('qz' in data) else r'$y$ ($\mathrm{\AA^{-1}}$)'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set title if given.
    if title is not None:
        ax.set_title(title)
    
    ax.set_aspect('equal')
    fig.tight_layout()
    
    # Annotate each point with its hkl label.
    # We use annotate_offset as an offset in points.
    for xi, yi, hkl in zip(x, y, hkl_labels):
        ax.annotate(f"{hkl}", (xi, yi), textcoords="offset points",
                    xytext=annotate_offset, fontsize=9, color="black",
                    ha="center", va="bottom")
    if d_spaces is not None:
        for d in d_spaces:
            radius = 2 * np.pi / d
            circle1 = plt.Circle((0, 0), 2*np.pi/d, color='r', fill=False)
            ax.add_patch(circle1)
            ax.text(0, radius + 0.12, f'd = {d:.4f}', color='blue', fontsize=10, ha='center')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect(1)
    
    return ax, scatter_handle


def _to_hkl_tuple(value):
    """
    Convert an HKL value stored as a tuple, list, NumPy array, or string into
    a plain Python tuple.

    Examples
    --------
    (1, 0, 2)     -> (1, 0, 2)
    [1, 0, 2]     -> (1, 0, 2)
    np.array(...) -> (1, 0, 2)
    "(1, 0, 2)"   -> (1, 0, 2)
    
    Parameters
    ----------
    value : tuple, list, ndarray, or str
        HKL value stored either as an iterable of indices or as a string
        representation such as ``"(1, 0, 2)"``.

    Returns
    -------
    hkl : tuple
        HKL indices as a plain Python tuple.
    """
    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    if isinstance(value, np.ndarray):
        return tuple(value.tolist())

    return tuple(ast.literal_eval(value))

    
def _ensure_hkl_cols(df, hkl_col="hkl"):
    """
    Return a copy of `df` with explicit h, k, and l columns.

    If the columns already exist, the input table is copied and returned
    unchanged. Otherwise, the HKL values are parsed from `hkl_col`.
    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing either existing ``'h'``, ``'k'``, and ``'l'``
        columns, or an HKL column specified by `hkl_col`.
    hkl_col : str, default 'hkl'
        Name of the column containing HKL values when explicit ``'h'``,
        ``'k'``, and ``'l'`` columns are absent.

    Returns
    -------
    out : pandas.DataFrame
        Copy of `df` with explicit ``'h'``, ``'k'``, and ``'l'`` columns.
    """
    out = df.copy()
    
    if {"h", "k", "l"}.issubset(out.columns):
        return out
        
    # hkl_t = out[hkl_col].apply(lambda v: v if isinstance(v, tuple) else ast.literal_eval(v))
    hkl_t = out[hkl_col].apply(_to_hkl_tuple)
    
    out[["h", "k", "l"]] = pd.DataFrame(hkl_t.tolist(), 
                                        index=out.index
                                       )
    return out


def extract_hk_symmetry_from_table(
    df, h, k, l=None,
    hkl_col="hkl",
    key_cols=None,
    use_abs_x=True,
    rtol=1e-8, atol=1e-6,
    return_rows=True
):
    """
    Extract hk symmetry family from the table by matching coordinates.

    Default behavior:
      - if l is None: match by |x| only (same rod family across all L)
      - if l is given: match by |x| and l (you can also include 'y' if you want)

    Parameters
    ----------
    df : pandas.DataFrame
        Peak table containing either explicit ``'h'``, ``'k'``, ``'l'`` columns
        or an HKL column specified by `hkl_col`.
    h, k : int
        Reference in-plane Miller indices.
    l : int or float, optional
        Optional reference out-of-plane index. If provided, rows are also
        filtered to the same ``l`` value.
    hkl_col : str, default 'hkl'
        Name of the column containing HKL tuples or tuple-like strings.
    key_cols : sequence of str, optional
        Columns used to match the reference family. If None, the current
        implementation matches by ``'x'`` only.
    use_abs_x : bool, default True
        If True and ``'x'`` is among `key_cols`, match by ``abs(x)`` so that
        positive and negative x coordinates are grouped together.
    rtol, atol : float
        Relative and absolute tolerances passed to ``numpy.isclose``.
    return_rows : bool, default True
        If True, return both the list of HK pairs and the matching table rows.
        If False, return only the list of HK pairs.
    
    Returns
    -------
    hk_list : list of tuple[int, int]
        Sorted list of unique ``(h, k)`` pairs matching the reference
        coordinate family.
    rows : pandas.DataFrame
        Returned only when ``return_rows=True``. Matching rows with selected
        columns sorted by ``l``, ``h``, and ``k``.

    Raises
    ------
    ValueError
        If the reference peak is not present in the table, or if a requested
        matching column is missing.


    
    """
    out = _ensure_hkl_cols(df, hkl_col=hkl_col)

    ref = out[(out["h"] == h) & (out["k"] == k)]
    if l is not None:
        ref = ref[ref["l"] == l]
    if ref.empty:
        raise ValueError(f"Reference peak (h,k,l)=({h},{k},{l}) not found in df.")

    if key_cols is None:
        key_cols = ["x"] if l is None else ["x"]

    ref_row = ref.iloc[0]

    mask = np.ones(len(out), dtype=bool)
    for c in key_cols:
        if c not in out.columns:
            raise ValueError(f"Column '{c}' not in df. Available: {list(out.columns)}")

        if c == "x" and use_abs_x:
            mask &= np.isclose(out[c].abs().to_numpy(), abs(ref_row[c]), rtol=rtol, atol=atol)
        else:
            mask &= np.isclose(out[c].to_numpy(), ref_row[c], rtol=rtol, atol=atol)

    if l is not None:
        mask &= (out["l"].to_numpy() == l)

    res = out.loc[mask].copy()

    hk_list = sorted({(int(r.h), int(r.k)) for r in res.itertuples(index=False)})

    if return_rows:
        cols = [c for c in ["h","k","l","x","y",hkl_col] if c in res.columns]
        return hk_list, res.sort_values(["l","h","k"])[cols].reset_index(drop=True)

    return hk_list


def hexagonal_hk_symmetry(h, k):
    """
    Return unique 6-fold hexagonal symmetry equivalents of (h, k).
   
    Parameters
    ----------
    h, k : int
        In-plane Miller indices.

    Returns
    -------
    family : list of tuple[int, int]
        Unique symmetry-equivalent ``(h, k)`` pairs in hexagonal indexing.

    """
    fam = [
        ( h,      k      ),
        (-k,      h + k  ),
        (-(h+k),  h      ),
        (-h,     -k      ),
        ( k,     -(h+k)  ),
        ( h + k, -h      ),
    ]
    return list(dict.fromkeys(fam))
    

def brute_force_hk_symmetry_search(
    rods,
    keep_frac,
    theta_grid_coarse,
    omes_ang, UBinv, Lambda, x0, y0, pix_size, SDD,
    lock_first=True,
    skip_duplicate_hk=False,
    hk_families = None,
):
    """
    Brute-force over symmetry-equivalent hk labels for each rod.
    Search over candidate HK assignments for several rods.

    For each combination of candidate ``(h, k)`` labels, the function builds a
    temporary rod list, scans `theta_grid_coarse`, and stores the best coarse
    theta offset and corresponding cost.

    Parameters
    ----------
    rods : list of dict
        Rod definitions. Each dictionary must contain:

        - ``'points'`` : array of shape ``(n_points, 3)`` with
          ``(image_index, gamma_pixel, delta_pixel)`` rows.
        - ``'hk'`` : default or reference ``(h, k)`` label.

        Optional keys are:

        - ``'l'`` : target L value.
        - ``'wl'`` : weight applied to the L residual.
    keep_frac : float
        Fraction of lowest point-wise residuals kept when computing each rod
        cost. Passed to :func:`robust_best_fraction`.
    theta_grid_coarse : array-like
        Candidate theta offsets in degrees.
    omes_ang : array-like
        Omega angle values indexed by image number.
    UBinv : ndarray of shape (3, 3)
        Inverse UB matrix used to convert reciprocal-space vectors to HKL.
    Lambda : float
        X-ray wavelength.
    x0, y0 : float
        Direct-beam detector center in pixel coordinates.
    pix_size : float
        Detector pixel size in the same length unit as `SDD`.
    SDD : float
        Sample-detector distance in the same length unit as `pix_size`.
    lock_first : bool, default True
        If True, keep the first rod fixed to its original ``'hk'`` label to
        remove global symmetry duplicates.
    skip_duplicate_hk : bool, default False
        If True, skip HK combinations where the same HK label appears more
        than once.
    hk_families : list of list of tuple[int, int], optional
        Candidate HK labels for each rod. If None, sixfold hexagonal symmetry
        families are generated from ``rod['hk']``.

    Returns
    -------
    df : pandas.DataFrame
        Search results sorted by ``'cost_coarse'`` and ``'theta0_coarse'``.
        The table contains ``'theta0_coarse'``, ``'cost_coarse'``, and one
        ``'hk{i}'`` column per rod.

    """
    if hk_families is None:
        print("  Assuming hexagonal 6 fold symmentry ...")
        hk_families = [hexagonal_hk_symmetry(*r["hk"]) for r in rods]

    # remove global symmetry duplicates by fixing first rod label
    if lock_first:
        hk_families[0] = [rods[0]["hk"]]

    results = []

    # coarse_step = float(theta_grid_coarse[1] - theta_grid_coarse[0])

    for hk_combo in tqdm(itertools.product(*hk_families),):
        if skip_duplicate_hk and len(set(hk_combo)) < len(hk_combo):
            continue

        rods_test = []
        for r, hk in zip(rods, hk_combo):
            rods_test.append({
                "points": r["points"],
                "hk": hk,
                "l": r.get("l", None),
                "wl": r.get("wl", 0.0),
            })

        # coarse search
        th0_coarse, costs_coarse = scan_theta(
            theta_grid_coarse, rods_test, keep_frac,
            omes_ang, UBinv, Lambda, x0, y0, pix_size, SDD
        )

        row = {
            "theta0_coarse": float(th0_coarse),
            "cost_coarse": float(costs_coarse.min()),
        }

        for i, hk in enumerate(hk_combo, start=1):
            row[f"hk{i}"] = hk

        results.append(row)

    df = pd.DataFrame(results).sort_values(
        ["cost_coarse", "theta0_coarse"]
    ).reset_index(drop=True)

    return df



def scan_theta(theta_grid, rods, keep_frac, omes_ang, 
               UBinv, Lambda, x0, y0, pix_size, SDD):
    """
    Evaluate the total rod assignment cost on a grid of theta offsets.

    Parameters
    ----------
    theta_grid : array-like
        Candidate theta offsets in degrees.
    rods : list of dict
        Rod definitions passed to `total_cost`.
    keep_frac : float
        Fraction of lowest residuals kept for each rod.
    omes_ang : array-like
        Omega angle values indexed by image number.
    UBinv : ndarray of shape (3, 3)
        Inverse UB matrix.
    Lambda : float
        X-ray wavelength.
    x0, y0 : float
        Direct-beam detector center in pixel coordinates.
    pix_size : float
        Detector pixel size in the same length unit as `SDD`.
    SDD : float
        Sample-detector distance in the same length unit as `pix_size`.

    Returns
    -------
    best_theta : scalar
        Theta value from `theta_grid` with the lowest cost.
    costs : ndarray
        Cost value for each theta value in `theta_grid`.

    """
    
    costs = np.array([
        total_cost(t, rods, keep_frac, omes_ang, UBinv, Lambda, x0, y0, pix_size, SDD)
        for t in theta_grid
    ])
    best_idx = np.argmin(costs)
    return theta_grid[best_idx].item(), costs



# def robust_best_fraction(L2, keep_frac, mode = 'median'):
#     L2 = np.asarray(L2)
#     m = max(1, int(np.ceil(keep_frac * L2.size)))
#     if mode == 'mean':
#         return np.nanmean(np.partition(L2, m-1)[:m])  # faster than full sort
#     if mode == 'median':
#         return np.nanmedian(np.partition(L2, m-1)[:m])


def robust_best_fraction(L2, keep_frac, mode="median"):
    
    """
    Compute a robust cost from the lowest fraction of finite residuals.

    Parameters
    ----------
    L2 : array-like
        Point-wise squared residuals or costs.
    keep_frac : float
        Fraction of the smallest finite values to keep. Must satisfy
        ``0 < keep_frac <= 1``.
    mode : {'median', 'mean'}, default 'median'
        Statistic used to summarize the retained lowest residuals.

    Returns
    -------
    cost : float
        Mean or median of the retained lowest finite residuals.

    Raises
    ------
    ValueError
        If `L2` contains no finite values, if `keep_frac` is outside
        ``(0, 1]``, or if `mode` is not supported.
    """
    L2 = np.asarray(L2, dtype=float)
    L2 = L2[np.isfinite(L2)]

    if L2.size == 0:
        raise ValueError("L2 must contain at least one finite value.")
    if not (0 < keep_frac <= 1):
        raise ValueError("keep_frac must satisfy 0 < keep_frac <= 1.")
    if mode not in {"mean", "median"}:
        raise ValueError("mode must be 'mean' or 'median'.")

    m = max(1, int(np.ceil(keep_frac * L2.size)))
    best = np.partition(L2, m - 1)[:m]

    if mode == "mean":
        return float(np.mean(best))
    return float(np.median(best))


def total_cost(theta0, rods, keep_frac, omes_ang, UBinv, Lambda, x0, y0, pix_size, SDD):
    """
    Compute the total robust assignment cost for all rods at one theta offset.

    Parameters
    ----------
    theta0 : float
        Candidate theta offset in degrees.
    rods : list of dict
        Rod definitions. Each dictionary must contain ``'points'`` and
        ``'hk'``. Optional keys are ``'l'`` and ``'wl'``.
    keep_frac : float
        Fraction of the lowest point-wise residuals kept for each rod.
    omes_ang : array-like
        Omega angle values indexed by image number.
    UBinv : ndarray of shape (3, 3)
        Inverse UB matrix.
    Lambda : float
        X-ray wavelength.
    x0, y0 : float
        Direct-beam detector center in pixel coordinates.
    pix_size : float
        Detector pixel size in the same length unit as `SDD`.
    SDD : float
        Sample-detector distance in the same length unit as `pix_size`.

    Returns
    -------
    total : float
        Sum of robust costs over all rods.
    """
    
    # rods is a list of dicts: {"points": ..., "hk": (h,k), "l": L or None, "wl": ...}
    costs = []
    for r in rods:
        L2 = rod_L2(
            r["points"], r["hk"], theta0, omes_ang, UBinv, Lambda, x0, y0, pix_size, SDD,
            l_target=r.get("l", None), wl=r.get("wl", 0.0)
        )
        costs.append(robust_best_fraction(L2, keep_frac))
    return np.sum(costs)



def pixels_to_hkl_pointwise(xpix, ypix, theta_deg, UBinv, Lambda, x0, y0, pix_size, SDD):
    """
    Convert detector pixel coordinates to HKL values point by point.

    The function computes detector angles from pixel coordinates, forms the
    corresponding scattering vector, rotates it by the effective theta angle,
    and applies the inverse UB matrix.

    Parameters
    ----------
    xpix, ypix : array-like or float
        Detector pixel coordinates. `xpix` corresponds to the delta direction
        and `ypix` to the gamma direction.
    theta_deg : array-like or float
        Sample rotation angle in degrees. Must be broadcast-compatible with
        `xpix` and `ypix`.
    UBinv : ndarray of shape (3, 3)
        Inverse UB matrix.
    Lambda : float
        X-ray wavelength. Must be positive.
    x0, y0 : float
        Direct-beam detector center in pixel coordinates.
    pix_size : float
        Detector pixel size in the same length unit as `SDD`.
    SDD : float
        Sample-detector distance in the same length unit as `pix_size`.

    Returns
    -------
    h, k, l : ndarray
        Calculated Miller-index coordinates. Shapes follow NumPy broadcasting
        of `xpix`, `ypix`, and `theta_deg`.

    Raises
    ------
    ValueError
        If `Lambda` is not positive or `UBinv` does not have shape ``(3, 3)``.
    """
    # detector angles
    delta = np.arctan(-(xpix - x0) * pix_size / SDD)
    gamma = np.arctan(-(ypix - y0) * pix_size / SDD)
    theta = np.deg2rad(theta_deg)
    
    if Lambda <= 0:
        raise ValueError("Lambda must be positive.")
        
    k0 = 2 * np.pi / Lambda

    cg, sg = np.cos(gamma), np.sin(gamma)
    cd, sd = np.cos(delta), np.sin(delta)
    ct, st = np.cos(theta), np.sin(theta)

    # q_lab = (Rdelta @ Rgamma - I) @ [0, k0, 0]
    qx = -k0 * cg * sd
    qy =  k0 * (cg * cd - 1.0)
    qz =  k0 * sg

    # rotate into sample frame by Rz(-theta)
    rx =  ct * qx + st * qy
    ry = -st * qx + ct * qy
    rz =  qz

    q = np.stack([rx, ry, rz], axis=0)   # shape (3, N)
    UBinv = np.asarray(UBinv, dtype=float)
    if UBinv.shape != (3, 3):
        raise ValueError("UBinv must have shape (3, 3).")
    hkl = UBinv @ q

    return hkl[0], hkl[1], hkl[2]


def rod_L2(points, hk_target, theta0, omes_ang, UBinv, Lambda, x0, y0, pix_size, SDD,
           l_target=None, wl=0.0):
    """
    Compute point-wise squared HK residuals for one rod.

    Parameters
    ----------
    points : ndarray of shape (n_points, 3)
        Rod pixel coordinates as ``(image_index, gamma_pixel, delta_pixel)``.
    hk_target : tuple[float, float]
        Target in-plane ``(h, k)`` assignment for the rod.
    theta0 : float
        Theta offset in degrees subtracted from the omega angle of each point.
    omes_ang : array-like
        Omega angle values indexed by image number.
    UBinv : ndarray of shape (3, 3)
        Inverse UB matrix.
    Lambda : float
        X-ray wavelength.
    x0, y0 : float
        Direct-beam detector center in pixel coordinates.
    pix_size : float
        Detector pixel size in the same length unit as `SDD`.
    SDD : float
        Sample-detector distance in the same length unit as `pix_size`.
    l_target : float, optional
        Optional target L value.
    wl : float, default 0.0
        Weight applied to the L residual when `l_target` is provided.

    Returns
    -------
    d2 : ndarray of shape (n_points,)
        Squared residuals in HK space. If `l_target` is provided and `wl > 0`,
        the weighted L residual is added.
    """
    n = points[:, 0].astype(int)
    y = points[:, 1].astype(float)
    x = points[:, 2].astype(float)

    theta_eff = omes_ang[n] - theta0

    h, k, l = pixels_to_hkl_pointwise(x, y, theta_eff, UBinv, Lambda, x0, y0, pix_size, SDD)

    d2 = (h - hk_target[0])**2 + (k - hk_target[1])**2
    if l_target is not None and wl > 0:
        d2 = d2 + wl * (l - l_target)**2

    return d2



def rod_hkl_L2(points, hk_target, theta0, omes_ang, UBinv, Lambda, x0, y0, pix_size, SDD, l_target=None, wl=0.0):
    """
    Compute HKL values and point-wise squared residuals for one rod.

    Parameters
    ----------
    points : ndarray of shape (n_points, 3)
        Rod pixel coordinates as ``(image_index, gamma_pixel, delta_pixel)``.
    hk_target : tuple[float, float]
        Target in-plane ``(h, k)`` assignment for the rod.
    theta0 : float
        Theta offset in degrees subtracted from the omega angle of each point.
    omes_ang : array-like
        Omega angle values indexed by image number.
    UBinv : ndarray of shape (3, 3)
        Inverse UB matrix.
    Lambda : float
        X-ray wavelength.
    x0, y0 : float
        Direct-beam detector center in pixel coordinates.
    pix_size : float
        Detector pixel size in the same length unit as `SDD`.
    SDD : float
        Sample-detector distance in the same length unit as `pix_size`.
    l_target : float, optional
        Optional target L value.
    wl : float, default 0.0
        Weight applied to the L residual when `l_target` is provided.

    Returns
    -------
    h, k, l : ndarray
        Calculated HKL coordinates for each point.
    L2 : ndarray of shape (n_points,)
        Squared residuals in HK space. If `l_target` is provided and `wl > 0`,
        the weighted L residual is added.
    """
    n = points[:, 0].astype(int)
    y = points[:, 1].astype(float)
    x = points[:, 2].astype(float)

    theta_eff = omes_ang[n] - theta0

    h, k, l = pixels_to_hkl_pointwise(
        x, y, theta_eff, UBinv, Lambda, x0, y0, pix_size, SDD
    )

    L2 = (h - hk_target[0])**2 + (k - hk_target[1])**2
    if l_target is not None and wl > 0:
        L2 = L2 + wl * (l - l_target)**2

    return h, k, l, L2



def report_best_fraction(theta0, rods, keep_frac, omes_ang, UBinv, Lambda, x0, y0, pix_size, SDD,
                         mode='median'):
    """
    Print diagnostic HKL statistics for the best residual fraction of each rod.

    For each rod, the function computes HKL coordinates and residuals at
    `theta0`, keeps the lowest `keep_frac` fraction of finite residuals, and
    prints the best point, summary HKL values, and summary residual cost.

    Parameters
    ----------
    theta0 : float
        Theta offset in degrees.
    rods : list of dict
        Rod definitions. Each dictionary must contain ``'points'`` and
        ``'hk'``. Optional keys are ``'l'`` and ``'wl'``.
    keep_frac : float
        Fraction of lowest finite residuals to summarize.
    omes_ang : array-like
        Omega angle values indexed by image number.
    UBinv : ndarray of shape (3, 3)
        Inverse UB matrix.
    Lambda : float
        X-ray wavelength.
    x0, y0 : float
        Direct-beam detector center in pixel coordinates.
    pix_size : float
        Detector pixel size in the same length unit as `SDD`.
    SDD : float
        Sample-detector distance in the same length unit as `pix_size`.
    mode : {'median', 'mean'}, default 'median'
        Statistic used to summarize the retained HKL values and residuals.

    Returns
    -------
    None
        Results are printed to standard output.

    Raises
    ------
    ValueError
        If `mode` is not ``'mean'`` or ``'median'``.
    """
    for i, r in enumerate(rods, start=1):
        h, k, l, L2 = rod_hkl_L2(
            r["points"], r["hk"], theta0, omes_ang, UBinv, Lambda, x0, y0, pix_size, SDD,
            l_target=r.get("l", None), wl=r.get("wl", 0.0)
        )

        ok = np.isfinite(L2)
        h, k, l, L2 = h[ok], k[ok], l[ok], L2[ok]

        if L2.size == 0:
            print(f"   rod{i}: no finite points")
            continue

        m = max(1, int(np.ceil(keep_frac * L2.size)))

        idx_keep = np.argpartition(L2, m - 1)[:m]
        idx_best = idx_keep[np.argmin(L2[idx_keep])]

        if mode not in {"mean", "median"}:
            raise ValueError("mode must be 'mean' or 'median'.")
        if mode == 'mean':
            h_stat = np.mean(h[idx_keep])
            k_stat = np.mean(k[idx_keep])
            l_stat = np.mean(l[idx_keep])
            cost_stat = np.mean(L2[idx_keep])
        else:
            h_stat = np.median(h[idx_keep])
            k_stat = np.median(k[idx_keep])
            l_stat = np.median(l[idx_keep])
            cost_stat = np.median(L2[idx_keep])

        print(
            f"   rod{i} " # hk={r['hk']} l={r.get('l', None)} "
            f"len={m}/{L2.size} "
            f"best_hkl=({h[idx_best]:0.5f}, {k[idx_best]:0.5f}, {l[idx_best]:0.5f}) "
            f"{mode}_hkl=({h_stat:0.5f}, {k_stat:0.5f}, {l_stat:0.5f}) "
            f"{mode}_L2={cost_stat:0.6f}"
        )


##########################################
#  modified functions form xrayutilities #
##########################################

def get_allowed_hkl_custom(qmax, lattice):
    """
    Return a set of all allowed reflections up to a maximal momentum transfer qmax,
    using bounding-box enumeration (vector approach) but also calling
    lattice.hkl_allowed(...) to respect reflection conditions.

    Parameters
    ----------
    qmax : float
        Maximum momentum transfer.
    lattice : SGLattice (or similar)
        Lattice object that provides:
        - B (reciprocal lattice matrix)
        - iscentrosymmetric property
        - hkl_allowed((h,k,l), returnequivalents=True) -> (bool, set_of_eqHKLs)

    Returns
    -------
    hklset : set
        Set of allowed (h, k, l) tuples (including equivalents). (0,0,0) is discarded.
    """
    B = lattice.B
    # metric from B
    gij = np.dot(B.T, B)
    max_h = int(math.ceil(qmax / math.sqrt(gij[0, 0])))
    max_k = int(math.ceil(qmax / math.sqrt(gij[1, 1])))
    max_l = int(math.ceil(qmax / math.sqrt(gij[2, 2])))

    # Enumerate all integer HKLs in the bounding box
    hs = np.arange(-max_h, max_h+1)
    ks = np.arange(-max_k, max_k+1)
    ls = np.arange(-max_l, max_l+1)
    H, K, L = np.meshgrid(hs, ks, ls, indexing='ij')
    hkl_candidates = np.vstack((H.ravel(), K.ravel(), L.ravel())).T

    # Compute q-vectors and filter by q <= qmax
    qvecs = np.dot(hkl_candidates, B.T)
    norms = np.linalg.norm(qvecs, axis=1)
    mask_q = (norms <= qmax)
    hkl_candidates = hkl_candidates[mask_q]

    # Now apply reflection conditions:
    hklset = set()
    tested = set()
    is_centro = lattice.iscentrosymmetric

    for (h, k, l) in hkl_candidates:
        # If we have already tested this reflection (or its equivalents), skip
        if (h, k, l) in tested:
            continue

        allowed, eqhkl = lattice.hkl_allowed((h, k, l),
                                             returnequivalents=True)
        # Mark all equivalents as tested
        tested.update(eqhkl)

        # For non-centrosymmetric crystals, also treat (-h,-k,-l) as tested
        # so we do not double-check them later
        if not is_centro:
            tested.update((-hh, -kk, -ll) for (hh, kk, ll) in eqhkl)

        if allowed:
            # If reflection is allowed, store all eqhkl in final set
            hklset.update(eqhkl)

            if not is_centro:
                # Also store the negative equivalents
                neg_equiv = lattice.equivalent_hkls((-h, -k, -l))
                hklset.update(neg_equiv)

    # Typically, we remove the (0,0,0) reflection:
    hklset.discard((0, 0, 0))

    return hklset



def _resolve_xray_energy(en0):
    """
    Resolve x-ray energy from a numeric value or xrayutilities config.
    
    This helper imports ``xrayutilities.config`` lazily.

    Parameters
    ----------
    en0 : float or {'config'}
        Numeric energy value, or ``'config'`` to read the configured
        xrayutilities energy.

    Returns
    -------
    energy : float
        Resolved x-ray energy value.

    Raises
    ------
    ValueError
        If `en0` is a string other than ``'config'``.

    """
    if isinstance(en0, str):
        if en0 != "config":
            raise ValueError("en0 must be a number or 'config'.")

        from xrayutilities import config, utilities
        return utilities.energy(config.ENERGY)

    return en0


def fast_StructureFactorForQ(qs, en0='config', temp=0, threshold=1e-3,
                             lattice=None, debyewallerFunc=None, get_f_func=None):
    """
    This is a vectorized helper adapted from xrayutilities structure
    factor calculations. It computes atomic form factors, applies occupancies,
    phases, and Debye-Waller factors, and returns one complex structure factor
    per input q-vector.
    Vectorized structure factor calculation with early threshold filtering.
    Threshold is applied based on a form factor upper bound.

    
    Parameters:
      qs            : array-like (N x 3) q-vectors.
      en0           : energy value or 'config' (if so, utilities.energy(config.ENERGY) is used)
      temp          : temperature (0 assumes zero temperature)
      threshold     : minimum intensity threshold for early rejection.
      lattice       : lattice object (provides nsites, base(), GetPoint(), B)
      debyewallerFunc: function for Debye–Waller calculations (e.g. lattice._debyewallerfactor)
      get_f_func    : function to compute form factors (e.g. lattice._get_f)
    
    Returns:
      A NumPy array (length N) of complex structure factors.
    """
    qs = np.asarray(qs, dtype=np.double)
    qnorm = np.linalg.norm(qs, axis=1)
    if isinstance(en0, str) and en0 == 'config':
        # Assumes utilities and config are available in the environment.
        # en0 = utilities.energy(config.ENERGY)
        en0 = _resolve_xray_energy(en0)
    if lattice.nsites == 0:
        return np.ones(len(qs), dtype=complex)
    
    base = list(lattice.base())
    occupancies = np.array([o for (_, _, o, _) in base])
    b_factors = np.array([b for (_, _, _, b) in base])
    positions = np.array([lattice.GetPoint(p) for (_, p, _, _) in base])
    
    f_list = get_f_func(qnorm, en0)  # Expected: list of 1D arrays of length len(qs)
    f_array = np.stack(f_list, axis=1)  # Shape: (N, n_atoms)
    
    upper_bound = np.sum(np.abs(f_array * occupancies), axis=1)
    
    # mask = upper_bound >= threshold
    if threshold is None:
        mask = np.ones(len(qs), dtype=bool)
    else:
        mask = upper_bound >= threshold
    if not np.any(mask):
        return np.zeros(len(qs), dtype=complex)
    
    qs_masked = qs[mask]
    qnorm_masked = qnorm[mask]
    f_array_masked = f_array[mask]
    
    if temp == 0:
        dwf = np.exp(-np.outer(qnorm_masked**2, b_factors) / (4 * np.pi)**2)
    else:
        dwf = np.array([debyewallerFunc(temp, q) for q in qnorm_masked])
    
    phase = np.exp(-1j * np.dot(qs_masked, positions.T))  # Shape: (n_masked, n_atoms)
    contributions = f_array_masked * occupancies * phase * dwf
    S_masked = np.sum(contributions, axis=1)
    
    S = np.zeros(len(qs), dtype=complex)
    S[mask] = S_masked
    
    return S
    

def show_reciprocal_space_plane(mat, exp, ttmax=None, maxqout=0.01, scalef=100, ax=None,
                                color=None, show_Laue=True, show_legend=True,
                                projection='perpendicular', label=None, min_intensity=None,
                                sf_threshold=None, q_max=None, **kwargs):
    """
    Plots the coplanar diffraction plane with peak positions.
    Uses early rejection of peaks via a structure factor threshold.

    The function enumerates allowed HKL reflections, transforms their q-vectors
    into the experimental frame, computes relative structure-factor
    intensities, filters peaks near the selected diffraction plane, and plots
    them as an interactive scatter plot. 
    
    Parameters:
        mat : object
            Material/crystal object expected to provide ``name``, ``lattice``,
            ``Q(hkls)``, ``StructureFactor(hkl, energy)``,
            ``_debyewallerfactor``, and ``_get_f``.
        exp : object
            Experiment object expected to provide ``k0``, ``energy``,
            ``Transform(q)``, and ``Q2Ang(q, trans=False, geometry='real')``.
        ttmax : float, optional
            Maximum two-theta angle in degrees used to define the q-limit. If None,
            the current implementation sets it to ``180.0``.
        maxqout : float, default 0.01
            Maximum allowed absolute out-of-plane component as a fraction of
            ``exp.k0``. Peaks are plotted only when ``abs(qx) < maxqout * exp.k0``.
        scalef : float or callable, default 100
            Marker-size scaling. If callable, it is applied to each relative
            intensity. Otherwise marker size is ``relative_intensity * scalef``.
        ax : matplotlib.axes.Axes, optional
            Axes object on which to plot. If None, a new figure and axes are
            created.
        color : matplotlib color, optional
            Marker color passed to ``Axes.scatter``.
        show_Laue : bool, default True
            If True, draw Laue-circle guide elements.
        show_legend : bool, default True
            If True, draw a figure legend.
        projection : {'perpendicular', other}, default 'perpendicular'
            If ``'perpendicular'``, plot ``qy`` versus ``qz``. For any other value,
            plot ``sign(qy) * sqrt(qx**2 + qy**2)`` versus ``qz``.
        label : str, optional
            Legend label for the plotted material. If None, ``mat.name`` is used.
        min_intensity : float, optional
            If provided, discard peaks with relative intensity below this value
            after structure-factor calculation.
        sf_threshold : float or None, default None
            Threshold passed to :func:`fast_StructureFactorForQ` and also used to
            filter relative intensities inside the nested peak-generation helper
            when not None.
        q_max : float, optional
            Explicit reciprocal-space cutoff. If provided, it overrides the q-limit
            calculated from `ttmax`.
        **kwargs
            Additional keyword arguments passed to ``Axes.scatter``.

    
    Returns:
      (ax, scatter_handle, peak_data) where peak_data is a dictionary containing plotted coordinates.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the reciprocal-space plot.
    h : matplotlib.collections.PathCollection
        Scatter artist returned by ``Axes.scatter``.
    peak_data : dict
        Dictionary containing plotted peak data:

        - ``'x'`` : plotted x-coordinates.
        - ``'y'`` : plotted y-coordinates.
        - ``'hkl'`` : HKL labels for plotted peaks.
        - ``'qvec'`` : q-vectors ``(qx, qy, qz)`` for plotted peaks.
        - ``'angles'`` : angles returned by ``exp.Q2Ang``.
        - ``'r'`` : relative structure-factor intensities.

    """
    if ttmax is None:
       ttmax = 180.0
    
    def get_peaks(mat, exp, ttmax, sf_threshold=None):
        t0 = time.time()
        qmax = 2 * exp.k0 * math.sin(math.radians(ttmax/2.))
        #print(f'before qmax = {qmax}')
        if q_max is not None:
            qmax = q_max
        #print(f'qmax = {qmax}')
        hkls = tuple(get_allowed_hkl_custom(qmax, mat.lattice))
        #print(f'len(hkls) = {len(hkls)}')
        t_hkl = time.time() - t0
        #print(f"get_allowed_hkl_custom: {t_hkl:.3f}s, {len(hkls)} peaks")
    
        t0 = time.time()
        q = mat.Q(hkls)
        t_q = time.time() - t0
        #print(f"mat.Q: {t_q:.3f}s")
    
        data = np.zeros(len(hkls), dtype=[('qx', np.double),
                                           ('qy', np.double),
                                           ('qz', np.double),
                                           ('r',  np.double),
                                           ('hkl', np.object_)])
        t0 = time.time()
        qvec = exp.Transform(q)
        data['qx'] = qvec[:, 0]
        data['qy'] = qvec[:, 1]
        data['qz'] = qvec[:, 2]
        t_transform = time.time() - t0
        #print(f"Transform: {t_transform:.3f}s")
    
        t0 = time.time()
        # Use the vectorized, early-rejection structure factor function.
        sf_values = fast_StructureFactorForQ(q, exp.energy, threshold=sf_threshold,
                                             lattice=mat.lattice,
                                             debyewallerFunc=mat._debyewallerfactor,
                                             get_f_func=mat._get_f)
        sf_magnitudes = np.abs(sf_values)**2
        rref = abs(mat.StructureFactor((0, 0, 0), exp.energy))**2
        data['r'] = sf_magnitudes / rref
        t_sf = time.time() - t0
        #print(f"StructureFactor (fast): {np.sort(data['r'][5:35])}")
        #print(f"StructureFactor (fast): {t_sf:.3f}s, max: {data['r'].max()}, min: {data['r'].min()}")

        if sf_threshold is not None:
            mask = data['r'] >= sf_threshold
            data = data[mask]  # Filter entire array
        
            filtered_hkls = [hkl for hkl, m in zip(hkls, mask) if m]
            # Create a 1D object array to store each (h, k, l) tuple
            hkl_objects = np.empty(len(filtered_hkls), dtype=object)
            for i, hkl in enumerate(filtered_hkls):
                hkl_objects[i] = hkl
        
            data['hkl'] = hkl_objects
            
        return data

    # Get (or create) the matplotlib Axes.
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure
        plt.sca(ax)
    
    plt.axis('scaled')
    ax.set_autoscaley_on(False)
    ax.set_autoscalex_on(False)
    k0 = exp.k0
    plt.xlim(-2.05 * k0, 2.05 * k0)
    plt.ylim(-0.05 * k0, 2.05 * k0)
    
    if show_Laue:
        c = plt.Circle((0, 0), 2 * k0, facecolor='#FF9180', edgecolor='none')
        ax.add_patch(c)
        qmax_circle = 2 * k0 * math.sin(math.radians(ttmax/2.))
        c = plt.Circle((0, 0), qmax_circle, facecolor='#FFFFFF', edgecolor='none')
        ax.add_patch(c)
        c = plt.Circle((0, 0), 2 * k0, facecolor='none', edgecolor='0.5')
        ax.add_patch(c)
        c = plt.Circle((k0, 0), k0, facecolor='none', edgecolor='0.5')
        ax.add_patch(c)
        c = plt.Circle((-k0, 0), k0, facecolor='none', edgecolor='0.5')
        ax.add_patch(c)
        plt.hlines(0, -2 * k0, 2 * k0, color='0.5', lw=0.5)
        plt.vlines(0, -2 * k0, 2 * k0, color='0.5', lw=0.5)
    
    data = get_peaks(mat, exp, ttmax, sf_threshold=sf_threshold)
    if min_intensity is not None:
        data = data[data['r'] >= min_intensity]
    
    # Select only the peaks with a small qx (i.e. within the in-plane region)
    mask_proj = np.abs(data['qx']) < maxqout * k0
    if projection == 'perpendicular':
        x = data['qy'][mask_proj]
    else:
        x = np.sign(data['qy'][mask_proj]) * np.sqrt(data['qx'][mask_proj]**2 + data['qy'][mask_proj]**2)
    y = data['qz'][mask_proj]
    
    if callable(scalef):
        s = np.array([scalef(val) for val in data['r'][mask_proj]])
    else:
        s = data['r'][mask_proj] * scalef
    
    kwargs.setdefault("label", label if label else mat.name)
    kwargs.setdefault("zorder", 2)
    kwargs.setdefault("s", s)
    kwargs.setdefault("c", color)
    h = ax.scatter(x, y, **kwargs)
    
    ax.set_xlabel(r'$Q$ inplane ($\mathrm{\AA^{-1}}$)')
    ax.set_ylabel(r'$Q$ out of plane ($\mathrm{\AA^{-1}}$)')
    
    if show_legend:
        if fig.legends:
            for leg in fig.legends:
                leg.remove()
        fig.legend(*ax.get_legend_handles_labels(), loc='upper right')
    plt.tight_layout()
    
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    def update_annot(ind):
        pos = h.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        # text = f"{mat.name}\n{str(np.array(data['hkl'])[mask_proj][ind['ind'][0]])}"
        hkl_val = np.array(data['hkl'])[mask_proj][ind['ind'][0]]
        hkl_val = tuple(int(v) for v in hkl_val)
        text = f"{mat.name}\n{hkl_val}"
        annot.set_text(text)
        if h.get_facecolors().size > 0:
            c_color = h.get_facecolors()[0]
        elif h.get_edgecolors().size > 0:
            c_color = h.get_edgecolors()[0]
        else:
            c_color = 'w'
        annot.get_bbox_patch().set_facecolor(c_color)
        annot.get_bbox_patch().set_alpha(0.2)
    
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = h.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    
    def click(event):
        if event.inaxes == ax:
            cont, ind = h.contains(event)
            if cont:
                popts = np.get_printoptions()
                np.set_printoptions(precision=4, suppress=True)
                q_val = (data['qx'][mask_proj][ind["ind"][0]],
                         data['qy'][mask_proj][ind["ind"][0]],
                         data['qz'][mask_proj][ind["ind"][0]])
                angles = exp.Q2Ang(q_val, trans=False, geometry='real')
                text = f"""{mat.name}
hkl: {np.array(data['hkl'])[mask_proj][ind['ind'][0]]}
exp.Q2Ang angles (om, tilt, azimuth, 2th): {angles}"""
                np.set_printoptions(**popts)
                print(text)
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect("button_press_event", click)

    # Prepare peak_data
    coords_masked = np.column_stack((data['qx'][mask_proj], data['qy'][mask_proj], data['qz'][mask_proj]))
    # Compute angles for each q-vector
    angles = np.array([exp.Q2Ang(q, trans=False, geometry='real') for q in coords_masked])
    
    peak_data = {
            'x': x,
            'y': y,
            'hkl': np.array(data['hkl'])[mask_proj],
           # 'hkl_raw': np.array(data['hkl']),
            'qvec': coords_masked,
            'angles': angles,
            'r': data['r'][mask_proj],
        }

    return ax, h, peak_data
