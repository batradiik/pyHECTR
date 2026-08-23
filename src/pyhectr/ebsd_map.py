import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull


__all__ = [
    "axis_directions_from_euler",
    "dz_to_rgb",
    "plot_axis_orientation_map_clickable",
    "select_footprint",
    "read_ebsd_region",
    "calibrate_ebsd_to_mm",
    "coerce_angle_cols",
    "apply_affine",
    "fit_affine",
    "invert_affine",
    "best_corner_pairing",
]

def coerce_angle_cols(df, cols=("phi1", "PHI", "phi2")):
    """Make sure Euler columns are float, support comma or dot as decimal."""
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"There is no column '{c}' in the table. Check the header.")
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
    return df


# def _unit(v):
#     """Return unit vectors along the last array axis."""
#     v = np.asarray(v, float)
#     n = np.linalg.norm(v, axis=-1, keepdims=True)
#     return v / n


def axis_directions_from_euler(df, axis = "Z", euler_cols=("phi1", "PHI", "phi2")):
    """
    For each row in df, return the direction of the sample axis (X/Y/Z)
    expressed in the crystal frame, using Bunge ZXZ Euler angles.

    Returns: array of shape (N, 3)
    """
    axis = axis.upper()
    axis_map = {"X": 0, "Y": 1, "Z": 2}
    if axis not in axis_map:
        raise ValueError("axis must be 'X', 'Y' or 'Z'")
    iax = axis_map[axis]

    phi1 = df[euler_cols[0]].to_numpy(float)
    PHI  = df[euler_cols[1]].to_numpy(float)
    phi2 = df[euler_cols[2]].to_numpy(float)

    angles = np.vstack([phi1, PHI, phi2]).T        # (N,3)
    G = R.from_euler("ZXZ", angles, degrees=True).as_matrix()  # (N,3,3)

    # column of G are crystal directions of sample X,Y,Z
    dirs = G[:, :, iax]                            # (N,3)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dirs / norms


def dz_to_rgb(dz, mode="gamma", gamma=0.55):
    """
    Map crystal frame direction cosines to RGB colours.

    Parameters
    ----------
    dz : ndarray, shape (N, 3)
        Direction vectors, for example the sample Z-axis expressed in the
        crystal frame.
    mode : {"simple", "global", "gamma"}, optional
        Colour normalization mode.

        ``"simple"``
            Normalize each direction by its largest absolute component:

            ``RGB_i = abs(n_i) / max_j(abs(n_j))``

        ``"global"``
            Stretch each RGB channel independently to the interval ``[0, 1]``
            using the minimum and maximum values over the complete map.

        ``"gamma"``
            Apply the same global channel normalization followed by gamma
            correction. Values of ``gamma < 1`` increase contrast.

    gamma : float, default 0.55
        Gamma exponent used only when ``mode="gamma"``.

    Returns
    -------
    rgb : ndarray, shape (N, 3)
        RGB values clipped to the interval ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"simple"``, ``"global"``, or ``"gamma"``.
    """
    dz_abs = np.abs(dz)

    if mode == "simple":
        # Normalise so max component = 1 for each vector
        max_comp = dz_abs.max(axis=1, keepdims=True)
        max_comp[max_comp == 0] = 1.0
        rgb = dz_abs / max_comp

    elif mode in ("global", "gamma"):
        # Per-channel min/max across all points
        ch_min = dz_abs.min(axis=0, keepdims=True)
        ch_max = dz_abs.max(axis=0, keepdims=True)
        span = ch_max - ch_min
        span[span == 0] = 1.0

        rgb = (dz_abs - ch_min) / span  # [0,1]

        if mode == "gamma":
            rgb = np.power(rgb, gamma)

    else:
        raise ValueError("mode must be 'simple', 'global', or 'gamma'")

    return np.clip(rgb, 0.0, 1.0)


def plot_axis_orientation_map_clickable(
    df,
    axis = "Z",              # <-- choose "X", "Y" or "Z"
    x_col = "X",
    y_col = "Y",
    euler_cols=("phi1", "PHI", "phi2"),
    figsize=(6, 8),
    point_size=8,
    flip_y = True,
    title  = None,
    gamma = 0.55,
    mode = "simple"
):
    """
    EBSD map where each point is at (x,y) and coloured by the
    orientation of the chosen sample axis (X/Y/Z) in the crystal frame.

    Clicking shows:
      - Euler angles
      - direction n_axis in crystal frame
      - optional nearest [uvw]_axis + misorientation if those columns exist
    """
    axis = axis.upper()
    if title is None:
        title = f"{axis}-axis orientation map"

    # Positions
    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)

    # Axis direction + colours
    d_axis = axis_directions_from_euler(df, axis=axis, euler_cols=euler_cols)  # (N,3)
    colors = dz_to_rgb(d_axis, gamma = gamma, mode = mode)                                                # reuse your function

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        x,
        y,
        c=colors,
        s=point_size,
        marker="s",
        linewidths=0,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if flip_y:
        ax.invert_yaxis()
    ax.set_title(title)

    ax.text(
        0.01,
        0.01,
        f"Click on a point to see {axis}-axis orientation.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    # ---------------- interactive part ----------------
    phi1_arr = df[euler_cols[0]].to_numpy(float)
    PHI_arr  = df[euler_cols[1]].to_numpy(float)
    phi2_arr = df[euler_cols[2]].to_numpy(float)

    # optional: if df already contains uvw_X/Y/Z and ang_uvw_*_deg
    uvw_col = f"uvw_{axis}"
    ang_col = f"ang_uvw_{axis}_deg"
    have_hkl = (uvw_col in df.columns) and (ang_col in df.columns)
    if have_hkl:
        uvw_arr = df[uvw_col].to_numpy()
        ang_arr = df[ang_col].to_numpy(float)

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    annot.set_visible(False)

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        cx, cy = event.xdata, event.ydata
        dx = x - cx
        dy = y - cy
        dist2 = dx*dx + dy*dy
        idx = int(np.argmin(dist2))

        px, py = x[idx], y[idx]
        p_dir  = d_axis[idx]
        p_phi1 = phi1_arr[idx]
        p_PHI  = PHI_arr[idx]
        p_phi2 = phi2_arr[idx]

        lines = [
            f"Index: {idx}",
            f"{x_col}={px:.1f}, {y_col}={py:.1f}",
            "Euler (deg):",
            f"  φ1={p_phi1:.2f}, Φ={p_PHI:.2f}, φ2={p_phi2:.2f}",
            f"{axis}-axis (crystal frame):",
            f"  n{axis} = [{p_dir[0]:+.3f}, {p_dir[1]:+.3f}, {p_dir[2]:+.3f}]",
        ]
        if have_hkl:
            lines.append(f"{axis} orientation: {uvw_arr[idx]}")
            lines.append(f"  angle(n{axis}, hkl) = {ang_arr[idx]:.3f}°")

        text = "\n".join(lines)
        annot.xy = (px, py)
        annot.set_text(text)
        annot.set_visible(True)
        fig.canvas.draw_idle()

        print("="*60)
        print(text)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    return fig, ax


def select_footprint(df, x0, y0,
                     shape="ellipse",
                     rx=20.0, ry=None,
                     width=None, height=None,
                     angle_deg=0.0,
                     x_col="X", y_col="Y",
                     scale_x=1.0, scale_y=1.0,
                     return_mask=False):
    """
    Select EBSD datapoints inside a footprint, assuming df is already loaded.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain x_col, y_col.
    x0, y0 : float
        Footprint center in the SAME units as df[x_col], df[y_col] after scaling.
    shape : {"circle","ellipse","rect","rectangle"}
    rx, ry : float
        For circle: radius=rx.
        For ellipse: semi-axes rx, ry (if ry is None -> ry=rx).
    width, height : float
        For rect: full width/height. If None, uses 2*rx, 2*ry.
    angle_deg : float
        Rotation of ellipse/rect in EBSD map coords (CCW, degrees).
    x_col, y_col : str
        Column names for map coordinates.
    scale_x, scale_y : float
        Multiply df coords by these factors BEFORE selection.
        Use if df coords are in pixels/steps but x0,y0,rx,ry are in µm (or vice versa).
    return_mask : bool
        If True, return (df_subset, mask).

    Returns
    -------
    df_subset : pd.DataFrame
        Filtered points.
    (optional) mask : np.ndarray[bool]
        Boolean mask over df rows.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"df must contain columns '{x_col}' and '{y_col}'")

    X = df[x_col].to_numpy(dtype=float) * float(scale_x)
    Y = df[y_col].to_numpy(dtype=float) * float(scale_y)

    dx = X - float(x0)
    dy = Y - float(y0)

    shape_l = shape.lower()

    if shape_l == "circle":
        r = float(rx)
        mask = (dx*dx + dy*dy) <= r*r
        out = df.loc[mask].copy()
        return (out, mask) if return_mask else out

    # rotate coords for ellipse/rect
    th = np.deg2rad(float(angle_deg))
    c, s = np.cos(th), np.sin(th)
    # rotate by +th: put footprint axes into xr/yr
    xr =  c*dx + s*dy
    yr = -s*dx + c*dy

    if shape_l == "ellipse":
        if ry is None:
            ry = rx
        rxv = float(rx)
        ryv = float(ry)
        mask = (xr*xr)/(rxv*rxv) + (yr*yr)/(ryv*ryv) <= 1.0
        out = df.loc[mask].copy()
        return (out, mask) if return_mask else out

    if shape_l in ("rect", "rectangle"):
        if width is None:
            width = 2.0*float(rx)
        if height is None:
            height = 2.0*float(ry if ry is not None else rx)
        hw = float(width) / 2.0
        hh = float(height) / 2.0
        mask = (np.abs(xr) <= hw) & (np.abs(yr) <= hh)
        out = df.loc[mask].copy()
        return (out, mask) if return_mask else out

    raise ValueError(f"Unknown shape={shape!r}")


def read_ebsd_region(df, x0, y0, **footprint_kwargs):
    """
    Backwards compatible name: previously read from file, now assumes df is already loaded.
    """
    return select_footprint(df, x0, y0, **footprint_kwargs)


def _order_clockwise(pts):
    """Return pts ordered clockwise around centroid."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def _min_area_rect(pts):
    """
    Minimum area bounding rectangle corners for 2D points.
    Returns 4 corners (unordered).
    """
    hull = ConvexHull(pts)
    hp = pts[hull.vertices]

    # edge angles
    edges = np.diff(np.vstack([hp, hp[0]]), axis=0)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.mod(angles, np.pi / 2.0)
    angles = np.unique(np.round(angles, 12))

    best_area = np.inf
    best_rect = None
    best_R = None

    for a in angles:
        ca, sa = np.cos(a), np.sin(a)
        R = np.array([[ca, -sa],
                      [sa,  ca]])
        rp = hp @ R  # rotate points

        xmin, ymin = rp.min(axis=0)
        xmax, ymax = rp.max(axis=0)
        area = (xmax - xmin) * (ymax - ymin)

        if area < best_area:
            best_area = area
            best_R = R
            best_rect = np.array([
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ])

    # rotate rectangle back
    rect = best_rect @ best_R.T
    return rect


def fit_affine(src, dst):
    """
    Fit affine transform dst ≈ src @ A.T + t
    src,dst: (N,2)
    Returns A(2,2), t(2,)
    """
    N = src.shape[0]
    M = np.zeros((2*N, 6), float)
    b = dst.reshape(-1)

    M[0::2, 0:2] = src
    M[0::2, 4] = 1.0
    M[1::2, 2:4] = src
    M[1::2, 5] = 1.0

    p, *_ = np.linalg.lstsq(M, b, rcond=None)
    A = np.array([[p[0], p[1]],
                  [p[2], p[3]]])
    t = np.array([p[4], p[5]])
    return A, t


def apply_affine(xy, A, t):
    """Apply an affine transform xy @ A.T + t to coordinates."""
    xy = np.asarray(xy, float)
    return xy @ A.T + t


def invert_affine(A, t):
    """Return the inverse affine transform parameters."""
    Ai = np.linalg.inv(A)
    # ti = -Ai @ t
    ti = -t @ Ai.T
    return Ai, ti


def best_corner_pairing(src4, dst4):
    """
    src4: 4 corners in EBSD
    dst4: 4 corners in mm
    Tries cyclic shifts and reversed order, returns best (A,t,src_ordered,dst_ordered).
    """
    src4 = _order_clockwise(src4)
    dst4 = _order_clockwise(dst4)

    best = None
    best_err = np.inf

    for rev in [False, True]:
        s = src4[::-1] if rev else src4.copy()
        for k in range(4):
            s_k = np.roll(s, shift=k, axis=0)
            A, t = fit_affine(s_k, dst4)
            pred = apply_affine(s_k, A, t)
            err = np.sqrt(np.mean(np.sum((pred - dst4)**2, axis=1)))
            if err < best_err:
                best_err = err
                best = (A, t, s_k, dst4)

    return best_err, best


def calibrate_ebsd_to_mm(df, mm_corners, x_col="X", y_col="Y", sample_n=200000):
    """
    df: EBSD dataframe
    mm_corners: (4,2) array, known scan corners in mm
    Returns: A,t and inverse transform, plus EBSD corners used.
    """
    # optional downsample for speed (rectangle uses hull; can be heavy on 1e6 pts)
    XY = df[[x_col, y_col]].to_numpy(float)
    if len(XY) > sample_n:
        idx = np.random.choice(len(XY), size=sample_n, replace=False)
        XY = XY[idx]

    ebsd_rect = _min_area_rect(XY)

    err, (A, t, ebsd_corners_ord, mm_corners_ord) = best_corner_pairing(ebsd_rect, np.asarray(mm_corners, float))
    Ai, ti = invert_affine(A, t)

    return {
        "A_ebsd_to_mm": A,
        "t_ebsd_to_mm": t,
        "A_mm_to_ebsd": Ai,
        "t_mm_to_ebsd": ti,
        "ebsd_corners": ebsd_corners_ord,
        "mm_corners": mm_corners_ord,
        "rmse_mm": err,
    }
