from __future__ import annotations
from math import gcd
import os
import re
import numpy as np
import pandas as pd


from .polycrystal import (
    angle_deg,
    canonicalize_uvw,
    nearest_small_dir,
)

__all__ = [
    "parse_spt_table",
    "group_peaks_by_frame",
    "unit",
    "get_spot_id",
    "get_xy_from_peak",
    "best_small_index_ub",
    "round_reduce",
    "B_from_cell",
    "summarize_grain_axes",
    "build_multigrain_table",
    "floats_in_line",
    "next_numeric_line",
    "next_numeric_rows",
    "parse_grainspotter_log",
    "build_reflections_table",
    "build_spot_coord_table",
]


def parse_spt_table(spt_path):
    """
    Parse a flat ImageD11 ``.spt`` or ``.flt`` peak table.

    Parameters
    ----------
    spt_path : str or path
        Path to an ImageD11 text table.  The header may be commented with
        ``#`` and is expected to contain ``omega`` plus peak coordinate or
        peak identifier columns.

    Returns
    -------
    peaks : list of dict
        One dictionary per parsed data row.  Header names are converted to
        lowercase and numeric values are returned as floats.

    Raises
    ------
    ValueError
        If no recognizable peak table header is found.
    """
    peaks = []
    header = None

    def _header_candidate(text):
        fields = [field.lower() for field in re.split(r"\s+", text.strip())]
        known = {
            "omega", "sc", "fc", "s_raw", "f_raw", "dety", "detz",
            "spot3d_id", "peak_id", "number_of_pixels", "avg_intensity",
        }
        if "omega" in fields and len(known.intersection(fields)) >= 2:
            return fields
        return None

    with open(spt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                cand = line.lstrip("#").strip()
                if header is None:
                    header = _header_candidate(cand)
                continue

            if header is None and re.search(r"[A-Za-z]", line):
                header = _header_candidate(line)
                continue

            if header is not None:
                parts = re.split(r"\s+", line)
                if len(parts) < len(header):
                    continue
                parts = parts[:len(header)]
                row = {}
                for k, v in zip(header, parts):
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
                peaks.append(row)

    if header is None:
        raise ValueError(f"No ImageD11 peak-table header found in {spt_path!s}.")
    return peaks


def group_peaks_by_frame(peaks, omegas, tol_deg=0.5):
    """Associate peaks with the nearest measured detector frame angle.

    Parameters
    ----------
    peaks : iterable of mapping
        Peak records such as those returned by :func:`parse_spt_table`.  A
        record without a finite ``omega`` value is skipped.
    omegas : array, shape (n_frames,)
        Measured frame angles in degrees.  Values need not be sorted.
    tol_deg : float, default=0.5
        Maximum absolute angular difference, in degrees, for accepting a
        peak to frame assignment.

    Returns
    -------
    grouped : dict
        Mapping ``frame_index -> list[peak]``.  Frames without accepted peaks
        are omitted.

    Raises
    ------
    ValueError
        If ``omegas`` is empty, not one dimensional, contains non-finite
        values, or if ``tol_deg`` is negative or non-finite.
    """
    omegas = np.asarray(omegas, dtype=float)
    if omegas.ndim != 1 or omegas.size == 0:
        raise ValueError("omegas must be a non-empty one dimensional array.")
    if not np.all(np.isfinite(omegas)):
        raise ValueError("omegas must contain only finite values.")
    if not np.isfinite(tol_deg) or tol_deg < 0:
        raise ValueError("tol_deg must be a finite non-negative number.")

    out = {}
    for p in peaks:
        if "omega" not in p:
            continue
        try:
            om = float(p["omega"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(om):
            continue
        idx = int(np.argmin(np.abs(omegas - om)))
        if abs(omegas[idx] - om) <= tol_deg:
            out.setdefault(idx, []).append(p)
    return out

_AXES = {
    'x': np.array([1.,0.,0.]),
    'y': np.array([0.,1.,0.]),
    'z': np.array([0.,0.,1.]),
}


def unit(v):
    """Return a finite three component vector normalized to unit length.

    Parameters
    ----------
    v : array, shape (3,)
        Vector to normalize.

    Returns
    -------
    unit_vector : ndarray, shape (3,)
        Normalized vector.

    Raises
    ------
    ValueError
        If the input is not a finite three component vector or has zero
        length.
    """
    v = np.asarray(v, float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError("v must be a finite vector with shape (3,).")
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero length vector.")
    return v / n


def get_spot_id(p):
    """Extract the identifier that corresponds to GrainSpotter ``peak_id``.

    Parameters
    ----------
    p : mapping
        ImageD11 peak record.

    Returns
    -------
    peak_id : int or None
        First valid identifier found under ``peak_id``, ``spot3d_id``, or
        ``id``; otherwise ``None``.
    """
    for key in ('peak_id', 'spot3d_id', 'id'):
        if key in p:
            try:
                return int(p[key])
            except (TypeError, ValueError, OverflowError):
                pass
    return None


def get_xy_from_peak(p):
    """Return detector ``(x, y)`` coordinates from a peak record.

    Coordinate pairs are tried in this order: ``(fc, sc)``,
    ``(f_raw, s_raw)``, ``(detz, dety)``, then ``(f, s)``.  This matches the
    plotting convention where ``x`` is the image column and ``y`` is the image
    row.

    Parameters
    ----------
    p : mapping
        ImageD11 peak record.

    Returns
    -------
    x, y : tuple
        Coordinate values, or ``(None, None)`` when no supported pair exists.
    """
    if 'fc' in p and 'sc' in p:
        return p['fc'], p['sc']
    if 'f_raw' in p and 's_raw' in p:
        return p['f_raw'], p['s_raw']
    if 'detz' in p and 'dety' in p:
        return p['detz'], p['dety']
    if 'f' in p and 's' in p:
        return p['f'], p['s']
    return None, None


def best_small_index_ub(n, maxh=6):
    """Find the compact integer triplet nearest to a fractional direction.

    Parameters
    ----------
    n : array, shape (3,)
        Direction expressed in fractional reciprocal coordinates.
    maxh : int, default=6
        Inclusive search limit for each of ``h``, ``k``, and ``l``.

    Returns
    -------
    angle : float
        Smallest angular separation in degrees, treating opposite directions
        as equivalent.
    hkl : tuple of int
        Canonical integer triplet whose first non-zero component is positive.

    Raises
    ------
    ValueError
        If ``n`` is invalid or zero, or if ``maxh`` is smaller than one.
    """
    n = unit(n)
    if not isinstance(maxh, (int, np.integer)) or maxh < 1:
        raise ValueError("maxh must be an integer greater than or equal to 1.")

    best = (999.0, None)
    for h in range(-maxh, maxh+1):
        for k in range(-maxh, maxh+1):
            for l in range(-maxh, maxh+1):
                if h == k == l == 0:
                    continue
                hkl = canonicalize_uvw(h, k, l)
                if hkl != (h, k, l):
                    continue
                divisor = gcd(gcd(abs(h), abs(k)), abs(l))
                if divisor != 1:
                    continue
                v = np.array(hkl, float)
                v /= np.linalg.norm(v)
                cosine = np.clip(abs(float(np.dot(n, v))), -1.0, 1.0)
                ang = float(np.degrees(np.arccos(cosine)))
                if ang < best[0]:
                    best = (ang, hkl)
    return best


def round_reduce(vec, max_mult=6, tol=0.03):
    """Approximate a direction by a reduced integer triplet.

    Parameters
    ----------
    vec : array, shape (3,)
        Fractional vector to approximate.
    max_mult : int, default=6
        Largest positive scale factor tested before rounding.
    tol : float, default=0.03
        Stop once ``norm(m * vec - round(m * vec))`` is below this value.

    Returns
    -------
    triplet : tuple of int
        Canonical, greatest common divisor reduced integer triplet.
    residual : float
        Best rounding residual divided by the corresponding multiplier.

    Raises
    ------
    ValueError
        If the vector is invalid or zero, ``max_mult`` is smaller than one,
        or ``tol`` is negative or non-finite.
    """
    v = np.asarray(vec, float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError("vec must be a finite vector with shape (3,).")
    if np.linalg.norm(v) == 0:
        raise ValueError("vec must be non-zero.")
    if not isinstance(max_mult, (int, np.integer)) or max_mult < 1:
        raise ValueError("max_mult must be an integer greater than or equal to 1.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite non-negative number.")

    best = None
    for m in range(1, max_mult+1):
        r = np.rint(m * v)
        res = np.linalg.norm(m * v - r)
        if best is None or res < best[1]:
            best = (r, res, m)
        if res < tol:
            break
    r, res, m = best
    r = canonicalize_uvw(*(int(value) for value in r))
    g = max(1, gcd(gcd(abs(r[0]), abs(r[1])), abs(r[2])))
    r = (r[0]//g, r[1]//g, r[2]//g)
    return r, float(res/m)


def B_from_cell(a, b, c, alpha_deg, beta_deg, gamma_deg, two_pi=False):
    """Construct the reciprocal basis matrix for a triclinic unit cell.

    Parameters
    ----------
    a, b, c : float
        Positive direct lattice lengths.
    alpha_deg, beta_deg, gamma_deg : float
        Direct cell angles in degrees.  ``alpha`` is between ``b`` and ``c``,
        ``beta`` between ``a`` and ``c``, and ``gamma`` between ``a`` and ``b``.
    two_pi : bool, default=False
        Multiply reciprocal vectors by :math:`2π` when true.  The default
        follows the ImageD11 convention used by this workflow.

    Returns
    -------
    B : ndarray, shape (3, 3)
        Matrix satisfying ``g_cart = B @ [h, k, l]``.  Its columns are the
        reciprocal vectors ``a*``, ``b*``, and ``c*``.

    Raises
    ------
    ValueError
        If cell lengths or angles are invalid, or if the cell is singular.
    """
    lengths = np.asarray([a, b, c], dtype=float)
    angles = np.asarray([alpha_deg, beta_deg, gamma_deg], dtype=float)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("a, b, and c must be finite positive values.")
    if not np.all(np.isfinite(angles)) or np.any((angles <= 0) | (angles >= 180)):
        raise ValueError("Cell angles must be finite values strictly between 0 and 180 degrees.")

    α = np.deg2rad(alpha_deg); β = np.deg2rad(beta_deg); γ = np.deg2rad(gamma_deg)
    cosα, cosβ, cosγ = np.cos(α), np.cos(β), np.cos(γ)
    sinα, sinβ, sinγ = np.sin(α), np.sin(β), np.sin(γ)

    # Direct basis in Cartesian coords
    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b*cosγ, b*sinγ, 0.0])
    # c_y chosen so that angle(c,b)=α; c_x so that angle(c,a)=β
    c_x = c*cosβ
    if abs(sinγ) < 1e-14:
        raise ValueError("gamma defines a singular direct-cell basis.")
    c_y = c*(cosα - cosβ*cosγ)/sinγ
    radicand = 1.0 - cosβ**2 - ((cosα - cosβ*cosγ)/sinγ)**2
    if radicand < -1e-12:
        raise ValueError("The supplied lengths and angles do not define a valid unit cell.")
    c_z = c*np.sqrt(max(0.0, radicand))
    c_vec = np.array([c_x, c_y, c_z])

    # Reciprocal basis (no 2π)
    V = np.dot(a_vec, np.cross(b_vec, c_vec))
    if not np.isfinite(V) or abs(V) < 1e-14:
        raise ValueError("The supplied unit cell is singular.")
    a_star = np.cross(b_vec, c_vec) / V
    b_star = np.cross(c_vec, a_vec) / V
    c_star = np.cross(a_vec, b_vec) / V

    if two_pi:
        twopi = 2.0*np.pi
        a_star, b_star, c_star = twopi*a_star, twopi*b_star, twopi*c_star

    # Columns are reciprocal basis vectors
    return np.column_stack([a_star, b_star, c_star])


def summarize_grain_axes(grain,
                         axes=('x','y','z'),
                         max_index:int=6, 
                         a = 3.3004,
                         new_order_indices = (0, 1, 2)):
    """Summarize one cubic GrainSpotter orientation along specimen axes.

    Parameters
    ----------
    grain : mapping
        Grain dictionary returned by :func:`parse_grainspotter_log`.  It must
        contain ``U``, ``UBI``, ``grain_id``, ``euler_bunge``, ``mean_IA``,
        and ``position``.
    axes : sequence of {"x", "y", "z"}, default=("x", "y", "z")
        Specimen axes for which rows are generated.
    max_index : int, default=6
        Maximum absolute Miller index considered by the compact index search.
    a : float, default=3.3004
        Cubic lattice parameter.
    new_order_indices : sequence of int, default=(0, 1, 2)
        Permutation applied to the historical display columns.  The unmodified
        crystallographic values are also returned in columns ending in
        ``_raw``.

    Returns
    -------
    rows : list of dict
        One record per requested specimen axis.  Records contain Bunge Euler
        angles, grain position, compact ``[uvw]`` directions, compact ``(hkl)``
        normals, and angular back checks in degrees.

    Raises
    ------
    KeyError
        If a required grain field is absent.
    ValueError
        If the matrices, axes, lattice parameter, search limit, or component
        permutation are invalid, or if ``U`` and ``UBI`` are inconsistent.
    """
    required = {"U", "UBI", "grain_id", "euler_bunge", "mean_IA", "position"}
    missing = sorted(required.difference(grain))
    if missing:
        raise KeyError(f"grain is missing required fields: {', '.join(missing)}")

    axes = tuple(str(ax).lower() for ax in axes)
    invalid_axes = [ax for ax in axes if ax not in _AXES]
    if invalid_axes:
        raise ValueError(f"Unknown specimen axes: {invalid_axes!r}.")
    if not isinstance(max_index, (int, np.integer)) or max_index < 1:
        raise ValueError("max_index must be an integer greater than or equal to 1.")
    if not np.isfinite(a) or a <= 0:
        raise ValueError("a must be a finite positive cubic lattice parameter.")

    perm = np.asarray(new_order_indices, dtype=int)
    if perm.shape != (3,) or sorted(perm.tolist()) != [0, 1, 2]:
        raise ValueError("new_order_indices must be a permutation of (0, 1, 2).")

    U   = np.asarray(grain['U'], float)
    UBI = np.asarray(grain['UBI'], float)
    if U.shape != (3, 3) or UBI.shape != (3, 3):
        raise ValueError("grain['U'] and grain['UBI'] must both have shape (3, 3).")
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(UBI)):
        raise ValueError("grain['U'] and grain['UBI'] must contain finite values.")
    try:
        UB = np.linalg.inv(UBI)
    except np.linalg.LinAlgError as exc:
        raise ValueError("grain['UBI'] is singular.") from exc
    
    B = B_from_cell(a, a, a, 90, 90, 90, two_pi=False)
    U_from_UBI = np.linalg.inv(UBI) @ np.linalg.inv(B)
    if not np.allclose(U_from_UBI, U, atol=5e-4, rtol=5e-4):
        difference = float(np.max(np.abs(U_from_UBI - U)))
        raise ValueError(
            "grain['U'] is inconsistent with grain['UBI'] and the supplied "
            f"cubic lattice parameter; maximum difference is {difference:.3g}."
        )

    phi1, PHI, phi2 = (float(value) for value in grain['euler_bunge'])
    direction_by_axis = {
        label: unit(U.T @ specimen_axis)
        for label, specimen_axis in _AXES.items()
    }
    nearest_by_axis = {
        label: nearest_small_dir(direction, maxu=max_index)
        for label, direction in direction_by_axis.items()
    }
    
    rows = []
    for ax in axes:
        e = _AXES[ax]
        
        plane_normal_frac = UBI @ e  # fractional (hkl) vector
        direction_crys_raw = direction_by_axis[ax]
        hkl_from_plane_raw, plane_reduce_resid = round_reduce(
            plane_normal_frac, max_mult=max_index, tol=0.03
        )
        ang_plane_vs_hkl_reduce = angle_deg(plane_normal_frac, hkl_from_plane_raw, plane_equiv=True)
        ang_plane_vs_ub_hkl_raw, hkl_best_ub_raw = best_small_index_ub(plane_normal_frac, maxh=max_index)
        hkl_round_raw = canonicalize_uvw(*hkl_from_plane_raw)
        hkl_round_resid = plane_reduce_resid
         n_lab_from_h = UB @ hkl_best_ub_raw
        ang_axis_vs_ub_plane = angle_deg(n_lab_from_h, e, plane_equiv=True)
        ang_frac_vs_ub_hkl   = angle_deg(plane_normal_frac, hkl_best_ub_raw, plane_equiv=True)
        
        # Compact directions from the GrainSpotter U matrix.
        angX_m, uvwX_raw = nearest_by_axis['x']
        angY_m, uvwY_raw = nearest_by_axis['y']
        angZ_m, uvwZ_raw = nearest_by_axis['z']

        # angle between Euler axis dir and UB
        axis_to_uvw_raw = {'x': uvwX_raw, 'y': uvwY_raw, 'z': uvwZ_raw}
        ang_euler_axis_vs_ub_plane = angle_deg(axis_to_uvw_raw[ax], hkl_best_ub_raw, plane_equiv=True)
        direction_crys = tuple(direction_crys_raw[perm])
        plane_normal_frac_perm = tuple(plane_normal_frac[perm])
        hkl_from_plane = tuple(np.array(hkl_from_plane_raw)[perm])
        hkl_round      = tuple(np.array(hkl_round_raw)[perm])
        ub_hkl         = tuple(np.array(canonicalize_uvw(*hkl_best_ub_raw))[perm])

        uvwX_m = tuple(np.array(uvwX_raw)[perm])
        uvwY_m = tuple(np.array(uvwY_raw)[perm])
        uvwZ_m = tuple(np.array(uvwZ_raw)[perm])

        rows.append(dict(
            grain_id=grain['grain_id'],
            axis=ax,
            phi1=phi1, PHI=PHI, phi2=phi2,
            mean_IA=grain['mean_IA'],
            pos_x=grain['position'][0], pos_y=grain['position'][1], pos_z=grain['position'][2],
                     
            crys_dir_axis = direction_crys,
            crys_dir_axis_raw = tuple(direction_crys_raw),

            hkl_from_plane_reduce = hkl_from_plane, 
            ang_plane_vs_hkl_reduce_deg = ang_plane_vs_hkl_reduce,

            ub_hkl = ub_hkl,
            ub_hkl_raw = tuple(canonicalize_uvw(*hkl_best_ub_raw)),
            ang_plane_vs_ub_hkl_deg = ang_plane_vs_ub_hkl_raw,
            ang_norm_raw_vs_ub_hkl = ang_plane_vs_ub_hkl_raw, 
            
            h_k_l_back_check = hkl_round, 
            hkl_resid_back_check = hkl_round_resid,
            plane_normal_frac = plane_normal_frac_perm,
            plane_normal_frac_raw = tuple(plane_normal_frac),
            
            ang_axis_vs_ub_plane_deg = ang_axis_vs_ub_plane, # angle between the specimen axis e and the lab normal UB @ (hkl)
            ang_frac_vs_ub_hkl_deg   = ang_frac_vs_ub_hkl, # angle between fractional normal UBI @ e and its nearest small integer (hkl) in reciprocal space
            
            euler_axisX_mis_deg = angX_m,
            euler_axisY_mis_deg = angY_m,
            euler_axisZ_mis_deg = angZ_m, 
            euler_axisX_uvw = uvwX_m,
            euler_axisY_uvw = uvwY_m,
            euler_axisZ_uvw = uvwZ_m,
            
            # IMPORTANT: this angle is computed in the same frame as hkl_best_ub_raw
            ang_euler_axis_vs_ub_plane_deg = ang_euler_axis_vs_ub_plane,
        ))
    return rows


def build_multigrain_table(log_path,
                           axes=('x','y','z'),
                           max_index=6, 
                           order_indices = (0, 1, 2),
                           a=3.3004):
    """Build an orientation summary table for every parsed grain.

    Parameters
    ----------
    log_path : str or path
        GrainSpotter ``.log`` file.
    axes : sequence of {"x", "y", "z"}, default=("x", "y", "z")
        Specimen axes to report.
    max_index : int, default=6
        Maximum absolute index used by compact ``[uvw]`` and ``(hkl)``
        searches.
    order_indices : sequence of int, default=(0, 1, 2)
        Display component permutation passed to
        :func:`summarize_grain_axes`.
    a : float, default=3.3004
        Cubic lattice parameter passed to :func:`summarize_grain_axes`.

    Returns
    -------
    table : pandas.DataFrame
        Table indexed by ``(grain_id, axis)``.  An empty table is returned
        when the parsed log contains no grain summaries.

    Raises
    ------
    FileNotFoundError
        If ``log_path`` does not exist.
    ValueError
        If parsing or orientation validation fails.

    Notes
    -----
    The orientation summary assumes a cubic unit cell.
    """
    data = parse_grainspotter_log(log_path)
    all_rows = []
    for g in data['grains']:
        summary_ = summarize_grain_axes(
            g,
            axes=axes,
            max_index=max_index,
            a=a,
            new_order_indices=order_indices,
        )
        if summary_:
            all_rows.extend(summary_)
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows).set_index(['grain_id','axis']).sort_index()


_float_pat = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?')


def _float_token(value):
    """Convert a decimal token that may use a Fortran exponent."""
    return float(value.replace('D', 'E').replace('d', 'e'))


def floats_in_line(s):
    """Extract decimal values, including Fortran ``D`` exponents, from text.

    Parameters
    ----------
    s : str
        Input line.

    Returns
    -------
    values : list of float
        Numeric tokens in their original order.
    """
    return [_float_token(token) for token in _float_pat.findall(s)]


def next_numeric_line(lines, start_idx, expected):
    """Find the next line containing at least ``expected`` numeric values.

    Parameters
    ----------
    lines : sequence of str
        Full text split into lines.
    start_idx : int
        Zero line index after which the search starts.
    expected : int
        Minimum number of numeric values required.

    Returns
    -------
    values : list of float
        First ``expected`` values from the matched line.
    line_index : int
        Zero index of the matched line.

    Raises
    ------
    ValueError
        If ``expected`` is invalid or no suitable line is found.
    """
    if not isinstance(expected, (int, np.integer)) or expected < 1:
        raise ValueError("expected must be an integer greater than or equal to 1.")
    i = start_idx + 1
    while i < len(lines):
        nums = floats_in_line(lines[i])
        if len(nums) >= expected:
            return nums[:expected], i
        i += 1
    raise ValueError(f"Expected a line with >= {expected} floats after line {start_idx}")


def next_numeric_rows(lines, start_idx, nrows, min_per_row = 3):
    """Collect the next numeric rows from a GrainSpotter text block.

    Parameters
    ----------
    lines : sequence of str
        Full text split into lines.
    start_idx : int
        Zero line index after which the search starts.
    nrows : int
        Number of rows to collect.
    min_per_row : int, default=3
        Minimum numeric values required in each accepted row.  Only the first
        ``min_per_row`` values are returned.

    Returns
    -------
    rows : ndarray, shape (nrows, min_per_row)
        Parsed numeric block.
    last_line_index : int
        Zero index of the final accepted line.

    Raises
    ------
    ValueError
        If row counts are invalid or not enough numeric rows are found.
    """
    if not isinstance(nrows, (int, np.integer)) or nrows < 1:
        raise ValueError("nrows must be an integer greater than or equal to 1.")
    if not isinstance(min_per_row, (int, np.integer)) or min_per_row < 1:
        raise ValueError("min_per_row must be an integer greater than or equal to 1.")
    rows = []
    i = start_idx + 1
    while i < len(lines) and len(rows) < nrows:
        nums = floats_in_line(lines[i])
        if len(nums) >= min_per_row:
            rows.append(nums[:min_per_row])
        i += 1
    if len(rows) != nrows:
        raise ValueError(f"Could not find {nrows} numeric rows after line {start_idx}: got {len(rows)}")
    return np.array(rows, float), i - 1


def parse_grainspotter_log(path):
    """Parse GrainSpotter grain, orientation, and reflection records.

    Parameters
    ----------
    path : str or path
        GrainSpotter output log containing blocks headed by lines such as
        ``Grain    1, 36``.

    Returns
    -------
    result : dict
        Dictionary with key ``"grains"``.  Each grain record contains counts,
        mean internal angle, position, ``U`` and ``UBI`` matrices, Rodrigues
        vector, Bunge Euler angles, quaternion, a reflection DataFrame, and a
        list of any unparsed reflection lines.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If no grain blocks are found or a required numeric block is missing.

    Notes
    -----
    Reflection columns follow GrainSpotter 0.90 output. 
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\n') for ln in f]

    grains = []
    pat_grain = re.compile(r'^\s*Grain\s+(\d+)\s*,\s*(\d+)\s*$')

    i = 0
    while i < len(lines):
        m = pat_grain.match(lines[i])
        if not m:
            i += 1
            continue

        grain_id = int(m.group(1))
        grain_header_count = int(m.group(2))
        counts, i = next_numeric_line(lines, i, expected=4)
        n_expected, n_measured, n_once, n_more = [int(round(x)) for x in counts[:4]]

        five, i = next_numeric_line(lines, i, expected=5)
        mean_IA, px, py, pz, pos_chisq = five

        U, i   = next_numeric_rows(lines, i, nrows=3, min_per_row=3)
        UBI, i = next_numeric_rows(lines, i, nrows=3, min_per_row=3)

        r, i   = next_numeric_line(lines, i, expected=3)
        eul,i  = next_numeric_line(lines, i, expected=3)
        quat,i = next_numeric_line(lines, i, expected=4)

        refl_rows = []
        unparsed_reflections = []
        i += 1
        while i < len(lines):
            s = lines[i].strip()
            if pat_grain.match(lines[i]) or s.startswith('In total'):
                break
            if re.match(r'^\d+\s', s):
                parts = s.split()
                try:
                    (idx, gvid, pid,
                     h, k, l,
                     hpr, kpr, lpr,
                     dh, dk, dl,
                     tth_m, tth_p, dtth,
                     om_m, om_p, dom,
                     eta_m, eta_p, deta,
                     IA) = parts[:22]
                    row = dict(
                        row_index=int(idx),
                        gvector_id=int(gvid),
                        peak_id=int(pid),
                        h=int(h), k=int(k), l=int(l),
                        h_pred=_float_token(hpr), k_pred=_float_token(kpr), l_pred=_float_token(lpr),
                        dh=_float_token(dh), dk=_float_token(dk), dl=_float_token(dl),
                        tth_meas=_float_token(tth_m), tth_pred=_float_token(tth_p), dtth=_float_token(dtth),
                        omega_meas=_float_token(om_m), omega_pred=_float_token(om_p), domega=_float_token(dom),
                        eta_meas=_float_token(eta_m), eta_pred=_float_token(eta_p), deta=_float_token(deta),
                        IA=_float_token(IA)
                    )
                except (TypeError, ValueError):
                    unparsed_reflections.append(s)
                else:
                    refl_rows.append(row)
            i += 1

        reflections = pd.DataFrame(refl_rows)

        grains.append(dict(
            grain_id=grain_id,
            grain_header_count=grain_header_count,
            n_expected=n_expected, n_measured=n_measured, n_once=n_once, n_more=n_more,
            mean_IA=mean_IA, position=(px,py,pz), pos_chisq=pos_chisq,
            U=U, UBI=UBI,
            rodrigues=tuple(r),
            euler_bunge=tuple(eul),
            quat=tuple(quat),
            reflections=reflections,
            unparsed_reflections=unparsed_reflections,
        ))

    if not grains:
        raise ValueError("No grain blocks found. Ensure lines like 'Grain    1, 36' exist.")
    return {'grains': grains}



def build_reflections_table(log_path):
    """Create a flat indexed reflection table from a GrainSpotter log.

    Parameters
    ----------
    log_path : str or path
        GrainSpotter output log.

    Returns
    -------
    reflections : pandas.DataFrame
        One row per assigned reflection, indexed by ``(grain_id, peak_id)``.
        Columns include measured and predicted ``hkl``, two theta, omega, eta,
        residuals, and internal angle.

    Raises
    ------
    FileNotFoundError
        If ``log_path`` does not exist.
    ValueError
        If no valid reflection rows are found or required identifier columns
        are absent.
    """
    data = parse_grainspotter_log(log_path)
    rows = []
    for g in data['grains']:
        grain_id = g['grain_id']
        refl = g['reflections']
        if isinstance(refl, pd.DataFrame) and not refl.empty:
            df = refl.copy()
            df['grain_id'] = grain_id
            rows.append(df)
    if not rows:
        raise ValueError("No reflections found in log.")
    df_all = pd.concat(rows, ignore_index=True)
    # enforce numeric types where appropriate
    numeric_cols = [
        'row_index', 'gvector_id', 'peak_id',
        'h','k','l','h_pred','k_pred','l_pred',
        'dh','dk','dl',
        'tth_meas','tth_pred','dtth',
        'omega_meas','omega_pred','domega',
        'eta_meas','eta_pred','deta',
        'IA'
    ]
    for c in numeric_cols:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors='coerce')
    required = {"grain_id", "peak_id"}
    missing = sorted(required.difference(df_all.columns))
    if missing:
        raise ValueError(f"Reflection table is missing columns: {', '.join(missing)}")

    df_all = df_all.set_index(['grain_id','peak_id']).sort_index()
    return df_all




def build_spot_coord_table(peaks_by_frame):
    """Flatten frame peaks into detector coordinates by peak ID.

    Parameters
    ----------
    peaks_by_frame : mapping
        Mapping ``frame_index -> iterable[peak]``, normally returned by
        :func:`group_peaks_by_frame`.

    Returns
    -------
    spots : pandas.DataFrame
        Table indexed by ``peak_id`` with columns ``frame``, ``x``, ``y``,
        ``npix``, and ``avg_intensity``.  The index is suitable for joining
        with :func:`build_reflections_table`.

    Raises
    ------
    ValueError
        If no peak has both a usable identifier and supported detector
        coordinates.
    """
    rows = []
    for frame_idx, peaks in peaks_by_frame.items():
        for p in peaks:
            pid = get_spot_id(p)
            if pid is None:
                continue
            x, y = get_xy_from_peak(p)
            if x is None or y is None:
                continue
            try:
                x = float(x)
                y = float(y)
                npix = float(p.get('number_of_pixels', p.get('npix', 0.0)))
                avg_intensity = float(
                    p.get('avg_intensity', p.get('average_counts', 0.0))
                )
            except (TypeError, ValueError):
                continue
            if not np.all(np.isfinite([x, y, npix, avg_intensity])):
                continue
            rows.append(dict(
                frame=int(frame_idx),
                peak_id=int(pid),
                x=x,
                y=y,
                npix=npix,
                avg_intensity=avg_intensity,
            ))
    if not rows:
        raise ValueError("No usable peaks found in peaks_by_frame.")
    spot_df = pd.DataFrame(rows).drop_duplicates(subset=['peak_id'])
    spot_df = spot_df.set_index('peak_id').sort_index()
    return spot_df
