from concurrent.futures import ProcessPoolExecutor
from math import gcd
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

__all__ = [
    "unit",
    "angle_from_cos_deg",
    "canonicalize_uvw",
    "float_hkl_from_n_via_nearest_int",
    "nearest_small_dir_all_axes_stream",
    "uvw_to_str",
    "plot_nearest_small_dir_results",
    "nearest_small_dir",
    "generate_uvw_set",
    "angle_deg",
    "add_axes_indices_poly_parallel",
    "G_from_Gcols",
    "G_from_euler",
]

def unit(v):
    """Return a unit-length copy of a vector."""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero length vector.")
    return v / n


def angle_from_cos_deg(c):
    """Convert a clipped cosine value to an angle in degrees."""
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def canonicalize_uvw(u, v, w):
    """Treat [uvw] ~ [-u,-v,-w]; choose sign so first nonzero is positive."""
    if (u, v, w) == (0, 0, 0):
        return (0, 0, 0)
    if u != 0:
        s = 1 if u > 0 else -1
    elif v != 0:
        s = 1 if v > 0 else -1
    else:
        s = 1 if w > 0 else -1
    return (s*u, s*v, s*w)


def float_hkl_from_n_via_nearest_int(n, nearest_int_hkl):
    """
    n: unit direction in crystal frame
    nearest_int_hkl: integer (h,k,l) or [u,v,w] triple
    returns: (float triple, alpha, misfit)
    """
    n = unit(n)
    m = np.array(nearest_int_hkl, float)
    alpha = float(np.dot(n, m))            # LS-optimal scale along m
    hkl_float = alpha * n
    misfit = np.linalg.norm(hkl_float - m)
    return hkl_float, alpha, misfit


def nearest_small_dir_all_axes_stream(dx, dy, dz, limits):
    """
    For given crystal directions dx, dy, dz (for sample X/Y/Z),
    and a list of integer search limits, sweep max|u|,|v|,|w| from 1..max(limits),
    tracking the best [uvw] per axis (in the ± equivalence sense).

    Returns dict with angle errors and indices vs limit, suitable for plotting.
    """
    dx, dy, dz = unit(dx), unit(dy), unit(dz)
    limits = sorted(set(int(L) for L in limits if L >= 1))
    if not limits:
        return {}

    Lmax = limits[-1]
    D = [dx, dy, dz]                 # for axes X, Y, Z

    # running best for each axis
    best_cos = np.array([-np.inf, -np.inf, -np.inf], float)
    best_uvw = [(0,0,0), (0,0,0), (0,0,0)]

    index_x, index_y, index_z = [], [], []
    ang_err_x, ang_err_y, ang_err_z = [], [], []

    want_idx = 0  # index in 'limits'

    for L in range(1, Lmax + 1):
        # shell: max(|u|,|v|,|w|) == L
        for u in range(-L, L + 1):
            for v in range(-L, L + 1):
                for w in range(-L, L + 1):
                    if max(abs(u), abs(v), abs(w)) != L or (u == 0 and v == 0 and w == 0):
                        continue

                    uu, vv, ww = canonicalize_uvw(u, v, w)
                    m = np.array([uu, vv, ww], float)
                    m_hat = m / np.linalg.norm(m)

                    for i, d in enumerate(D):
                        c = abs(float(np.dot(m_hat, d)))  # ± line equivalence
                        if c > best_cos[i]:
                            best_cos[i] = c
                            best_uvw[i] = (uu, vv, ww)

        # emit results whenever we hit one of the requested limits
        while want_idx < len(limits) and limits[want_idx] == L:
            angs = [angle_from_cos_deg(best_cos[i]) for i in range(3)]
            uvws = [best_uvw[i] for i in range(3)]

            index_x.append((uvws[0], L))
            index_y.append((uvws[1], L))
            index_z.append((uvws[2], L))
            ang_err_x.append(angs[0])
            ang_err_y.append(angs[1])
            ang_err_z.append(angs[2])

            Lval = limits[want_idx]
            print(f"L={Lval}:")
            print("  X", uvws[0], f"({angs[0]:.5f}°)")
            print("  Y", uvws[1], f"({angs[1]:.5f}°)")
            print("  Z", uvws[2], f"({angs[2]:.5f}°)")
            print("="*80)

            want_idx += 1

    return {
        "index_x": index_x, "index_y": index_y, "index_z": index_z,
        "ang_err_x": ang_err_x, "ang_err_y": ang_err_y, "ang_err_z": ang_err_z
    }


def uvw_to_str(uvw):
    """Format a Miller direction triple as a space separated string."""
    return " ".join(f"{int(c):d}" for c in uvw)


def plot_nearest_small_dir_results(res_arrays, logx=False):
    """Plot nearest [uvw] direction and angular error versus search limit."""

    axis_data = [
        ("X", res_arrays["index_x"], res_arrays["ang_err_x"]),
        ("Y", res_arrays["index_y"], res_arrays["ang_err_y"]),
        ("Z", res_arrays["index_z"], res_arrays["ang_err_z"]),
    ]

    for axis_label, indexed_directions, angle_errors in axis_data:
        limits = np.array(
            [limit for _, limit in indexed_directions]
        )

        labels = [
            uvw_to_str(uvw)
            for uvw, _ in indexed_directions
        ]

        angle_errors = np.asarray(angle_errors)

        fig, ax = plt.subplots(figsize=(10, 4))

        for label in dict.fromkeys(labels):
            mask = np.array([value == label for value in labels])

            ax.scatter(
                limits[mask],
                angle_errors[mask],
                s=100,
                label=label,
            )

        if logx:
            ax.set_xscale("log")

        ax.grid(True)
        ax.set_xlabel("search limit: max(|u|, |v|, |w|)")
        ax.set_ylabel("angle difference (°)")
        ax.set_title(
            f"Angle between crystal axis and nearest [uvw] – {axis_label}"
        )
        ax.legend(
            title="nearest [uvw]",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        fig.tight_layout()
        plt.show()


def nearest_small_dir(d, maxu = 6):
    """Nearest reduced [uvw] to a direction d (treat ± the same line)."""
    d = unit(d)
    best_ang, best_uvw = 1e9, None
    for u, v, w in generate_uvw_set(maxu):
        ang = angle_deg(d, (u, v, w), plane_equiv=True)  # ± equivalence for a line
        if ang < best_ang:
            best_ang, best_uvw = ang, (u, v, w)
    return best_ang, best_uvw


# def generate_uvw_set(max_index = 6):
#     """Generate canonical reduced [uvw] directions up to a maximum index."""
#     S = set()
#     for u in range(-max_index, max_index + 1):
#         for v in range(-max_index, max_index + 1):
#             for w in range(-max_index, max_index + 1):
#                 if u == 0 and v == 0 and w == 0:
#                     continue
#                 S.add(canonicalize_uvw(u, v, w))
#     S.discard((0, 0, 0))
#     return sorted(S, key=lambda t: (abs(t[0])+abs(t[1])+abs(t[2]), abs(t[0]), abs(t[1]), abs(t[2])))

    
def generate_uvw_set(max_index=6):
    """Generate unique reduced [uvw] directions."""
    directions = set()

    for u in range(-max_index, max_index + 1):
        for v in range(-max_index, max_index + 1):
            for w in range(-max_index, max_index + 1):
                if u == 0 and v == 0 and w == 0:
                    continue

                g = gcd(gcd(abs(u), abs(v)), abs(w))
                if g == 0:
                    g = 1

                u_r = u // g
                v_r = v // g
                w_r = w // g

                directions.add(canonicalize_uvw(u_r, v_r, w_r))

    return sorted(
        directions,
        key=lambda t: (
            abs(t[0]) + abs(t[1]) + abs(t[2]),
            abs(t[0]),
            abs(t[1]),
            abs(t[2]),
            t,
        ),
    )



def angle_deg(u, v, plane_equiv=True):
    """
    Calculate the angle between two vectors.

    Parameters
    ----------
    u, v : array, shape (3,)
        Input direction vectors.
    plane_equiv : bool, default True
        If True, use the absolute value of the normalized dot product so that
        ``v`` and ``-v`` are treated as the same crystallographic direction
        line.

    Returns
    -------
    angle : float
        Angle between the vectors in degrees.
    """
    u = unit(u)
    v = unit(v)
    dot = abs(float(np.dot(u, v))) if plane_equiv else float(np.dot(u, v))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def _init_worker(maxu):
    """Initializer for worker processes. Pre-computes candidates once per worker."""
    global _worker_candidates
    # Use the vectorized generator
    uvw_int, mhat, maxabs, l1 = _candidate_arrays_vectorized(int(maxu))
    _worker_candidates = {
        'uvw_int': uvw_int,
        'mhat': mhat,
        'maxabs': maxabs,
        'l1': l1,
        'N': uvw_int.shape[0]
    }


def _candidate_arrays_vectorized(maxu: int):
    """
    Generates candidate directions using NumPy vectorization.
    faster than the nested loop version.
    """
    # Create grid
    r = np.arange(-maxu, maxu + 1, dtype=np.int32)
    u, v, w = np.meshgrid(r, r, r, indexing='ij')
    
    # Flatten
    u = u.ravel()
    v = v.ravel()
    w = w.ravel()
    
    #  removing 0,0,0:
    nonzero_mask = (u != 0) | (v != 0) | (w != 0)
    u, v, w = u[nonzero_mask], v[nonzero_mask], w[nonzero_mask]
    
    # Vectorized GCD
    g = np.gcd(np.gcd(np.abs(u), np.abs(v)), np.abs(w))
    g[g == 0] = 1 # Safety, though nonzero_mask handles 0,0,0
    
    u //= g
    v //= g
    w //= g
    
    # Canonical Sign (First non-zero must be positive)
    # Determine sign multiplier
    signs = np.ones_like(u, dtype=np.int8)
    mask_u = u < 0
    mask_v = (u == 0) & (v < 0)
    mask_w = (u == 0) & (v == 0) & (w < 0)
    
    signs[mask_u] = -1
    signs[mask_v] = -1
    signs[mask_w] = -1
    
    u *= signs
    v *= signs
    w *= signs
    
    # Unique rows (Canonicalization)
    # Stack and find unique
    uvw = np.stack([u, v, w], axis=1)
    # np.unique with axis is efficient
    uvw_unique, indices = np.unique(uvw, axis=0, return_index=True)
    
    # Sort to ensure deterministic order (optional but good for caching)
    # np.unique already sorts
    
    uvw_int = uvw_unique.astype(np.int32)
    
    # Precompute floats
    norms = np.linalg.norm(uvw_int.astype(np.float64), axis=1, keepdims=True)
    mhat = np.ascontiguousarray(uvw_int.astype(np.float32) / norms.astype(np.float32))
    
    maxabs = np.max(np.abs(uvw_int), axis=1).astype(np.float32)
    l1 = np.sum(np.abs(uvw_int), axis=1).astype(np.float32)
    
    return uvw_int, mhat, maxabs, l1


@njit(parallel=True, cache=True, fastmath=True)
def _pick_dir_numba(dirs, mhat, maxabs, l1, tol_cos, use_tol, prefer_low):
    """
    dirs: (M, 3) float32
    mhat: (N, 3) float32
    Returns: indices (M,), angles (M,)
    """
    M = dirs.shape[0]
    N = mhat.shape[0]
    
    best_idx = np.zeros(M, dtype=np.int32)
    best_ang = np.zeros(M, dtype=np.float32) # Store cos for comparison, convert later
    
    # Initialize best cos (we want max cos => min angle)
    best_cos = np.full(M, -1.0, dtype=np.float32)
    
    # For tolerance logic
    best_score = np.full(M, np.inf, dtype=np.float32)
    best_tol_idx = np.zeros(M, dtype=np.int32)
    
    # Parallel loop over grains
    for i in prange(M):
        dx, dy, dz = dirs[i, 0], dirs[i, 1], dirs[i, 2]
        
        # Local bests for this grain
        loc_best_cos = -1.0
        loc_best_idx = 0
        
        loc_best_score = np.inf
        loc_best_tol_idx = 0
        loc_has_tol = False
        
        for j in range(N):
            # Dot product
            c = np.abs(dx * mhat[j, 0] + dy * mhat[j, 1] + dz * mhat[j, 2])
            
            # Update Global Best (Min Angle)
            if c > loc_best_cos:
                loc_best_cos = c
                loc_best_idx = j
            
            # Update Tolerance Best
            if use_tol:
                if c >= tol_cos:
                    # Score: lower is better. maxabs * 1e8 + l1 * 1e4 - cos
                    # We want to minimize score.
                    score = (maxabs[j] * 1e8) + (l1[j] * 1e4) - c
                    
                    if score < loc_best_score:
                        loc_best_score = score
                        loc_best_tol_idx = j
                        loc_has_tol = True
        
        # Store results
        if use_tol and prefer_low and loc_has_tol:
            best_idx[i] = loc_best_tol_idx
            best_cos[i] = np.abs(dx * mhat[loc_best_tol_idx, 0] + dy * mhat[loc_best_tol_idx, 1] + dz * mhat[loc_best_tol_idx, 2])
        else:
            best_idx[i] = loc_best_idx
            best_cos[i] = loc_best_cos
            
        # Calculate angle immediately to save pass
        # Clip to avoid nan
        # val = loc_best_cos
        val = best_cos[i]
        if val > 1.0: val = 1.0
        if val < -1.0: val = -1.0 # Should be abs, but safety
        best_ang[i] = np.degrees(np.arccos(val))
        
    return best_idx, best_ang


def _process_chunk(args):
    """Worker function that uses the global _worker_candidates"""
    chunk_idx, phi1, PHI, phi2, s, e, maxZ, maxX, maxY, tolZ, tolX, tolY, prefer, mode, fix_Z, euler_mode = args
    
    # Retrieve candidates from worker global memory
    if not _worker_candidates:
        raise RuntimeError("Worker not initialized")
        
    uvw_int = _worker_candidates['uvw_int']
    mhat = _worker_candidates['mhat']
    maxabs = _worker_candidates['maxabs']
    l1 = _worker_candidates['l1']
    
    # 1. Euler to Axes
    # (Inline small helper to avoid import issues in worker if needed, but R is picklable)
    g = R.from_euler("ZXZ", np.column_stack([phi1[s:e], PHI[s:e], phi2[s:e]]), degrees=True).as_matrix()
    if euler_mode == "transpose":
        dx, dy, dz = g[:, 0, :], g[:, 1, :], g[:, 2, :]
    else:
        dx, dy, dz = g[:, :, 0], g[:, :, 1], g[:, :, 2]
    
    # Normalize safety
    for d in (dx, dy, dz):
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        norms[norms==0] = 1
        d /= norms
        
    dx = dx.astype(np.float32)
    dy = dy.astype(np.float32)
    dz = dz.astype(np.float32)
    
    Mc = e - s
    
    # 2. Pick Directions (Numba)
    tol_cos_Z = np.cos(np.deg2rad(tolZ)) if tolZ is not None else -1.0
    tol_cos_X = np.cos(np.deg2rad(tolX)) if tolX is not None else -1.0
    tol_cos_Y = np.cos(np.deg2rad(tolY)) if tolY is not None else -1.0
    
    use_tol = (tolZ is not None and prefer)
    
    # We need to call numba func 3 times (X, Y, Z) potentially with different max indices
    # However, for simplicity and memory, we assume MAX_Z dominates the candidate list.
    # If maxX/Y < maxZ, we should ideally slice the candidate list. 
    # For this optimization, we assume one global candidate list (max of all) to simplify shared mem.
    # To strictly respect maxX/Y, we would need to filter indices where maxabs <= maxX.
    # Optimization: Filter indices once in initializer if specific maxX/Y are provided.
    # Here we assume maxZ is the bound for all for simplicity, or we pass sliced views.
    # To keep it robust: We will use the full list, the 'maxabs' check in score handles preference, 
    # but strict cutoff requires slicing. Let's assume maxZ is the hard limit for all for now 
    # as per typical usage, or we slice inside worker.
    
    # Slicing for X/Y if needed (Optimization)
    # Find mask for maxX
    # maskX = maxabs <= maxX if maxX < _worker_candidates['N'] else np.ones(_worker_candidates['N'], dtype=bool)
    # maskY = maxabs <= maxY if maxY < _worker_candidates['N'] else np.ones(_worker_candidates['N'], dtype=bool)
    # maskZ = maxabs <= maxZ if maxZ < _worker_candidates['N'] else np.ones(_worker_candidates['N'], dtype=bool)
    maskX = maxabs <= maxX
    maskY = maxabs <= maxY
    maskZ = maxabs <= maxZ
    
    # To use Numba efficiently, we should pass contiguous arrays. 
    # Creating views inside worker is okay.
    
    # Z
    idxZ, angZ = _pick_dir_numba(dz, mhat[maskZ], maxabs[maskZ], l1[maskZ], 
                                 tol_cos_Z, (tolZ is not None and prefer), prefer)
    # Map indices back to original uvw_int
    orig_idx_Z = np.where(maskZ)[0][idxZ]
    uvwZ = uvw_int[orig_idx_Z]
    
    # Y
    idxY, angY = _pick_dir_numba(dy, mhat[maskY], maxabs[maskY], l1[maskY], 
                                 tol_cos_Y, (tolY is not None and prefer), prefer)
    orig_idx_Y = np.where(maskY)[0][idxY]
    uvwY = uvw_int[orig_idx_Y]
    
    # X
    idxX, angX = _pick_dir_numba(dx, mhat[maskX], maxabs[maskX], l1[maskX], 
                                 tol_cos_X, (tolX is not None and prefer), prefer)
    orig_idx_X = np.where(maskX)[0][idxX]
    uvwX = uvw_int[orig_idx_X]
    
    # 3. Post-processing (Sign & Handedness) - Keep in NumPy (fast enough for M=5000)
    def align(uvw, d):
        dots = np.sum(uvw.astype(np.float64) * d, axis=1)
        return uvw * np.where(dots >= 0, 1, -1).reshape(-1, 1).astype(np.int32)

    uvwZ = align(uvwZ, dz)
    uvwY = align(uvwY, dy)
    uvwX = align(uvwX, dx)
    
    # Handedness
    cross = np.cross(uvwX.astype(np.float64), uvwY.astype(np.float64))
    h = np.sum(cross * uvwZ.astype(np.float64), axis=1)
    bad = h < 0
    if np.any(bad):
        if fix_Z:
            fX = bad & (angX >= angY); fY = bad & ~fX
            uvwX[fX] *= -1; uvwY[fY] *= -1
        else:
            # Simplified fix for snippet
            uvwX[bad] *= -1 
            
    return s, e, uvwX, uvwY, uvwZ, angX, angY, angZ


def add_axes_indices_poly_parallel(
    df, euler_cols=("phi1", "PHI", "phi2"), maxZ = 12,
    maxXY = None, maxX = None, maxY = None,
    tolZ_deg = None, tolXY_deg = None, tolX_deg = None,
    tolY_deg = None, prefer_low_index_within_tol = True,
    mode = "independent_rh", fix_Z_sign = True,
    euler_to_axes_mode = "columns", chunk_size = 5000,
    n_workers = None,):
    """
    Parallelized version using Multiprocessing + Numba + Vectorized Candidate Gen.
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
        
    # Defaults
    if maxZ is None and maxXY is None and maxX is None and maxY is None: maxZ = 12
    if maxXY is None:  maxXY = maxZ if maxZ is not None else 12
    if maxX is None:   maxX = maxXY
    if maxY is None:   maxY = maxXY
    if maxZ is None:   maxZ = maxXY
    if tolX_deg is None: tolX_deg = tolXY_deg
    if tolY_deg is None: tolY_deg = tolXY_deg
    
    # Determine the global max needed for candidate generation
    global_max = max(maxX, maxY, maxZ)
    
    p1c, Pc, p2c = euler_cols
    phi1 = df[p1c].values.astype(np.float64)
    PHI  = df[Pc].values.astype(np.float64)
    phi2 = df[p2c].values.astype(np.float64)
    M = len(df)
    
    # Prepare chunks
    n_chunks = (M + chunk_size - 1) // chunk_size
    tasks = []
    for ci in range(n_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, M)
        tasks.append((ci, phi1, PHI, phi2, s, e, maxZ, maxX, maxY, 
                      tolZ_deg, tolX_deg, tolY_deg, prefer_low_index_within_tol, 
                      mode, fix_Z_sign, euler_to_axes_mode))
    
    # Output allocation
    all_uvwX = np.empty((M, 3), dtype=np.int32)
    all_uvwY = np.empty((M, 3), dtype=np.int32)
    all_uvwZ = np.empty((M, 3), dtype=np.int32)
    all_angX = np.empty(M, dtype=np.float64)
    all_angY = np.empty(M, dtype=np.float64)
    all_angZ = np.empty(M, dtype=np.float64)
    
    print(f"Starting parallel indexing with {n_workers} workers. Max Index: {global_max}")
    print(f"Candidate space size approx: {(2*global_max+1)**3 / 2:.1f} directions")
    
    # Use ProcessPoolExecutor with initializer
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(global_max,)) as executor:
        # Use tqdm for the map
        for res in tqdm(executor.map(_process_chunk, tasks), total=n_chunks, desc="Indexing grains"):
            s, e, uX, uY, uZ, aX, aY, aZ = res
            all_uvwX[s:e] = uX
            all_uvwY[s:e] = uY
            all_uvwZ[s:e] = uZ
            all_angX[s:e] = aX
            all_angY[s:e] = aY
            all_angZ[s:e] = aZ
            
    # Build output
    out = df.copy()
    # Formatting
    fmt = lambda arr: [f"[{u:+d},{v:+d},{w:+d}]" for u, v, w in arr]
    out["uvw_X"] = fmt(all_uvwX)
    out["uvw_Y"] = fmt(all_uvwY)
    out["uvw_Z"] = fmt(all_uvwZ)
    out["ang_uvw_X_deg"] = all_angX
    out["ang_uvw_Y_deg"] = all_angY
    out["ang_uvw_Z_deg"] = all_angZ
    
    # Handedness check
    cross = np.cross(all_uvwX.astype(np.float64), all_uvwY.astype(np.float64))
    out["handedness"] = np.sum(cross * all_uvwZ.astype(np.float64), axis=1)
    
    return out


def G_from_Gcols(row):
    """Build a 3x3 orientation matrix from G11..G33 dataframe row columns."""
    return np.array([[row.G11, row.G12, row.G13],
                     [row.G21, row.G22, row.G23],
                     [row.G31, row.G32, row.G33]], float)


def G_from_euler(row):
    """Build a 3x3 orientation matrix from Bunge ZXZ Euler angles in a row."""
    ang = [row.phi1, row.PHI, row.phi2]
    return R.from_euler("ZXZ", ang, degrees=True).as_matrix()
