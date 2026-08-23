# P07 sapphire: preparation, theta refinement, and CTR masks

This page follows `P07_sapphire_data_preparation.ipynb`.

## 1. Imports and experimental geometry

The notebook imports NumPy, Matplotlib, xrayutilities, pandas, OpenCV, tqdm, and
the pyHECTR geometry/read/Bragg modules.

```python
from pyhectr import bragg_simulations as bs
from pyhectr import read_plot
from pyhectr import xrd_geom
```

The detector ROI is converted to `delta` and `gamma`, then the reciprocal basis and inverse UB matrix are created from the sapphire cell.

## 2. Load P07 detector frames and omega metadata

```python
data, omes_with_z = read_plot.read_P07_imgs_with_metadata(
    img_path_pattern,
    ROIx,
    ROIy,
)

omes = np.array([
    row[0] if len(row) else np.nan
    for row in omes_with_z
])
```

Image/omega ordering must be verified before reconstruction.

## 3. Theoretical scattering map

Candidate sapphire in plane directions are generated and passed to
`show_reciprocal_space_plane`.

```python
ax, scatter, peak_data = bs.show_reciprocal_space_plane(
    material,
    experiment,
    ttmax=180,
    ax=ax,
    projection="polar",
    scalef=200,
    sf_threshold=6e-9,
    show_legend=False,
    q_max=12.1,
)
```

Use the current keyword names when filtering:

```python
peak_data_filtered = bs.filter_peaks_in_plot_range(
    peak_data,
    x_lim=(-6.05, 6.05),
    y_lim=(-0.01, 8.51),
)
```

The filtered records are collected into a pandas reflection table.

## 4. Theta-offset search: Bragg-point mode

Selected detector regions are assigned to expected Bragg positions. A coarse
`scan_theta` is followed by a finer scan around the minimum.

The notebook also evaluates several `keep_frac` values and uses
`report_best_fraction` to inspect the actual HKL distribution behind a cost minimum.

## 5. Theta offset search: CTR mode and symmetry

A second mode represents selected detector trajectories as rods with target `(h, k)` values.

For hexagonal sapphire, candidate assignments can come from:

```python
bs.hexagonal_hk_symmetry(h, k)
```

or from coordinate equivalent reflections in the theoretical table:

```python
bs.extract_hk_symmetry_from_table(...)
```

`brute_force_hk_symmetry_search` tests combinations before the refined theta scan.

## 6. Reconstruct reciprocal space

Use the selected final offset:

```python
theta_eff = theta_ang - theta0

h, k, l = xrd_geom.hkl_calc(
    delta_,
    gamma_,
    theta_eff,
    UB_inv,
    Lambda,
)
```

Large data are gridded with the new chunk aware helper:

```python
gridder = xrd_geom.grid_hkl_3d(
    h,
    k,
    l,
    data,
    bins=(bins, bins, bins_angle),
    max_points_per_call=500_000_000,
)

INT = xu.maplog(
    gridder.data.transpose(),
    6,
    0,
)
```

## 7. Detect rod centers and construct the mask

```python
peaks_h = xrd_geom.find_CTR(
    INT, 1, min_peak_height, peak_dist, med_filt
)

peaks_k = xrd_geom.find_CTR(
    INT, 2, min_peak_height, peak_dist, med_filt
)

h_peak = gridder.xaxis[peaks_h]
k_peak = gridder.yaxis[peaks_k]

Mask = xrd_geom.make_mask_fast(
    h,
    k,
    h_peak,
    k_peak,
    threshold=0.024,
)
```

Inspect the data and mask together before saving:

```python
read_plot.image_mask_slider(
    data,
    Mask,
    vmax_im1=250,
    vmax_im2=1,
)
```

## 8. Beamstop treatment

The notebook estimates the beamstop/blocked region from detector image statistics and OpenCV operations. This part is experiment specific and should be revalidated for a new detector exposure or beamstop geometry.

## 9. Save products for the integration notebook

The preparation stage saves:

```text
<scan>_theta_adjusted.npy
<scan>_h_arr.npy
<scan>_k_arr.npy
<scan>_l_arr.npy
P07_<scan>_Mask.npy
<scan>_detected_beamstops.npy
```
