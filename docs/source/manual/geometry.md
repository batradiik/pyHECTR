# Detector geometry and reciprocal-space reconstruction

## Detector ROI to delta/gamma

```python
delta = xrd_geom.ROI_to_angle(
    ROIx,
    x0,
    pix_size,
    SDD,
)

gamma = xrd_geom.ROI_to_angle(
    ROIy,
    y0,
    pix_size,
    SDD,
)
```

## Reciprocal basis and inverse UB

```python
B = xrd_geom.set_reciprocal_cell_5(
    a,
    b,
    c,
    transformation_flag=False,
)

UB_inv = xrd_geom.UBinv(
    B,
    phi0,
    chi0,
    mu0,
)
```

## HKL coordinates for the full image stack

After selecting the effective sample angle:

```python
theta_eff = theta_ang - theta0
```

calculate the fractional reciprocal coordinates:

```python
h, k, l = xrd_geom.hkl_calc(
    delta,
    gamma,
    theta_eff,
    UB_inv,
    wavelength,
)
```

The three output arrays have the same shape as the detector data.

## Chunked 3-D gridding

Large HEGIXRD detector stacks can contain more points than a single
xrayutilities `Gridder3D` call can safely pass through its 32-bit point count path.

Use:

```python
gridder = xrd_geom.grid_hkl_3d(
    h,
    k,
    l,
    data,
    bins=(800, 800, 800),
    max_points_per_call=500_000_000,
)

INT = xu.maplog(
    gridder.data.transpose(),
    6,
    0,
)
```

For small arrays, `grid_hkl_3d` performs one ordinary `Gridder3D` call. For larger
arrays it fixes the global HKL range, keeps grid data between calls, and processes
complete image-frame chunks sequentially.

## Detect in-plane CTR positions

```python
peaks_h = xrd_geom.find_CTR(
    INT,
    axNum=1,
    height=0.05,
    dist=10,
    med_kernel=3,
)

peaks_k = xrd_geom.find_CTR(
    INT,
    axNum=2,
    height=0.05,
    dist=10,
    med_kernel=3,
)

h_peak = gridder.xaxis[peaks_h]
k_peak = gridder.yaxis[peaks_k]
```

The threshold, distance, smoothing kernel, and background mode are experimental
parameters and should be checked on the projected intensity profiles.

## Detector space CTR mask

```python
mask = xrd_geom.make_mask_fast(
    h,
    k,
    h_peak,
    k_peak,
    threshold=0.024,
)
```

When both `h_peak` and `k_peak` contain multiple values, all Cartesian combinations
are used as candidate rod centers.

## Reciprocal space image visualization

`Q_grid2` builds a `(q_r, q_z)` grid and inverse detector lookup arrays for
displaying detector images in reciprocal coordinates:

```python
x, y, RR_r, RR_z, q_r, q_z = xrd_geom.Q_grid2(
    ROIx,
    ROIy,
    pix_size,
    SDD,
    wavelength,
    x0,
    y0,
    incidence_angle,
)
```
