# EBSD and crystallographic direction indexing

## Load and normalize the EBSD table

```python
df = pd.read_csv(
    "Nb_706_sample_primary_CS0.txt",
    sep=r"\s+",
    engine="python",
)

df = coerce_angle_cols(
    df,
    ("phi1", "PHI", "phi2"),
)
```

## Index the sample axes

The current poly-Nb example uses the transpose Euler convention:

```python
indexed = add_axes_indices_poly_parallel(
    df,
    maxZ=12,
    euler_to_axes_mode="transpose",
    n_workers=24,
)
```

The returned table contains:

```text
uvw_X
uvw_Y
uvw_Z
ang_uvw_X_deg
ang_uvw_Y_deg
ang_uvw_Z_deg
handedness
```

`maxZ`, `maxX`, and `maxY` limit the maximum absolute component of the candidate
integer directions.

For high index orientations, compare several search limits: the continuous
orientation does not change, but the nearest discrete `[uvw]` approximation can.

## Orientation maps

```python
fig, ax = plot_axis_orientation_map_clickable(
    indexed,
    axis="Z",
    gamma=0.51,
    mode="gamma",
    title="Z-axis orientation map",
)
```

When the table contains the `uvw_*` and angular error columns, clicking a point
reports both its Euler orientation and nearest crystallographic direction.

## Calibrate EBSD coordinates to sample millimetres

```python
cal = calibrate_ebsd_to_mm(
    df,
    points_mm,
    x_col="X",
    y_col="Y",
)
```

The result contains forward and inverse affine transforms.

Manual control points can also be fit directly:

```python
A, t = fit_affine(ebsd_points, mm_points)
A_inv, t_inv = invert_affine(A, t)

point_ebsd = apply_affine(
    point_mm,
    A_inv,
    t_inv,
)
```

## Select the beam footprint

```python
roi = read_ebsd_region(
    df,
    x0=x0,
    y0=y0,
    shape="ellipse",
    rx=rx_pixels,
    ry=ry_pixels,
    angle_deg=0,
)
```

The same ROI can then be indexed at several maximum `[uvw]` limits.
