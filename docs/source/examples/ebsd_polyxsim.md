# EBSD-supported polycrystalline Nb and PolyXSim

This page follows `EBSD_polyNb_PolyXSim.ipynb`.

## 1. Load and rename EBSD columns

The notebook reads a whitespace separated EBSD export and renames the Euler columns
to:

```text
phi1, PHI, phi2
```

Then:

```python
df = coerce_angle_cols(
    df,
    ("phi1", "PHI", "phi2"),
)
```

## 2. Index the full EBSD map

```python
MAX_Z = 12
N_WORKERS = 24

df_indexed = add_axes_indices_poly_parallel(
    df,
    maxZ=MAX_Z,
    euler_to_axes_mode="transpose",
    n_workers=N_WORKERS,
)
```

## 3. Plot X/Y/Z orientation maps

```python
plot_axis_orientation_map_clickable(
    df_indexed,
    axis="Z",
    gamma=0.51,
    mode="gamma",
)
```

Repeat for X and Y when needed.

## 4. Fit the EBSD to mm mapping

The notebook shows both:

- automatic calibration with `calibrate_ebsd_to_mm`;
- a manual affine fit from three corresponding control points using
  `fit_affine`/`invert_affine`.

## 5. Convert the selected beam position

```python
grain_ebsd = apply_affine(
    grain_mm,
    A_mm_to_ebsd,
    t_mm_to_ebsd,
)
```

## 6. Convert the physical footprint size to EBSD pixels

The affine transform is sampled along the EBSD X/Y unit vectors to obtain millimetres per EBSD pixel. The physical grazing incidence footprint radius is divided by these values to obtain the ellipse semi axes.

## 7. Select and reindex the footprint

```python
df_roi = read_ebsd_region(
    df,
    x0=x0,
    y0=y0,
    shape="ellipse",
    rx=rx_px,
    ry=ry_px,
)
```

The supplied notebook then indexes this ROI with `maxZ=6`, optional cells compare higher search limits.

## 8. Write the PolyXSim input

```python
write_polyxsim_inp_from_ebsd(
    "polyNb_grain1_from_ebsd_r_13_327.inp",
    ...,
    region_df=df_roi,
    use_Gcols=True,
    U_equals="gt",
    pos_mode="zero",
)
```
