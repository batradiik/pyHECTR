# Theoretical Bragg peaks and theta-offset refinement

The P07 preparation example uses theoretical reflection positions to label selected detector features and refine a constant azimuthal offset.

## Theoretical reciprocal-space map

`show_reciprocal_space_plane` calculates allowed reflections, transforms their
reciprocal space vectors into the structure factors, and plots the diffraction plane.

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

Filter the returned peak table to the plotted region with the current API:

```python
peak_data = bs.filter_peaks_in_plot_range(
    peak_data,
    x_lim=(-6.05, 6.05),
    y_lim=(-0.01, 8.51),
)
```

## Sparse detector pixels to HKL

The theta offset cost does not need the complete 3-D reconstruction. Selected
detector points are converted directly by
`theta0_finder.pixels_to_hkl_pointwise`.

A rod point is stored as

```text
(image_index, gamma_pixel, delta_pixel)
```

and the trial effective angle is

```text
theta_eff = omega[image_index] - theta0
```

## Robust cost scan

```python
theta_grid = np.arange(-181, 181, 1)

theta0_best, costs = bs.scan_theta(
    theta_grid,
    rods,
    keep_frac=0.10,
    omes_ang=theta_ang,
    UBinv=UB_inv,
    Lambda=wavelength,
    x0=x0,
    y0=y0,
    pix_size=pix_size,
    SDD=SDD,
)
```

`keep_frac` controls how much of the lowest finite point wise residual distribution is retained for the robust mean/median cost.

A finer grid can then be evaluated around the coarse minimum.

## Hexagonal symmetry

```python
family = bs.hexagonal_hk_symmetry(h, k)
```

For several rods:

```python
results = bs.brute_force_hk_symmetry_search(
    rods=rods,
    keep_frac=0.10,
    theta_grid_coarse=theta_grid,
    omes_ang=theta_ang,
    UBinv=UB_inv,
    Lambda=wavelength,
    x0=x0,
    y0=y0,
    pix_size=pix_size,
    SDD=SDD,
    hk_families=hk_families,
)
```
