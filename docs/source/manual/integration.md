# CTR masks and rocking scan integration

## Convert a mask to detector coordinates

```python
pixel_coord, max_mask = integration.rod_points_prep(
    mask_arr=mask,
    x_l=x_left,
    x_r=x_right,
)
```

The first output contains

```text
(image_index, gamma_pixel, delta_pixel)
```

rows.

## Restrict a rod by detector gamma

The P07 example uses a gamma angle threshold to remove the low angle part of a rod:

```python
gamma_threshold_index = integration.find_closest_value_index(
    gamma,
    gamma_threshold,
)

pixel_coord_plot = pixel_coord[
    pixel_coord[:, 1] < gamma_threshold_index
]
```

The direction of the inequality depends on the detector angle convention.

## Estimate window widths

```python
half_delta, delta_range = integration.get_delta_range(
    pixel_coord_plot,
    gamma_pxl_range=data.shape[1],
    sigma_factor=1,
)

half_omega = integration.get_omega_range(
    pixel_coord_plot,
    gamma_pxl_range=data.shape[1],
    sigma_factor=1,
)
```


## Background models

```python
background = integration.line_prof_bckg_subtr(
    rocking_curve,
    flag="median",
)
```

Available modes:

- `median` — linear fit through values at or below the median;
- `mean` — linear fit through values at or below the mean;
- `medfilt` — sliding median;
- `als` — asymmetric least squares.

## Two-dimensional detector corrections

```python
correction_map = integration.apply_corrections2D(
    delta_arr=delta,
    gamma_arr=gamma,
    incidence_ang=incidence_angle,
    rocking=True,
    return_map=True,
)

corrected_data = data / correction_map[None, :, :]
```

The map combines the selected Lorentz/rod expression, polarization, optional window
transmission, illuminated area scaling, and optional flat field.

## Gamma binning construction

```python
gamma_edges = np.arange(
    0,
    data.shape[1],
    bin_gamma_rate,
)

if gamma_edges[-1] != data.shape[1]:
    gamma_edges = np.append(gamma_edges, data.shape[1])

gamma_centres = (
    (gamma_edges[:-1] + gamma_edges[1:]) // 2
).astype(int)
```

## Rocking scan integration

```python
(
    intensities_summed,
    raw_profiles,
    subtracted_profiles,
    backgrounds,
    omega_windows,
    gamma_windows,
    delta_windows,
) = integration.rocking_scan_integration(
    corrected_data,
    pixel_coord_plot,
    half_omega,
    half_delta,
    bin_gamma_rate,
    gamma_edges,
    gamma_centres,
    FLAG="median",
    medfilt_kernel=51,
)
```

`intensities_summed` keeps one slot per gamma bin and uses `None` when a bin has no
signal hits. The profile and window collections contain the usable integration
information used by the P07 visualization workflow.

## Inspect rocking curves

```python
read_plot.om_profile_slider(
    raw_profiles_valid,
    subtracted_profiles_valid,
    backgrounds_valid,
    omega_values=theta,
    omega_windows=omega_windows,
    L_values=L_values,
    gamma_windows=gamma_windows,
    bin_omega_rate=2 * half_omega,
    bin_gamma_rate=bin_gamma_rate,
    bin_delta_rate=2 * half_delta,
    l_bin_rate=dL,
)
```
