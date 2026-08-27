# P07 sapphire: rocking-scan integration

This page follows `P07_sapphire_data_integration.ipynb`.

The notebook repeats the same workflow for the `(2,2,L)`, `(3,0,L)`, and `(1,1,L)` rod families.

## 1. Load data and preparation products

Load:

- raw P07 detector frames;
- adjusted theta array;
- reconstructed `h`, `k`, and `l`;
- CTR mask;
- beamstop mask.

All arrays must correspond to the same image order and detector ROI.

## 2. Build the display q-grid

```python
x, y, RR_r, RR_z, q_r, q_z = xrd_geom.Q_grid2(
    ROIx,
    ROIy,
    pix_size,
    SDD,
    Lambda,
    x0,
    y0,
    incidence_ang,
)
```

The notebook uses `scipy.interpolate.interpn` to display the maximum detector image on this grid.

## 3. Define gamma bins

```python
bin_gamma_rate = 15

gamma_edges = np.arange(
    0,
    data.shape[1],
    bin_gamma_rate,
)

if gamma_edges[-1] != data.shape[1]:
    gamma_edges = np.append(
        gamma_edges,
        data.shape[1],
    )

gamma_centres = (
    (gamma_edges[:-1] + gamma_edges[1:]) // 2
).astype(int)
```

## 4. Select one rod in detector space

For `(2,2,L)` the notebook selects an x/delta range and obtains its detector
coordinates:

```python
pixel_coord, rod_masked = integration.rod_points_prep(
    mask_arr=Mask,
    x_l=x_left,
    x_r=x_right,
)
```

The same pattern is repeated with different detector ranges for the other rods.

## 5. Apply a gamma threshold

The low angle region is removed by mapping a target gamma angle to a detector index:

```python
gamma_threshold_index = integration.find_closest_value_index(
    gamma_,
    gamma_threshold,
)

pixel_coord_plot = pixel_coord[
    pixel_coord[:, 1] < gamma_threshold_index
]
```

## 6. Estimate omega/delta support

```python
half_delta_r, _ = integration.get_delta_range(
    pixel_coord_plot,
    gamma_pxl_range=gamma_pxl_range,
    sigma_factor=1,
)

half_omega_r = integration.get_omega_range(
    pixel_coord_plot,
    gamma_pxl_range=data.shape[1],
    sigma_factor=1,
)
```

## 7. Apply the 2-D correction map

```python
Ctot_full = integration.apply_corrections2D(
    delta_,
    gamma_,
    incidence_ang,
    rocking=True,
    return_map=True,
)

corrected_data = data / Ctot_full[None, :, :]
```

## 8. Integrate the rod

```python
results = integration.rocking_scan_integration(
    corrected_data,
    pixel_coord_plot,
    half_omega_r,
    half_delta_r,
    bin_gamma_rate,
    gamma_edges,
    gamma_centres,
    FLAG="median",
    medfilt_kernel=51,
)

(
    intensities_summed,
    raw_profiles,
    subtracted_profiles,
    backgrounds,
    omega_windows,
    gamma_windows,
    delta_windows,
) = results
```

## 9. Inspect the rocking profiles

The notebook removes empty profile entries before passing them to
`read_plot.om_profile_slider`.


## 10. Beamstop cleaning and uncertainty

A binned beamstop fraction is used to invalidate detector bins above a selected
shadowing threshold.

The final structure factor quantity is plotted as `sqrt(I)` and an empirical
error is calculated with:

```python
error = read_plot.I_error(
    I_sqrt_clean,
    f_low=0.20,
    f_high=0.10,
    mode="poisson",
)
```

## 11. Save the rod table

The current table contains:

```text
H K L F Error
```

and is written as .txt files.
