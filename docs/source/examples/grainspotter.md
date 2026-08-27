# GrainSpotter input generation and result visualization

This page follows `Gvector_GrainSpotter.ipynb`. The notebook keeps the ImageD11 and GrainSpotter steps explicit so the intermediate `.spt`, `.flt`, `.gve`, `.ini`, and `grains.log` files can be inspected.

## 1. Load the scan and omega metadata

Load P07 detector frames through `pyhectr.read_plot` and keep one measured omega value per frame:

```python
data, omega_metadata = read_imgs_with_metadata(img_path_pattern, ROIx, ROIy)
omes = np.asarray(omega_metadata)[:, 0]
```

For the full workflow, keep `GREEDY=False`. 
The greedy branch is only for a maximum pixel preview.

## 2. Run ImageD11 peak search

The notebook writes a 2-D peak table (`.spt`) and merged spot table (`.flt`) with ImageD11 `labelimage`. The peak search loop sorts frames by measured omega before calling `peaksearch`.

## 3. Inspect peaks on detector frames

Parse the peak table and group peaks by measured omega:

```python
peaks_list = parse_spt_table(f"{PATH2SAVE}/{base_filename}.flt")
peaks_by_frame = group_peaks_by_frame(peaks_list, omegas=omes, tol_deg=...)
```

When adapting the notebook, set `tol_deg` from the measured omega step:

```python
unique_omega = np.sort(np.unique(omes))
omega_step = float(np.median(np.abs(np.diff(unique_omega))))
```

Use `image_slider_with_spt` from `pyhectr.read_plot` to inspect detected spots on the image stack.

## 4. Write ImageD11 geometry and g-vectors

The notebook writes the ImageD11 parameter file, loads the `.flt` table through
`transformer.transformer`, computes two-theta/eta/omega, and writes `peaks.gve`.

## 5. Write GrainSpotter.ini

The GrainSpotter input should use measured scan metadata for `domega` and
`omegarange`:

```python
domega {omega_step:.8g}
omegarange {omes.min() - 0.5 * omega_step:.8g} {omes.max() + 0.5 * omega_step:.8g}
```


## 6. Parse and visualize results

After external GrainSpotter execution, parse `grains.log`, build the orientation and reflection tables, and join detector coordinates through `spot3d_id`:

```python
axes_df = build_multigrain_table(
    log_path,
    axes=("x", "y", "z"),
    max_index=6,
    order_indices=(0, 2, 1),
    a=cell_a,
)

reflections_df = build_reflections_table(log_path)
spot_coord_df = build_spot_coord_table(peaks_list)
```

Finally, use `grain_slider_on_max_image` to overlay assigned reflections grain by grain.
