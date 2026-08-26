# Data loading and visualization with `read_plot`

`pyhectr.read_plot` remains the public entry point for detector I/O, prediction
loading, maximum projections, plotting, sliders, movies, and simple uncertainty
helpers.

## P07 TIFF/Varex data

```python
from pyhectr import read_plot

data, metadata = read_plot.read_P07_imgs_with_metadata(
    img_path_pattern,
    roix=ROIx,
    roiy=ROIy,
    metadata_line=25,
)
```

The returned stack has shape

```text
(n_images, n_gamma, n_delta)
```

and `metadata` is a list of numeric values extracted from each corresponding
`.metadata` file.

For a memory efficient maximum projection:

```python
max_image = read_plot.compute_max_pixel_image(
    img_path_pattern,
    roix=ROIx,
    roiy=ROIy,
)
```

## P03 Eiger/HDF5 helpers

```python
master_file, data_file, files = read_plot.find_eiger_h5_files(scan_dir)

data = read_plot.load_eiger_h5_data(data_file)

data_filtered, data_clip = read_plot.prepare_p03_eiger_stack(
    data,
    crop_y=slice(None, 2370),
    mean_threshold=4.2e9,
    clip_y=slice(1070, None),
    clip_x=slice(1030, None),
)
```

The crop and threshold values above reproduce the supplied P03 workflow and are not
universal Eiger defaults.

Prepared NumPy stacks can be discovered with:

```python
data_paths = read_plot.collect_npy_data_paths(processed_data_path)
```

## ESRF ID31 HDF5 helper

```python
data = read_plot.ID_31_read_schnucks_h5(
    folder,
    pattern="*.h5",
)
```

The helper searches each HDF5 file for a unique three dimensional dataset named
`data` and concatenates frames along axis 0.

## Prediction aggregation

Bounding-box/mask-coordinate files produced by the Mask R-CNN helper can be loaded
with:

```python
binary_masks = read_plot.load_and_sum_predictions(
    npy_files=bb_files,
    target_shape=(1024, 1440),
    score_thresh=0.01,
)
```

The result is a `(n_frames, height, width)` accumulated mask stack.

## Maximum projections

```python
max_image = read_plot.max_pxl_im(data)
max_mask = read_plot.max_pxl_im(mask)
```

## Sliders

```python
read_plot.image_slider(data, vmax1=250)

read_plot.image_mask_slider(
    data,
    mask,
    vmax_im1=250,
    vmax_im2=1,
)
```

For reciprocal space image stacks:

```python
read_plot.image_slider_Q(
    q_images,
    q_r,
    q_z,
    omega,
)
```



## ImageD11 and GrainSpotter overlays

Overlay merged ImageD11 peaks on the measured detector frames with:

```python
state = read_plot.image_slider_with_spt(
    images=data,
    omegas=omegas,
    peaks_by_frame=peaks_by_frame,
    vmax=950,
    size_by="npix",
    annotate=True,
    origin="upper",
)
```

`peaks_by_frame` is produced by
`pyhectr.grainspotter.group_peaks_by_frame`. The viewer uses ImageD11 coordinate
pairs such as `(fc, sc)` and provides slider, button, and keyboard navigation.

After GrainSpotter indexing, show the reflections assigned to one grain on the
maximum projection:

```python
state = read_plot.grain_slider_on_max_image(
    max_image=data.max(axis=0),
    refl_df=reflections,
    spot_df=spots,
    orient_df=orientations,
    vmax=200,
    origin="upper",
    annotate=True,
    preferred_axis="z",
)
```

The expected table indices are:

- `reflections`: `(grain_id, peak_id)`;
- `spots`: `peak_id`;
- `orientations`: `(grain_id, axis)`.

See [ImageD11 and GrainSpotter indexing](grainspotter.md) for the complete file
pipeline and table construction.

## MP4 movies

```python
read_plot.image_movie(
    data,
    output_file="detector_movie.mp4",
    fps=4,
    vmax1=880,
)

read_plot.image_mask_movie(
    data,
    mask,
    output_file="detector_mask_movie.mp4",
    fps=4,
    vmax_im1=880,
)
```

## Integration output folders

```python
paths = read_plot.rod_create_nested_folders(
    dir_n="rod--22",
    path=output_dir,
    gamma_check=True,
    return_paths=True,
)
```

## Intensity uncertainty helper

```python
sigma = read_plot.I_error(
    np.sqrt(intensity),
    f_low=0.20,
    f_high=0.10,
    mode="poisson",
)
```

Invalid/non-positive values are returned as `NaN`.
