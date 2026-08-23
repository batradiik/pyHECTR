# Mask R-CNN use case: P03 and P07

This page follows `mask_nn_P03_P07_usecase.ipynb`.

## P03 polycrystalline case

### 1. Load Eiger helpers from `read_plot`

```python
from pyhectr.read_plot import (
    find_eiger_h5_files,
    load_eiger_h5_data,
    prepare_p03_eiger_stack,
    collect_npy_data_paths,
    load_and_sum_predictions,
)
```

### 2. Prepare the P03 detector stack

```python
_, h5_file, _ = find_eiger_h5_files(current_scan)

h5_data = load_eiger_h5_data(h5_file)

data_filtered, data_clip = prepare_p03_eiger_stack(
    h5_data,
    crop_y=slice(None, 2370),
    mean_threshold=4.2e9,
    clip_y=slice(1070, None),
    clip_x=slice(1030, None),
)
```

Save full and cropped stacks as `.npy` if the preprocessing stage is needed.

### 3. Select prepared stacks

```python
data_paths = collect_npy_data_paths(
    processed_data_path
)

data_paths_clip = [
    path
    for path in data_paths
    if "clip" in os.path.basename(path)
]
```

### 4. Configure Mask R-CNN

```python
cfg = build_mask_rcnn_cfg(
    yaml_name="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    dataset_name="inference_dataset",
    score_thresh_test=0.05,
    device="cuda:0",
)
```

### 5. Preview an output directory

```python
example_output_dir = prediction_output_dir(
    output_dir,
    scan_name,
    model_name,
    output_prefix="P03_clip_resized_2025",
)
```

### 6. Run the model loop

```python
outputs = run_prediction_for_npy_files(
    data_paths=data_paths_for_inference,
    model_names=model_names,
    model_dir=model_dir,
    output_dir=output_dir,
    cfg=cfg,
    full_size_predict=False,
    save_predicted_mask=False,
    save_prediction_bb=True,
    target_height=1024,
    target_width=1440,
    output_prefix="P03_clip_resized",
    n_jobs=-1,
    fps=1,
)
```

### 7. Rebuild binary masks from stored predictions

```python
binary_masks = load_and_sum_predictions(
    npy_files=bb_files,
    target_shape=(1024, 1440),
    score_thresh=0.01,
)
```

## P07 single-crystal case

Load P07 frames through the same public read/plot module:

```python
from pyhectr.read_plot import (
    compute_max_pixel_image,
    read_P07_imgs_with_metadata,
)

max_image = compute_max_pixel_image(
    img_path_pattern,
    roix=ROIx,
    roiy=ROIy,
)

images_array, omega_metadata = read_P07_imgs_with_metadata(
    img_path_pattern,
    roix=ROIx,
    roiy=ROIy,
)
```

Infer preprocessing from the model filename, prepare the full stack, create a
`DefaultPredictor`, and call `run_inference_with_default_predictor`.

The returned aggregated mask can then be converted into the same detector space mask workflow used for classical integration.
