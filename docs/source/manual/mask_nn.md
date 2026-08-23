# Mask R-CNN detector space localization

The ML branch predicts CTR support regions directly in detector space.

P03/P07 loading is handled through `pyhectr.read_plot`. `pyhectr.mask_nn` is focused
on image preprocessing, Detectron2 configuration, inference, and model output paths.

## Intensity preprocessing

```python
from pyhectr.mask_nn import (
    robust_normalization,
    hist_equalization,
    clahe_equalization,
    pos_adjustment_robust_normalization,
)
```

## Three channel construction

```python
prepared = process_images_array_parallel(
    images_array,
    ch_method="neighbor_slices",
    method=clahe_equalization,
    n_jobs=-1,
)
```

Supported workflow names include:

- `neighbor_slices`;
- `custom_mix`;
- `duplicate_var_prep`;
- `duplicate` as a compatibility alias.

For neighboring slices, the first/last frame reuses itself when one neighbor is
missing.

## Recover preprocessing from model names

```python
channel_method = which_method(
    model_name,
    ["duplicate", "duplicate_var_prep", "custom_mix", "neighbor_slices"],
)

intensity_method = which_method_intensity(
    model_name,
    possible_methods_intensity,
)
```

## Detectron2 configuration

```python
cfg = build_mask_rcnn_cfg(
    yaml_name="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    dataset_name="inference_dataset",
    num_classes=1,
    score_thresh_test=0.05,
    device="cuda:0",
)
```


## Prediction output directory

```python
out_dir = prediction_output_dir(
    output_dir,
    scan_name,
    model_name,
    output_prefix="P03_clip_resized",
)
```

## High level prediction loop

```python
outputs = run_prediction_for_npy_files(
    data_paths=data_paths,
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

The function prepares each stack according to the model name convention, optionally
resizes it, runs `DefaultPredictor`, writes an MP4 visualization, and optionally writes
dense masks or instance bounding-box/mask coordinate objects.

## Direct predictor use

```python
predictor = DefaultPredictor(cfg)

aggregated_mask = run_inference_with_default_predictor(
    images_array=prepared_images,
    predictor=predictor,
    cfg=cfg,
    output_movie_path="prediction.mp4",
    target_size=(1024, 1440),
    save_mask=False,
    save_bb=True,
    output_bb_path="prediction-bb.npy",
)
```

## Load predictions for integration

Use the public `read_plot` helper:

```python
binary_masks = read_plot.load_and_sum_predictions(
    npy_files=bb_files,
    target_shape=(1024, 1440),
    score_thresh=0.01,
)
```

## Optional morphology

```python
grown = xrd_geom.grow_or_shrink(
    frame_mask,
    expand=5,
    metric="pixels",
    struct="rect",
)

ring = xrd_geom.grow_or_shrink(
    frame_mask,
    expand=5,
    metric="pixels",
    struct="rect",
    return_ring=True,
)
```
