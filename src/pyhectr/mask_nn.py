import gc
import os
import warnings
import cv2
import imageio
import numpy as np
from joblib import Parallel, delayed
from skimage import exposure


try:
    import torch
except Exception:  
    torch = None


try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
except Exception:  
    plt = None
    Slider = None

try:
    from yacs.config import CfgNode as CN
except Exception:  
    CN = None

try:
    from detectron2 import model_zoo
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog, build_detection_test_loader
    from detectron2.engine import DefaultPredictor
    from detectron2.modeling import build_model
    from detectron2.utils.visualizer import Visualizer
except Exception:  # pragma: no cover - optional dependency
    model_zoo = None
    DetectionCheckpointer = None
    get_cfg = None
    MetadataCatalog = None
    build_detection_test_loader = None
    DefaultPredictor = None
    build_model = None
    Visualizer = None


def _require_detectron2():
    """Raise an error if Detectron2 objects required for inference are unavailable."""
    if any(obj is None for obj in (model_zoo, get_cfg, MetadataCatalog, DefaultPredictor, Visualizer)):
        raise ImportError("Detectron2 is required for Mask R-CNN inference helpers.")


def _require_matplotlib():
    """Raise an error if matplotlib is unavailable."""
    if plt is None or Slider is None:
        raise ImportError("matplotlib is required for this visualization helper.")


def _ensure_hwc3(image):
    """Return an image as ``(H, W, 3)`` without changing numeric scaling."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Expected a 2-D image or an HWC image, got shape {arr.shape!r}.")
    return arr


def _preview_channel(image, channel = 1):
    """Return a 2-D channel suitable for grayscale/colormap visualization."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[..., min(channel, arr.shape[-1] - 1)]
    raise ValueError(f"Expected image with 2 or 3 dimensions, got shape {arr.shape!r}.")


def _normalise_to_uint8(image):
    """Normalize an image to uint8 for OpenCV visualization only."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.where(finite, arr, 0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def _clip_coords_to_shape(rows, cols, height, width):
    """Clip row/column coordinate arrays to a target image shape."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return rows[valid], cols[valid]


# -----------------------------------------------------------------------------
# Data loading helpers: P07 TIFF/Varex and P03 Eiger/HDF5 + prepared NumPy data
# -----------------------------------------------------------------------------


def _as_posix_path(path):
    """Return a path string with forward slashes for stable name parsing."""
    return os.fspath(path).replace("\\", "/")


def _safe_name(name):
    """Return a filesystem name fragment while keeping it readable."""
    name = str(name).strip().replace("\\", "/")
    name = name.strip("/")
    name = name.replace("/", "_")
    for old in (" ", "|", ":"):
        name = name.replace(old, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "unnamed"


def _model_group_name(model_name):
    """Return the first path component of a model path as a safe folder name."""
    parts = [p for p in _as_posix_path(model_name).split("/") if p]
    if not parts:
        return "model"
    return _safe_name(parts[0])


def _model_stem(model_name):
    """Return a readable safe model name stem without the ``.pth`` suffix."""
    stem = _as_posix_path(model_name)
    if stem.endswith(".pth"):
        stem = stem[:-4]
    return _safe_name(stem)


def prediction_output_dir(output_dir, scan_name, model_name, output_prefix=None):
    """Build the output directory used for one scan/model prediction run.

    Parameters
    ----------
    output_dir : str or path
        Base directory where all prediction products are written.
    scan_name : str
        Name of the processed detector stack, usually the ``.npy`` filename
        without extension.
    model_name : str or path
        Model path relative to ``model_dir``. The first path component is used
        as the model family subfolder.
    output_prefix : str, optional
        Experiment or workflow label, for example ``"P03_clip_resized_2025"``.
        If omitted, the scan name is used directly.

    Returns
    -------
    path : str
        Output directory path. The function centralizes path construction so
        experiment specific labels are not hard coded inside inference loops.
    """
    scan_name = _safe_name(scan_name)
    model_group = _model_group_name(model_name)
    if output_prefix is None or output_prefix == "":
        run_folder = scan_name
    else:
        run_folder = f"{_safe_name(output_prefix)}-{scan_name}-"
    return os.path.join(output_dir, run_folder, model_group)



# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def visualize_predictions(inputs_list, predictions_list, cfg,
                          method, test_dataset_name, output_movie_path,
                          fig_size=(12, 6), fps=1):
    """Visualize Detectron2 predictions next to input images and save a GIF/movie."""
    _require_detectron2()
    _require_matplotlib()

    if len(inputs_list) == 0 or len(predictions_list) == 0:
        raise ValueError("inputs_list and predictions_list must be non-empty.")

    frames = []
    fig, ax = plt.subplots(1, 2, figsize=fig_size)

    initial_image = inputs_list[0]["image"].cpu().numpy().transpose(1, 2, 0)
    image_plot = ax[0].imshow(_preview_channel(initial_image), cmap='viridis', vmin=0, vmax=250)
    ax[0].set_title('Input image', fontsize=12)

    prediction = predictions_list[0]["instances"].to("cpu")
    v = Visualizer(_ensure_hwc3(initial_image)[:, :, ::-1], MetadataCatalog.get(test_dataset_name), scale=1.2)
    out = v.draw_instance_predictions(prediction)
    vis_image = out.get_image()

    prediction_plot = ax[1].imshow(vis_image, vmin=0, vmax=250)
    ax[1].set_title('Predictions', fontsize=12)

    for input_dict, prediction in zip(inputs_list, predictions_list):
        image = input_dict["image"].cpu().numpy().transpose(1, 2, 0)
        image_plot.set_array(_preview_channel(image, channel=0))

        v = Visualizer(_ensure_hwc3(image)[:, :, ::-1], MetadataCatalog.get(test_dataset_name), scale=1.2)
        out = v.draw_instance_predictions(prediction["instances"].to("cpu"))
        vis_image = out.get_image()
        prediction_plot.set_array(vis_image)
        fig.canvas.draw()

        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame.copy())

    plt.close(fig)
    os.makedirs(os.path.dirname(output_movie_path) or '.', exist_ok=True)
    imageio.mimsave(output_movie_path, frames, fps=fps)
    print(f"###\nVisualization saved at {output_movie_path}\n###")


# -----------------------------------------------------------------------------
# Intensity preprocessing and channel construction
# -----------------------------------------------------------------------------


def robust_normalization(data, lower_bound=0, upper_bound=None, lower_percentile=0.01,
                         upper_percentile=99.7, scale_factor=255):
    """Clip an image by bounds/percentiles and scale it to ``scale_factor``.

    Constant or invalid ranges are returned as zeros instead of producing NaNs
    or infinities.
    """
    data = np.asarray(data, dtype=np.float32)
    if lower_bound is None:
        lower_bound = np.percentile(data, lower_percentile)
    if upper_bound is None:
        upper_bound = np.percentile(data, upper_percentile)

    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError(f"Normalization bounds must be finite, got {lower_bound=} and {upper_bound=}.")
    if upper_bound <= lower_bound:
        warnings.warn(
            f"upper_bound <= lower_bound in robust_normalization ({upper_bound} <= {lower_bound}); returning zeros.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros_like(data, dtype='float32')

    if lower_bound < 0:
        data = np.clip(data, 0, upper_bound)
        offset = 0.0
        denom = upper_bound
    else:
        data = np.clip(data, lower_bound, upper_bound)
        offset = lower_bound
        denom = upper_bound - lower_bound

    if denom <= 0:
        warnings.warn("Invalid normalization denominator; returning zeros.", RuntimeWarning, stacklevel=2)
        return np.zeros_like(data, dtype='float32')

    data = (data - offset) / denom
    return (scale_factor * data).astype('float32')

def positive_adjustment(img, adj_percentile):
    """Shift an image to non-negative values using a low percentile estimate."""
    p01 = np.percentile(img, adj_percentile)
    if p01 < 0:
        # img += abs(p01)
        img = img + abs(p01)
    img = np.clip(img, 0, None)
    return img.astype('float32')


def pos_adjustment_robust_normalization(data, adj_percentile=0.03, lower_bound=0, upper_bound=None, lower_percentile=0.1, upper_percentile=99.7, scale_factor=255):
    """Apply positive adjustment followed by robust normalization."""
    data_pos_adj = positive_adjustment(data, adj_percentile)
    return robust_normalization(data_pos_adj, lower_bound, upper_bound, lower_percentile, upper_percentile, scale_factor)


def hist_equalization(data, adj_percentile=0.03, rescale_percentile=99.7, scale_factor=255):
    """Apply positive adjustment, percentile rescaling, and histogram equalization."""
    data_eq = np.zeros_like(data)
    data = positive_adjustment(data, adj_percentile)
    p_upper = np.percentile(data, rescale_percentile)
    img = exposure.rescale_intensity(data, in_range=(0, p_upper))
    data_eq = exposure.equalize_hist(img)
    # assert abs(data_eq.max()-255) < 1, f'hist_equalization - range is not 255, {data_eq.max() =}'
    return scale_factor * data_eq.astype('float32')


def clahe_equalization(data, scale_factor=255, clahe_clip_limit=0.03, rescale_percentile=99.7, adj_percentile=0.03):
    """Apply positive adjustment, percentile rescaling, and CLAHE equalization."""
    data_clahe = np.zeros_like(data)
    img = positive_adjustment(data, adj_percentile)
    p_upper = np.percentile(img, rescale_percentile)
    img = exposure.rescale_intensity(img, in_range=(0, p_upper))
    # img = img / p_upper  # Normalize to [0, 1] range for CLAHE
    img = robust_normalization(img, scale_factor=1)
    data_clahe = exposure.equalize_adapthist(img, clip_limit=clahe_clip_limit)
    return (scale_factor * data_clahe).astype('float32')


def image_channels_construct(prev_slice, image_slice, next_slice,
                             ch_method, method):
    """Construct three model input channels from neighboring detector slices."""

    if 'neighbor_slices' in ch_method:
        channel3 = method(next_slice)
        channel2 = method(image_slice)
        channel1 = method(prev_slice)

    elif 'custom_mix' in ch_method:
        # Each slice uses a different intensity function
        # C3 = clahe of next, C2 = positive adjustment on current, C1 = hist eq of previous
        channel1 = hist_equalization(prev_slice)
        channel2 = pos_adjustment_robust_normalization(image_slice)
        channel3 = clahe_equalization(next_slice)
    else:
        # Optionally: raise an error if unrecognized
        raise ValueError(f"Unknown ch_method: {ch_method}")
    return channel3, channel2, channel1


def process_single_image(idx, images_array, ch_method, method, num_images):
    """Prepare one three channel model input image from an image stack index."""
    image_slice = images_array[idx]

    # Handle edge cases for first and last images
    if idx == 0:
        prev_slice = image_slice.copy()
    else:
        prev_slice = images_array[idx - 1]

    if idx == num_images - 1:
        next_slice = image_slice.copy()
    else:
        next_slice = images_array[idx + 1]

    channel3, channel2, channel1 = image_channels_construct(
        prev_slice, image_slice, next_slice, ch_method, method)

    # Stack the channels to form a 3-channel image
    image_channels = np.stack([channel3, channel2, channel1], axis=-1)

    return image_channels


def process_images_array_parallel(images_array, ch_method, method, n_jobs=-1):
    """Prepare a full image stack as three channel Mask R-CNN input in parallel."""
    images_array = np.asarray(images_array)
    if images_array.ndim < 3:
        raise ValueError(f"images_array must be a stack shaped (N,H,W), got {images_array.shape!r}.")

    num_images = images_array.shape[0]
    if num_images == 0:
        raise ValueError("images_array is empty.")

    if isinstance(ch_method, (list, tuple)):
        ch_method = ch_method[0] if len(ch_method) == 1 else 'duplicate_var_prep'

    if ('neighbor_slices' in ch_method) or ('custom_mix' in ch_method):
        if 'custom_mix' not in ch_method and method is None:
            raise ValueError("method must be provided for neighbor_slices channel construction.")
        prepared_images = Parallel(n_jobs=n_jobs)(
            delayed(process_single_image)(idx, images_array, ch_method, method, num_images)
            for idx in range(num_images)
        )

    elif 'duplicate_var_prep' in ch_method or ch_method == 'duplicate':
        if method is None:
            raise ValueError("method must be provided for duplicate_var_prep channel construction.")
        processed = Parallel(n_jobs=n_jobs)(delayed(method)(im) for im in images_array)
        processed = np.array(processed)
        prepared_images = np.stack([processed] * 3, axis=-1)

    else:
        raise ValueError(f"Unsupported ch_method: {ch_method!r}")

    print(f'\n {ch_method = } \n')
    return np.array(prepared_images)


def which_method(model_name, possible_methods):
    """Infer the channel construction method name from a trained model filename.

    Returns a string consistently.  If a short method name is a substring of a
    longer one, the longest match is preferred, e.g. ``duplicate_var_prep`` over
    ``duplicate``.
    """
    model_name = os.path.basename(str(model_name)).replace('mask_rcnn_', '')
    matches = [x for x in possible_methods if x in model_name]
    if not matches:
        raise ValueError(f"Got unknown channel method from model name {model_name!r}.")

    matches = sorted(matches, key=len, reverse=True)
    best = matches[0]
    # Ignore substring matches already contained in the best method name.
    non_nested = [m for m in matches if m == best or m not in best]
    if len(non_nested) > 1:
        raise ValueError(f"Ambiguous channel method in {model_name!r}: {matches}")
    if best == 'duplicate':
        return 'duplicate_var_prep'
    return best


possible_methods_intensity = {
    'clahe': clahe_equalization,
    'hist': hist_equalization,
    'pos_adjust': pos_adjustment_robust_normalization,
}


def which_method_intensity(name, possible_methods):
    """Infer the intensity preprocessing function from a trained model filename."""
    _name = os.path.basename(str(name)).replace('mask_rcnn_', '')
    matches = [x for x in possible_methods.keys() if x in _name]
    if not matches:
        raise ValueError(f"Got unknown intensity method from model name {_name!r}. Expected one of {list(possible_methods.keys())}.")
    matches = sorted(matches, key=len, reverse=True)
    method = matches[0]
    if len([m for m in matches if m == method or m not in method]) > 1:
        raise ValueError(f"Ambiguous intensity method in {_name!r}: {matches}")
    print(f'{method = }')
    print(f'{possible_methods[method] = }')
    return possible_methods[method]

def resize_image_worker(img, target_width, target_height, interpolation=cv2.INTER_AREA):
    """OpenCV resizing worker for parallel processing."""
    return cv2.resize(img, (target_width, target_height), interpolation=interpolation)


# -----------------------------------------------------------------------------
# Detectron2 Mask R-CNN setup and inference
# -----------------------------------------------------------------------------


def build_mask_rcnn_cfg(
    model_weights = None,
    dataset_name = "inference_dataset",
    yaml_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    num_classes = 1,
    batch_size_per_image = 256,
    score_thresh_test = 0.05,
    device = "cuda:0",
    num_workers = None,
):
    """Build the Detectron2 configuration used for CTR mask inference.

    Parameters
    ----------
    model_weights : str, optional
        Path to a trained model checkpoint. It can also be assigned later by
        :func:`run_prediction_for_npy_files`.
    dataset_name : str, default ``"inference_dataset"``
        Detectron2 dataset name stored in ``cfg.DATASETS.TEST``.
    yaml_name : str
        Detectron2 model-zoo YAML configuration.
    num_classes : int, default 1
        Number of predicted classes. CTR mask prediction uses one class.
    batch_size_per_image : int, default 256
        ROI heads batch size per image.
    score_thresh_test : float, default 0.05
        Detection score threshold used during inference.
    device : str, default ``"cuda:0"``
        Requested device. The function falls back to CPU if CUDA is absent.
    num_workers : int, optional
        Detectron2 dataloader worker count.

    Returns
    -------
    cfg : detectron2 CfgNode
        Configured Detectron2 object.
    """
    _require_detectron2()
    if torch is None:
        raise ImportError("torch is required for Detectron2 inference.")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(yaml_name))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
    cfg.MODEL.DEVICE = device if torch.cuda.is_available() and str(device).startswith('cuda') else "cpu"

    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = max(0, cpu_count // 2 - 1)
    cfg.DATALOADER.NUM_WORKERS = num_workers

    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test
    if CN is not None and 'CUSTOM' not in cfg:
        cfg.CUSTOM = CN()
    if CN is not None:
        cfg.CUSTOM.AUG = False
    cfg.DATASETS.TEST = (dataset_name,)
    if hasattr(cfg.SOLVER, 'AMP'):
        cfg.SOLVER.AMP.ENABLED = False
    if model_weights is not None:
        cfg.MODEL.WEIGHTS = model_weights
    return cfg

def visualize_aggregated_mask(aggregated_mask, output_path):
    """Save a viridis colored visualization of an aggregated prediction mask."""
    grayscale_mask = aggregated_mask
    grayscale_norm = cv2.normalize(grayscale_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored_mask = cv2.applyColorMap(grayscale_norm, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(output_path, colored_mask)
    print(f"Saved aggregated mask at {output_path}\n\n")


def run_inference_with_default_predictor(
    images_array,
    predictor,
    cfg,
    output_movie_path,
    target_size = None,
    fps=1,
    save_mask=False,
    save_bb=False,
    output_masks_path=None,
    output_bb_path=None,
):
    """Run Detectron2 Mask R-CNN inference and optionally save dense masks/BB coords.

    Parameters
    ----------
    images_array : ndarray
        Stack shaped ``(N,H,W)``, ``(N,H,W,1)``, or ``(N,H,W,3)``.
    target_size : tuple, optional
        Output/inference size as ``(height, width)``.  If omitted, the input
        frame size is used.  This avoids accidentally forcing full-size data to
        the old default ``(1024, 1440)``.
    """
    _require_detectron2()
    if torch is None:
        raise ImportError("torch is required for inference.")

    images_array = np.asarray(images_array)
    if images_array.shape[0] == 0:
        raise ValueError("images_array is empty.")

    sample_h, sample_w = images_array[0].shape[:2]
    if target_size is None:
        height, width = int(sample_h), int(sample_w)
    else:
        height, width = map(int, target_size)
    target_size = (height, width)

    os.makedirs(os.path.dirname(output_movie_path) or '.', exist_ok=True)
    aggregated_mask = np.zeros(target_size, dtype=np.uint8)
    needs_resizing = (sample_h != height) or (sample_w != width)

    masks_list = [] if save_mask else None
    bbcoords_list = [] if save_bb else None

    with imageio.get_writer(
        output_movie_path,
        fps=fps,
        codec='libx264',
        quality=6,
        ffmpeg_params=['-preset', 'slow', '-pix_fmt', 'yuv420p', '-crf', '28'],
    ) as writer:

        for idx, image in enumerate(images_array):
            image_hwc3 = _ensure_hwc3(image)
            resized_image = (
                cv2.resize(image_hwc3, (width, height), interpolation=cv2.INTER_LINEAR)
                if needs_resizing else image_hwc3.copy()
            )

            image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                outputs = predictor(image_bgr)
            instances = outputs["instances"].to("cpu")

            has_masks = instances.has("pred_masks") and len(instances) > 0
            if has_masks:
                masks = instances.pred_masks.numpy()
                combined_mask = (masks.max(axis=0) * 255).astype(np.uint8)
            else:
                combined_mask = np.zeros((height, width), dtype=np.uint8)

            if save_mask:
                masks_list.append(combined_mask)

            aggregated_mask = np.maximum(aggregated_mask, combined_mask)

            if save_bb:
                if len(instances) > 0 and instances.has("pred_boxes"):
                    boxes = instances.pred_boxes.tensor.numpy()
                    scores = instances.scores.numpy()[:, None] if instances.has("scores") else np.ones((len(boxes), 1), dtype=np.float32)
                    bb_with_scores = np.hstack((boxes, scores))
                    if has_masks:
                        masks_np = instances.pred_masks.numpy()
                        coords_list = [
                            np.column_stack(np.where(masks_np[i] > 0)).astype(np.int32)
                            for i in range(len(masks_np))
                        ]
                    else:
                        coords_list = []
                else:
                    bb_with_scores = np.empty((0, 5), dtype=np.float32)
                    coords_list = []
                bbcoords_list.append({"bb": bb_with_scores, "coords": coords_list})

            gray = _preview_channel(resized_image, channel=1).astype(np.float32)
            gray_norm = _normalise_to_uint8(gray)
            colored = cv2.applyColorMap(gray_norm, cv2.COLORMAP_VIRIDIS)

            vis = Visualizer(resized_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
            vis_img_bgr = cv2.cvtColor(
                vis.draw_instance_predictions(instances).get_image(),
                cv2.COLOR_RGB2BGR
            )
            if vis_img_bgr.shape[:2] != colored.shape[:2]:
                vis_img_bgr = cv2.resize(vis_img_bgr, (colored.shape[1], colored.shape[0]))
            frame = np.hstack((colored, vis_img_bgr))
            frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if (idx + 1) % 100 == 0 or idx + 1 == len(images_array):
                print(f"Processed {idx + 1}/{len(images_array)} images")

    if save_mask:
        if output_masks_path is None:
            raise ValueError("Provide output_masks_path when save_mask=True")
        os.makedirs(os.path.dirname(output_masks_path) or '.', exist_ok=True)
        np.save(output_masks_path, np.stack(masks_list, axis=0))
        print(f"Dense masks saved ➜ {output_masks_path}")

    if save_bb:
        if output_bb_path is None:
            output_bb_path = os.path.splitext(output_movie_path)[0] + "_bbcoords.npy"
        os.makedirs(os.path.dirname(output_bb_path) or '.', exist_ok=True)
        np.save(output_bb_path, np.array(bbcoords_list, dtype=object), allow_pickle=True)
        print(f"BB + coords saved ➜ {output_bb_path}")

    print(f"Visualisation video ➜ {output_movie_path}")
    return aggregated_mask


def run_prediction_for_npy_files(
    data_paths,
    model_names,
    model_dir,
    output_dir,
    cfg,
    possible_methods = ('duplicate_var_prep', 'custom_mix', 'neighbor_slices', 'duplicate'),
    possible_methods_intensity_map = None,
    full_size_predict = False,
    save_predicted_mask = False,
    save_prediction_bb = True,
    target_height = 1024,
    target_width = 1440,
    output_prefix = 'P03_clip_resized_2025',
    n_jobs = -1,
    fps = 1,
):
    """Run Mask R-CNN prediction for one or more prepared NumPy stacks.

    For each input ``.npy`` stack and each model checkpoint, this function
    infers the channel construction and intensity preprocessing from the model
    filename, prepares a three channel image stack, optionally resizes it,
    runs Detectron2 inference, and saves prediction products for later CTR
    integration.

    Parameters
    ----------
    data_paths : sequence of str
        Prepared detector stacks saved as ``.npy`` files.
    model_names : sequence of str
        Model checkpoint paths relative to ``model_dir``.
    model_dir : str or path
        Base directory containing trained model folders.
    output_dir : str or path
        Base directory where videos, masks, and bounding-box/coordinate files
        are written.
    cfg : detectron2 CfgNode
        Base Detectron2 config from :func:`build_mask_rcnn_cfg`.
    possible_methods : sequence of str
        Channel construction names that may be encoded in model filenames.
    possible_methods_intensity_map : dict, optional
        Mapping from method name fragments to preprocessing functions.
    full_size_predict : bool, default False
        If True, keep prepared images at their native shape. If False, resize
        to ``target_height`` × ``target_width`` before inference.
    save_predicted_mask : bool, default False
        Save dense per frame masks.
    save_prediction_bb : bool, default True
        Save bounding boxes and per instance mask coordinates.
    target_height, target_width : int
        Resize target used when ``full_size_predict=False``.
    output_prefix : str, default ``"P03_clip_resized_2025"``
        Experiment/workflow label used in output folder names.
    n_jobs : int, default -1
        Joblib parallelism for preprocessing and resizing.
    fps : int, default 1
        Frames per second for the visualization movie.

    Returns
    -------
    outputs : list of dict
        One record per completed scan/model pair with paths to written files.
    """
    _require_detectron2()
    if torch is None:
        raise ImportError("torch is required for inference.")

    if possible_methods_intensity_map is None:
        possible_methods_intensity_map = possible_methods_intensity

    outputs = []
    for _path in data_paths:
        _d = np.load(_path)
        scan_name_ = os.path.basename(_path).replace(".npy", "")
        for step, model_name in enumerate(model_names[:]):
            try:
                print(f'started step {step + 1}!\n')
                channel_method = which_method(model_name, possible_methods)
                print(f'{channel_method = }  ')
                if 'custom_mix' in channel_method:
                    intensity_method = None
                    intensity_method_name = 'custom_mix'
                    print(f'\n{intensity_method_name = }\n')
                else:
                    intensity_method = which_method_intensity(model_name, possible_methods_intensity_map)
                    intensity_method_name = intensity_method.__name__
                    print(f'\n{intensity_method_name = }\n')

                method_output_dir = prediction_output_dir(output_dir, scan_name_, model_name, output_prefix=output_prefix)
                cfg.OUTPUT_DIR = method_output_dir

                weights_path = os.path.join(model_dir, model_name)
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(f"Model weights not found: {weights_path}")
                cfg.MODEL.WEIGHTS = weights_path

                prepared_images = process_images_array_parallel(
                    _d,
                    ch_method=channel_method,
                    method=intensity_method,
                    n_jobs=n_jobs,
                )

                print(f'{model_name = }')
                _model_name = _model_stem(model_name)

                output_movie_path = os.path.join(cfg.OUTPUT_DIR, f"{scan_name_}-{_model_name}-{intensity_method_name}.mp4")
                output_masks_path = os.path.join(cfg.OUTPUT_DIR, f"{scan_name_}-{_model_name}-{intensity_method_name}_masks.npy")
                output_bb_path = os.path.join(cfg.OUTPUT_DIR, f"{scan_name_}-{_model_name}-{intensity_method_name}-bb.npy")
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

                if not full_size_predict:
                    original_height, original_width = prepared_images.shape[1:3]
                    interpolation = cv2.INTER_AREA if (original_height > target_height or original_width > target_width) else cv2.INTER_LINEAR
                    prepared_images = np.array(
                        Parallel(n_jobs=n_jobs)(
                            delayed(resize_image_worker)(img, target_width, target_height, interpolation)
                            for img in prepared_images
                        ),
                        dtype=np.float32,
                    )
                    inference_target_size = (target_height, target_width)
                    print(f'{full_size_predict = }\n{prepared_images.shape = }\n{output_movie_path = }\n\n')
                else:
                    inference_target_size = prepared_images.shape[1:3]
                    print(f'{prepared_images.shape = }\n{output_movie_path = }\n\n')

                predictor = DefaultPredictor(cfg)

                aggregated_mask = run_inference_with_default_predictor(
                    images_array=prepared_images,
                    predictor=predictor,
                    cfg=cfg,
                    output_movie_path=output_movie_path,
                    target_size=inference_target_size,
                    output_masks_path=output_masks_path,
                    fps=fps,
                    save_mask=save_predicted_mask,
                    save_bb=save_prediction_bb,
                    output_bb_path=output_bb_path,
                )

                aggregated_mask_path = None
                if aggregated_mask is not None:
                    aggregated_mask_path = os.path.join(cfg.OUTPUT_DIR, f"{scan_name_}-{_model_name}-{intensity_method_name}_max_pxl.png")
                    visualize_aggregated_mask(aggregated_mask, aggregated_mask_path)

                outputs.append({
                    "scan_name": scan_name_,
                    "model_name": model_name,
                    "intensity_method_name": intensity_method_name,
                    "output_movie_path": output_movie_path,
                    "output_masks_path": output_masks_path,
                    "output_bb_path": output_bb_path,
                    "aggregated_mask_path": aggregated_mask_path,
                })

                del prepared_images
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(f'finished step {step + 1}!\n{["#"] * 50}\n\n')
            except Exception as e:
                print(f" Error processing model '{model_name}': {str(e)}")
                print(f"Step is {step + 1}")
                print("Skipping this model...\n")
                continue
    return outputs

# # -----------------------------------------------------------------------------
# # Mask morphology helpers used before integration
# # -----------------------------------------------------------------------------


# def generate_halo(mask, expand_koef=0.00005):
#     """Generate an outer halo mask around a binary mask using contour dilation."""
#     mask_binary = (np.asarray(mask) > 0).astype(np.uint8)
#     contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     halo_mask = np.zeros_like(mask_binary, dtype=np.float32)
#     if len(contours) == 0:
#         return halo_mask

#     cv2.drawContours(halo_mask, contours, -1, 1, thickness=cv2.FILLED)

#     kernel_size = int(np.round(expand_koef * max(mask_binary.shape[:2])))
#     kernel_size = max(1, kernel_size)
#     kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
#     halo_mask = cv2.dilate(halo_mask, kernel)
#     halo_mask = np.subtract(halo_mask, mask_binary)

#     return halo_mask.astype(np.float32)


# def generate_dilate(mask,
#                     expand=0.00005,
#                     metric='percent',
#                     struct='rect',
#                     return_ring=False):
#     """Dilate a binary mask, optionally returning only the added outer ring.

#     This keeps backward compatibility with the notebook behavior: by default it
#     returns the full dilated mask.  Set ``return_ring=True`` to return only the
#     newly added halo outside the original mask.
#     """
#     mask_bin = (mask > 0).astype(np.uint8)

#     if metric == 'percent':
#         k = int(round(expand * max(mask.shape[:2])))
#     elif metric == 'pixels':
#         k = int(round(expand))
#     else:
#         raise ValueError("metric must be 'percent' or 'pixels'")

#     k = max(1, k)
#     kernel_shapes = {
#         'rect': cv2.MORPH_RECT,
#         'ellipse': cv2.MORPH_ELLIPSE,
#         'cross': cv2.MORPH_CROSS,
#     }
#     if struct not in kernel_shapes:
#         raise ValueError("struct must be one of 'rect', 'ellipse', or 'cross'")
#     kernel = cv2.getStructuringElement(kernel_shapes[struct], (k, k))

#     dilated = cv2.dilate(mask_bin, kernel)
#     if return_ring:
#         return (dilated - mask_bin).astype(np.float32)
#     return dilated.astype(np.float32)

# def grow_or_shrink(mask,
#                    expand=0.00005,      # >0 grow, <0 shrink
#                    metric='percent',    # 'percent' | 'pixels'
#                    struct='rect',       # 'rect' | 'ellipse' | 'cross'
#                    return_ring=False):
#     """
#     Dilate (grow) or erode (shrink) a binary mask.

#     Parameters
#     ----------
#     mask : 2-D ndarray
#         Binary mask (non-zeros are the object).
#     expand : float or int
#         +ve  ⇒ grow outward, -ve ⇒ shrink inward.
#         Interpreted as fraction of max(H,W) when *metric* == 'percent',
#         or as absolute pixels when *metric* == 'pixels'.
#     metric : {'percent', 'pixels'}
#         Units of *expand*.
#     struct : {'rect', 'ellipse', 'cross'}
#         Structuring-element shape.
#     return_ring : bool, default False
#         If True, return only the added (or removed) rim as 1s;
#         otherwise return the full dilated / eroded mask.

#     Returns
#     -------
#     out : float32 ndarray
#         Dilated / eroded mask, or the rim if *return_ring*.
#     """

#     # ── 1. clean binary mask ──────────────────────────────────────────────
#     mask_bin = (mask > 0).astype(np.uint8)

#     # ── 2. kernel size (always non-negative) ──────────────────────────────
#     if metric == 'percent':
#         k = int(round(abs(expand) * max(mask.shape)))
#     elif metric == 'pixels':
#         k = int(round(abs(expand)))
#     else:
#         raise ValueError("metric must be 'percent' or 'pixels'")
#     k = max(1, k)                                    # at least 1×1

#     # ── 3. structuring element ────────────────────────────────────────────
#     shapes = {'rect': cv2.MORPH_RECT,
#               'ellipse': cv2.MORPH_ELLIPSE,
#               'cross': cv2.MORPH_CROSS}
#     kernel = cv2.getStructuringElement(shapes.get(struct, cv2.MORPH_RECT),
#                                       (k, k))

#     # ── 4. choose operation: dilate <-> erode ───────────────────────────────
#     if expand > 0:
#         changed = cv2.dilate(mask_bin, kernel)
#         rim = changed - mask_bin          # outer halo
#     elif expand < 0:
#         changed = cv2.erode(mask_bin, kernel)
#         rim = mask_bin - changed          # inner rim
#     else:                                     # expand == 0
#         changed = mask_bin
#         rim = np.zeros_like(mask_bin)

#     return (rim if return_ring else changed).astype(np.float32)


# -----------------------------------------------------------------------------
# Legacy evaluation helpers kept from the notebooks
# -----------------------------------------------------------------------------


def evaluate_model(cfg, method, test_dataset_name, trainer_or_model, mapper=None):
    """
    Evaluate/visualize a Detectron2 model on a test dataset.
    ``mapper`` is optional.  If omitted, Detectron2's default test mapper is used.
    """
    _require_detectron2()
    if torch is None:
        raise ImportError("torch is required for evaluation.")

    model = trainer_or_model
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    kwargs = {"mapper": mapper} if mapper is not None else {}
    test_loader = build_detection_test_loader(cfg, test_dataset_name, **kwargs)
    return _eval(model, test_loader, method, cfg, test_dataset_name)


def _eval(model, test_loader, method, cfg, test_dataset_name):
    """Run inference on a Detectron2 test loader and save a prediction movie."""
    if torch is None:
        raise ImportError("torch is required for evaluation.")

    inputs_list = []
    predictions_list = []
    with torch.no_grad():
        for idx, inputs in enumerate(test_loader):
            outputs = model(inputs)
            inputs_list.extend(inputs)
            predictions_list.extend(outputs)

    output_movie_path = os.path.join(cfg.OUTPUT_DIR, "_predictions.mp4")
    if not os.path.exists(output_movie_path):
        visualize_predictions(inputs_list, predictions_list, cfg, method, test_dataset_name, output_movie_path)
    return {
        "num_inputs": len(inputs_list),
        "num_predictions": len(predictions_list),
        "output_movie_path": output_movie_path,
    }


def run_model(cfg, method, model_path, output_dir, test_datasets, npy_file_dir, mapper=None, output_group=None, model_dir=None):
    """Run the evaluation workflow for a saved model path.

    Parameters
    ----------
    cfg : detectron2 CfgNode
        Base Detectron2 configuration. The function clones it before editing.
    method : str
        Dataset/preprocessing method name stored in ``cfg.CUSTOM``.
    model_path : str
        Model weight path relative to ``model_dir``. If ``model_dir`` is not
        supplied, it is interpreted relative to ``output_dir`` for backward
        compatibility with the notebook version.
    output_dir : str or path
        Base directory for evaluation products.
    test_datasets : sequence of str
        Detectron2 dataset names to evaluate.
    npy_file_dir : str or path
        Directory with NumPy files used by the custom mapper.
    mapper : callable, optional
        Optional Detectron2 mapper passed to ``build_detection_test_loader``.
    output_group : str, optional
        Optional subfolder between ``output_dir`` and the model family folder.
    model_dir : str or path, optional
        Base directory for model weights. Defaults to ``output_dir``.

    Returns
    -------
    metrics : dict
        Evaluation summaries keyed by dataset name.
    """
    _require_detectron2()
    if torch is None:
        raise ImportError("torch is required for evaluation.")

    model_group = _model_group_name(model_path)
    if output_group is None or output_group == '':
        method_output_dir = os.path.join(output_dir, model_group)
    else:
        method_output_dir = os.path.join(output_dir, _safe_name(output_group), model_group)

    weights_root = output_dir if model_dir is None else model_dir
    weights_path = os.path.join(weights_root, model_path)

    cfg_new = cfg.clone()
    cfg_new.defrost()
    cfg_new.OUTPUT_DIR = method_output_dir
    cfg_new.MODEL.WEIGHTS = weights_path
    cfg_new.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_new.MODEL.ROI_HEADS.NUM_CLASSES = 1
    if CN is not None and 'CUSTOM' not in cfg:
        cfg_new.CUSTOM = CN()
    if CN is not None:
        cfg_new.CUSTOM.CUST_METHOD = method
        cfg_new.CUSTOM.MODEL_NAME = _model_stem(model_path)
        cfg_new.CUSTOM.NPY_FILE_DIR = npy_file_dir
    cfg_new.freeze()

    os.makedirs(cfg_new.OUTPUT_DIR, exist_ok=True)

    if len(test_datasets) == 0:
        raise ValueError("No test dataset specified!")

    metrics = {}
    for test_dataset_name in test_datasets:
        print(f"\n Evaluating on test dataset: {test_dataset_name}")

        cfg_eval = cfg_new.clone()
        cfg_eval.defrost()
        cfg_eval.DATASETS.TEST = (test_dataset_name,)
        cfg_eval.freeze()

        model = build_model(cfg_eval)
        checkpointer = DetectionCheckpointer(model)
        print(f'\n\n$$ {cfg_eval.MODEL.WEIGHTS = }$$\n\n')
        checkpointer.load(cfg_eval.MODEL.WEIGHTS)

        metrics[test_dataset_name] = evaluate_model(
            cfg_eval,
            method,
            test_dataset_name,
            model,
            mapper=mapper,
        )
    return metrics
