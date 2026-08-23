import glob
import linecache
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
from scipy.interpolate import interpn

import numpy as np
from PIL import Image
from pathlib import Path

try:
    import h5py
except Exception: 
    h5py = None

try:
    import hdf5plugin  
except Exception:  
    hdf5plugin = None


__all__ = [
    "not_empty_directory",
    "read_image",
    "read_metadata",
    "read_P07_imgs_with_metadata",
    "compute_max_pixel_image",
    "load_and_sum_npy_files",
    "rod_create_nested_folders",
    "find_eiger_h5_files",
    "load_eiger_h5_data",
    "prepare_p03_eiger_stack",
    "collect_npy_data_paths",
    "load_and_sum_predictions",
    "max_pxl_im",
    "I_error",
    "interpolate_image",
    "ID_31_read_schnucks_h5",
]




def interpolate_image(im, x, y, RR_r, RR_z):
    """
    Interpolate one detector image onto a precomputed reciprocal space grid.

    Parameters
    ----------
    im : ndarray of shape (n_y, n_x)
        Detector image to interpolate.
    x, y : ndarray
        Detector coordinate axes returned by `Q_grid` or `Q_grid2`.
    RR_r, RR_z : ndarray
        Reciprocal grid detector coordinate lookup arrays returned by
        `Q_grid` or `Q_grid2`.

    Returns
    -------
    q_image : ndarray
        Image interpolated onto the reciprocal space grid.
    """
    return interpn(
        (x, y),
        im.T,
        (RR_r, RR_z),
        method="linear",
        bounds_error=False,
        fill_value=0,
    )



def I_error(intensity, f_low=0.17, f_high=0.05, gamma=1, mode="poisson"):
    """
    Estimate absolute intensity errors for CTR intensity data.

    Invalid, non finite, and non positive intensity values are assigned NaN in
    the returned error array. Two empirical models are supported: a saturating
    fractional error model and a Poisson fractional error model.

    Parameters
    ----------
    intensity : array
        Intensity values. May contain ``None`` or ``NaN``.
    f_low : float, default 0.17
        Target fractional error at the lowest valid intensity scale.
    f_high : float, default 0.05
        Target fractional error at high intensity.
    gamma : float, default 1
        Shape exponent used by the ``'saturating'`` model.
    mode : {'poisson', 'saturating'}, default 'poisson'
        Error model. Matching is case insensitive.

    Returns
    -------
    sigma : ndarray
        Absolute error estimates with the same shape as `intensity`. Invalid
        input values are returned as NaN.

    Raises
    ------
    ValueError
        If ``f_low <= f_high``, if `gamma` is not positive, if either
        fractional error parameter is non positive, if no positive finite
        intensities are found, or if `mode` is unsupported.
    TypeError
        If `mode` is not a string.
    """
    if f_low <= f_high:
        raise ValueError("f_low must be greater than f_high.")
    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if f_low <= 0 or f_high <= 0:
        raise ValueError("f_low and f_high must be positive.")

    eps = 1e-12
    I_bg = 0.0 # background estimate

    x = np.asarray(intensity, dtype=float)  # None -> NaN when cast to float
    valid = np.isfinite(x) & (x > 0)        # strictly positive finite intensities

    if not np.any(valid):
        raise ValueError("No positive finite intensities found in 'intensity'.")

    sigma = np.full_like(x, np.nan, dtype=float)  # default NaN; fill only valid

    m = mode.lower()
    if m == "saturating":
        I0 = np.nanmedian(x[valid])  # robust scale; ignores NaNs by using 'valid'
        frac_valid = f_low - (f_low - f_high) * (x[valid] / (x[valid] + I0))**gamma
        sigma[valid] = frac_valid * x[valid]

    elif m == "poisson":
        f0 = f_high
        I_min = float(np.nanmin(x[valid]))
        c_sq = (f_low**2 - f0**2) * (I_min + I_bg)
        if c_sq < 0 or not np.isfinite(c_sq):
            c_sq = 0.0  # numeric guard
        frac_valid = np.sqrt(f0**2 + c_sq / (x[valid] + I_bg))
        sigma[valid] = frac_valid * x[valid]

    else:
        raise ValueError("mode must be 'saturating' or 'poisson'.")
    return sigma




def max_pxl_im(data):
    """
    Compute a maximum intensity projection over an image stack.

    Parameters
    ----------
    data : array of shape (n_images, height, width)
        Stack of detector images or masks.

    Returns
    -------
    max_im : ndarray of shape (height, width)
        Pixel wise maximum image over the first axis.
    """
    max_im = np.zeros(data.shape[1:])
    for im in data: 
        im = np.array(im)
        max_im = np.maximum(max_im, np.array(im))
    return max_im


def not_empty_directory(path_with_pattern, flag="bool"):
    """
    Check whether a glob pattern matches at least one file.

    Parameters
    ----------
    path_with_pattern : str or path
        File path pattern passed to ``glob.glob``.
    flag : {'bool', 'len'}, default 'bool'
        Output mode. If ``'bool'``, return whether at least one file matches.
        If ``'len'``, return the number of matched files.

    Returns
    -------
    result : bool or int
        Boolean match status when ``flag='bool'``; number of matched files
        when ``flag='len'``.

    Raises
    ------
    ValueError
        If `flag` is not ``'bool'`` or ``'len'``.
    """
    n_files = len(glob.glob(path_with_pattern))

    if flag == "bool":
        return n_files > 0
    if flag == "len":
        return n_files

    raise ValueError("flag must be 'bool' or 'len'")


def read_image(filename):
    """
    Read an image file as a NumPy float32 array.

    Parameters
    ----------
    filename : str or path
        Path to the image file.

    Returns
    -------
    image : ndarray
        Image data converted to ``float32``.
    """
    with Image.open(filename) as im:
        return np.array(im, dtype=np.float32)


def read_metadata(meta_path, regex, line_number=25):
    """
    Extract metadata values from a selected line of a metadata file.

    Parameters
    ----------
    meta_path : str or path
        Path to the metadata file.
    regex : re.Pattern
        Compiled regular expression used to extract values from the selected
        line.
    line_number : int, default 25
        One based line number read from each metadata file.

    Returns
    -------
    values : list of str
        Values matched by `regex` on the selected line.

    Raises
    ------
    FileNotFoundError
        If `meta_path` does not exist.
    ValueError
        If no values are matched on the selected line.
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    line = linecache.getline(meta_path, line_number)
    values = regex.findall(line)

    if not values:
        raise ValueError(
            f"No metadata values found in line {line_number} of {meta_path}"
        )

    return values


def read_P07_imgs_with_metadata(path, roix=None, roiy=None, metadata_line=25):
    """
    Read DESY P07 beamline detector images and corresponding metadata files.

    The function loads all image files matching `path`, reads the metadata file
    associated with each image, rotates the image stack by 180 degrees, and
    optionally applies a detector ROI.

    Parameters
    ----------
    path : str or path
        Glob pattern matching image files.
    roix : slice, ndarray, or sequence, optional
        Delta/x-pixel ROI applied after loading and rotating the image stack.
        The ROI is applied only when both `roix` and `roiy` are provided.
    roiy : slice, ndarray, or sequence, optional
        Gamma/y-pixel ROI applied after loading and rotating the image stack.
        The ROI is applied only when both `roix` and `roiy` are provided.
    metadata_line : int, default 25
        One line number read from each ``.metadata`` file.

    Returns
    -------
    img_arr : ndarray
        Loaded image stack. Shape is ``(n_images, height, width)`` without ROI,
        or ``(n_images, len(roiy), len(roix))`` depending on the ROI indexing.
    omes_list : list
        Metadata values extracted from each corresponding metadata file.

    Raises
    ------
    ValueError
        If no image files match `path`.
    FileNotFoundError
        If a corresponding metadata file is missing.
    """
    filenames = sorted(glob.glob(path))
    if not filenames:
        raise ValueError(f"Directory {path} is empty!")

    regex = re.compile(r"(-?\d+\.\d+)")
    num_images = len(filenames)

    with Image.open(filenames[0]) as first_image:
        img_height, img_width = first_image.size[::-1]

    img_arr_shape = (num_images, img_height, img_width)
    img_arr = np.empty(img_arr_shape, dtype=np.float32)
    omes_list = []

    with ThreadPoolExecutor() as executor:
        img_futures = [executor.submit(read_image, f) for f in filenames]
        meta_paths = [f + ".metadata" for f in filenames]
        meta_futures = [
            executor.submit(read_metadata, m, regex, metadata_line)
            for m in meta_paths
        ]

        for i, (img_future, meta_future) in enumerate(zip(img_futures, meta_futures)):
            img_arr[i] = img_future.result()
            omes_list.append([float(v) for v in meta_future.result()])

    img_arr = np.rot90(img_arr, k=2)

    if roiy is not None and roix is not None:
        return img_arr[:, roiy, roix], omes_list
    return img_arr, omes_list


def compute_max_pixel_image(path, roix=None, roiy=None):
    """
    Compute a streaming maximum intensity projection from image files.

    Images are loaded one by one from a glob pattern, rotated by 180 degrees,
    optionally cropped, and accumulated into a pixel wise maximum image.

    Parameters
    ----------
    path : str or path
        Glob pattern matching image files.
    roix : slice, ndarray, or sequence, optional
        Delta/x-pixel ROI. Applied only when both `roix` and `roiy` are
        provided.
    roiy : slice, ndarray, or sequence, optional
        Gamma/y-pixel ROI. Applied only when both `roix` and `roiy` are
        provided.

    Returns
    -------
    max_image : ndarray
        Pixel wise maximum image over all matched files.

    Raises
    ------
    ValueError
        If no files match `path`.
    """
    file_list = sorted(glob.glob(path))
    if not file_list:
        raise ValueError(f"Directory {path} is empty or no files match the pattern!")

    max_image = None
    for filename in file_list:
        with Image.open(filename) as img:
            img_array = np.rot90(img, k=-2)
            img_array = np.array(img_array)

            if roiy is not None and roix is not None:
                img_array = img_array[roiy, roix]

            if max_image is None:
                max_image = img_array
            else:
                np.maximum(max_image, img_array, out=max_image)

    return max_image


def load_and_sum_npy_files(dir_path, verbose=False):
    """
    Load all `.npy` files from a directory and return their element sum.

    Files are loaded one by one and summed in memory. All successfully loaded
    arrays must have the same shape. Files that cannot be loaded, or whose
    shapes do not match the accumulated sum, are skipped.

    Parameters
    ----------
    dir_path : str or path
        Path to the directory containing `.npy` files.
    verbose : bool, default False
        If True, print progress messages for loaded, skipped, and missing
        files.

    Returns
    -------
    arr_sum : ndarray or None
        Element sum of all successfully loaded `.npy` arrays. Returns
        None if the directory cannot be read, if no `.npy` files are found, or
        if no arrays are successfully loaded.
    """
    arr_sum = None

    try:
        npy_files = sorted(f for f in os.listdir(dir_path) if f.endswith(".npy"))
    except Exception as e:
        if verbose:
            print(f"Error listing directory {dir_path}: {e}")
        return None

    if not npy_files:
        if verbose:
            print(f"No .npy files found in directory: {dir_path}")
        return None

    for filename in npy_files:
        file_path = os.path.join(dir_path, filename)

        try:
            arr_loaded = np.load(file_path)

            if arr_sum is None:
                arr_sum = arr_loaded.copy()
                if verbose:
                    print(f"Initialized sum array with {filename}.")
            else:
                if arr_sum.shape != arr_loaded.shape:
                    raise ValueError(
                        f"Shape mismatch in {filename}: "
                        f"{arr_loaded.shape} vs {arr_sum.shape}"
                    )

                arr_sum += arr_loaded

                if verbose:
                    print(f"Added {filename} to sum array.")

        except (IOError, ValueError) as e:
            if verbose:
                print(f"Skipping {filename} due to error: {e}")
            continue

    if arr_sum is None and verbose:
        print("No arrays were successfully loaded and summed.")

    return arr_sum


def rod_create_nested_folders(dir_n, path=".", gamma_check=False, return_paths=False):
    """
    Create the standard folder structure for rod integration outputs.

    The function creates separate folders for data and images before and after
    correction. Optionally, it also creates a `gamma_check` folder inside the
    before correction image directory.

    Parameters
    ----------
    dir_n : str
        Name of the rod/output directory to create inside `path`.
    path : str or path, default "."
        Parent directory where the rod/output folder is created.
    gamma_check : bool, default False
        If True, create an additional `gamma_check` folder inside
        `before_correction/images`.
    return_paths : bool, default False
        If True, return a dictionary with the created folder paths. If False,
        return None.

    Returns
    -------
    paths : dict or None
        If `return_paths=True`, a dictionary with created folder paths.
        If `return_paths=False`, the function returns None.
    """
    base_folder = os.path.join(path, dir_n)

    after_correction_folder = os.path.join(base_folder, "after_correction")
    before_correction_folder = os.path.join(base_folder, "before_correction")

    data_after_correction_folder = os.path.join(after_correction_folder, "data")
    images_after_correction_folder = os.path.join(after_correction_folder, "images")

    data_before_correction_folder = os.path.join(before_correction_folder, "data")
    images_before_correction_folder = os.path.join(before_correction_folder, "images")

    os.makedirs(data_after_correction_folder, exist_ok=True)
    os.makedirs(images_after_correction_folder, exist_ok=True)

    os.makedirs(data_before_correction_folder, exist_ok=True)
    os.makedirs(images_before_correction_folder, exist_ok=True)

    gamma_check_folder = None
    if gamma_check:
        gamma_check_folder = os.path.join(images_before_correction_folder, "gamma_check")
        os.makedirs(gamma_check_folder, exist_ok=True)

    if return_paths:
        return {
            "base": base_folder,
            "after_correction": after_correction_folder,
            "before_correction": before_correction_folder,
            "data_after_correction": data_after_correction_folder,
            "images_after_correction": images_after_correction_folder,
            "data_before_correction": data_before_correction_folder,
            "images_before_correction": images_before_correction_folder,
            "gamma_check": gamma_check_folder,
        }

    return None


def find_eiger_h5_files(current_scan):
    """Find Eiger HDF5 master/data files inside one DESY P03 scan directory.

    Parameters
    ----------
    current_scan : str or path
        Directory containing the Eiger ``master`` file and one or more data
        HDF5 files.

    Returns
    -------
    h5_file_master : str or None
        Path containing ``"master"`` in the filename, if found.
    h5_file : str or None
        First non master HDF5 file.
    h5_files : list of str
        All files discovered directly inside ``current_scan``.
    """
    h5_files = []
    for name in os.listdir(current_scan):
        h5_files.append(os.path.join(current_scan, name))

    h5_file_master = None
    h5_file = None
    for f in h5_files:
        if "master" in f:
            h5_file_master = f
        else:
            h5_file = f
    return h5_file_master, h5_file, h5_files


def load_eiger_h5_data(h5_file, dataset_path="entry/data/data"):
    """Load an Eiger detector stack from an HDF5 file.

    Parameters
    ----------
    h5_file : str or path
        Eiger HDF5 data file.
    dataset_path : str, default ``"entry/data/data"``
        Dataset path inside the HDF5 file.

    Returns
    -------
    h5_data : ndarray
        Detector stack with shape ``(n_images, n_y, n_x)``.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    """
    if h5py is None:
        raise ImportError("h5py is required to read Eiger HDF5 data.")

    with h5py.File(h5_file, "r") as hdf:
        print("Keys: %s" % hdf["entry/data"].keys())
        h5_data = hdf[dataset_path][:]
    return h5_data


def prepare_p03_eiger_stack(
    h5_data,
    crop_y=slice(None, 2370),
    mean_threshold=4.2e9,
    clip_y=slice(1070, None),
    clip_x=slice(1030, None),
):
    """Prepare the DESY P03 Eiger stack and detector crop used for inference.

    The function follows the workflow: crop the unused detector y range, build
    a mean image, suppress very bright persistent pixels, and return both the
    filtered full size stack and the cropped stack used by the Mask R-CNN model.

    Parameters
    ----------
    h5_data : ndarray
        Raw detector stack with shape ``(n_images, n_y, n_x)``.
    crop_y : slice, default ``slice(None, 2370)``
        Initial y crop applied before mean image filtering.
    mean_threshold : float,`
        Pixels above this mean value are set to zero in all frames.
    clip_y, clip_x : slice
        Final detector crop used for the P03 cropped prediction case.

    Returns
    -------
    data_filtered : ndarray
        Full filtered stack after bright pixel suppression.
    data_filtered_clip : ndarray
        Cropped stack used as Mask R-CNN input.
    """
    h5_data = h5_data[:, crop_y, :]
    dd_mean = h5_data.mean(axis=0)
    mask = dd_mean > mean_threshold
    data_filtered = h5_data.copy()
    data_filtered[:, mask] = 0
    data_filtered_clip = data_filtered[:, clip_y, clip_x].copy()
    return data_filtered, data_filtered_clip


def collect_npy_data_paths(processed_data_path):
    """Collect prepared ``.npy`` detector stacks from subfolders.

    Parameters
    ----------
    processed_data_path : str or path
        Directory whose immediate subfolders contain prepared NumPy stacks.

    Returns
    -------
    data_paths : list of str
        Sorted list of discovered ``.npy`` files.
    """
    folders = sorted(os.listdir(processed_data_path))
    data_paths = []
    # for folder in folders:
    for file_name in sorted(os.listdir(folder_path)):
        folder_path = os.path.join(processed_data_path, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(folder_path, file_name)
                    data_paths.append(file_path)
    return data_paths


def _boxes_to_mask(boxes_scores: np.ndarray, height: int, width: int, score_thresh: float) -> np.ndarray:
    """Rasterise a *(N,5)* array of boxes into a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if boxes_scores.size == 0:
        return mask
    keep = boxes_scores[:, 4] >= score_thresh
    for x1, y1, x2, y2, _ in boxes_scores[keep]:
        x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))
        mask[y1_i : y2_i + 1, x1_i : x2_i + 1] = 1
    return mask


def _coords_to_mask(coords_list, height, width) -> np.ndarray:
    """Rasterise a list of (row,col) arrays into a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for coords in coords_list:
        if coords.size:          # skip empty instances
            rows, cols = coords[:, 0], coords[:, 1]
            mask[rows, cols] = 1
    return mask


def _bb_list_to_stack(
    bb_obj_arr: np.ndarray,
    height: int,
    width: int,
    score_thresh: float,
    COORDS_ONLY: bool = True,
) -> np.ndarray:
    """Convert object array (per frame) to mask stack, handling dict format."""
    print(f"{score_thresh = }\\n")
    masks = []
    for frame_entry in bb_obj_arr:
        if COORDS_ONLY:
            if isinstance(frame_entry, dict) and "coords" in frame_entry:
                coords_list = frame_entry["coords"]
                # Filter using bbox scores if available
                if "bb" in frame_entry and frame_entry["bb"].size > 0:
                    bb_scores = frame_entry["bb"][:, 4]
                    if len(bb_scores) != len(coords_list):
                        print(f"Warning: {len(coords_list)} masks but {len(bb_scores)} scores")
                    else:
                        # Filter masks using bbox scores
                        keep = bb_scores >= score_thresh
                        coords_list = [c for i, c in enumerate(coords_list) if keep[i]]
                mask = _coords_to_mask(coords_list, height, width)
            else:
                # Skip any bounding box representations
                mask = np.zeros((height, width), dtype=np.uint8)
        else:
            if isinstance(frame_entry, dict):
                if "coords" in frame_entry:
                    coords_list = frame_entry["coords"]
                    # Filter using bbox scores if available
                    if "bb" in frame_entry and frame_entry["bb"].size > 0:
                        bb_scores = frame_entry["bb"][:, 4]
                        if len(bb_scores) != len(coords_list):
                            print(f"Warning: {len(coords_list)} masks but {len(bb_scores)} scores")
                        else:
                            # Filter masks using bbox scores
                            keep = bb_scores >= score_thresh
                            coords_list = [c for i, c in enumerate(coords_list) if keep[i]]
                    mask = _coords_to_mask(coords_list, height, width)
                else:
                    mask = _boxes_to_mask(
                        frame_entry.get("bb", np.empty((0, 5))),
                        height,
                        width,
                        score_thresh,
                    )
            else:
                mask = _boxes_to_mask(frame_entry, height, width, score_thresh)
        masks.append(mask)
    return np.stack(masks, axis=0).astype(np.int32)


def load_and_sum_predictions(
    npy_files: Optional[List[str]] = None,
    dir_path: Optional[str] = None,
    target_shape: Tuple[int, int] = (1024, 1440),
    score_thresh: float = 0.066,
) -> Optional[np.ndarray]:
    """Aggregate predictions from multiple *.npy* files into *(T,H,W)* array."""

    # ---------------------------------------------------------------
    # Resolve list of files
    # ---------------------------------------------------------------
    if npy_files is None:
        if dir_path is None:
            raise ValueError("Either *npy_files* or *dir_path* must be provided.")
        try:
            npy_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".npy")]
        except Exception as e:
            print(f"Error listing directory {dir_path}: {e}")
            return None
    else:
        # Accept numpy arrays, tuples, whatever → convert to plain list
        if isinstance(npy_files, np.ndarray):
            npy_files = npy_files.tolist()
        else:
            npy_files = list(npy_files)

    if len(npy_files) == 0:
        print("No .npy files to load.")
        return None

    height, width = target_shape
    arr_sum: Optional[np.ndarray] = None
    frames_expected: Optional[int] = None

    # ---------------------------------------------------------------
    # Iterate through each file and accumulate
    # ---------------------------------------------------------------
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        try:
            arr_loaded = np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"Skipping {filename}: load failed – {e}")
            continue

        if arr_loaded.dtype == np.uint8 and arr_loaded.ndim == 3:
            file_masks = arr_loaded.astype(np.int32)
        elif arr_loaded.dtype == object:
            file_masks = _bb_list_to_stack(arr_loaded, height, width, score_thresh)
        else:
            print(f"Skipping {filename}: unsupported layout (dtype={arr_loaded.dtype}, ndim={arr_loaded.ndim})")
            continue

        # Frame count consistency
        if frames_expected is None:
            frames_expected = file_masks.shape[0]
        elif file_masks.shape[0] != frames_expected:
            print(f"Skipping {filename}: frame count mismatch {file_masks.shape[0]} vs {frames_expected}")
            continue

        if arr_sum is None:
            arr_sum = file_masks.copy()
            print(f"Initialized accumulator with {filename} – shape {arr_sum.shape}.")
        else:
            if arr_sum.shape != file_masks.shape:
                print(f"Skipping {filename}: shape mismatch {file_masks.shape} vs {arr_sum.shape}")
                continue
            arr_sum += file_masks
            print(f"Added {filename} to accumulator.")

    if arr_sum is None:
        print("No arrays were successfully loaded and summed.")
    return arr_sum



def find_data_path(hdf):
    """
    Find the detector image dataset inside an HDF5 file.

    The function searches recursively through the HDF5 structure for a
    three dimensional dataset named ``data``. 
    Exactly one matching dataset is expected.

    Parameters
    ----------
    hdf : h5py.File or h5py.Group
        Open HDF5 file or group to search.

    Returns
    -------
    data_path : str
        Path to the detected three dimensional ``data`` dataset.

    Raises
    ------
    ValueError
        If no matching dataset is found or if multiple matching datasets
        are present.
    """
    datasets = []

    def visitor(name, obj):
        if (
            isinstance(obj, h5py.Dataset)
            and obj.ndim == 3
            and name.split("/")[-1] == "data"
        ):
            datasets.append(name)

    hdf.visititems(visitor)

    if not datasets:
        raise ValueError("No 3D 'data' dataset found")

    if len(datasets) > 1:
        raise ValueError(
            f"Multiple 3D 'data' datasets found: {datasets}"
        )

    return datasets[0]



def ID_31_read_schnucks_h5(folder, pattern="*.h5"):
    """
    Read ESRF ID31 Schnucks HDF5 detector files from a folder.

    All files matching the specified pattern are read in sorted order.
    The detector image dataset is located automatically using
    `find_data_path`, and the resulting arrays are concatenated
    along the frame axis.

    Parameters
    ----------
    folder : str or pathlib.Path
        Path to the folder containing the HDF5 files.
    pattern : str, optional
        File pattern used to select HDF5 files. Default is ``*.h5``.

    Returns
    -------
    data : ndarray
        Detector images concatenated along the first frame axis, with shape
        ``(n_frames, n_rows, n_columns)``.

    Raises
    ------
    FileNotFoundError
        If no files matching ``pattern`` are found in ``folder``.
    ValueError
        If an HDF5 file contains no suitable three dimensional ``data``
        dataset or contains multiple matching datasets.
    """
    files = sorted(Path(folder).glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {folder}"
        )

    data_all = []

    for file in files:
        with h5py.File(file, "r") as hdf:
            data_path = find_data_path(hdf)
            data = hdf[data_path][:]

        print(f"{file.name}: {data_path}, shape={data.shape}")
        data_all.append(data)

    data = np.concatenate(data_all, axis=0)

    print(f"Total shape: {data.shape}")

    return data
