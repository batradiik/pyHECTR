import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import re
import linecache
from concurrent.futures import ThreadPoolExecutor
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interpn
import imageio
import os



def not_empty_directory(path_with_pattern, flag="bool"):
    """
    Check whether a glob pattern matches at least one file.

    Parameters
    ----------
    path_with_pattern : str or path-like
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




def max_pxl_im(data):
    """
    Compute a maximum-intensity projection over an image stack.

    Parameters
    ----------
    data : array-like of shape (n_images, height, width)
        Stack of detector images or masks.

    Returns
    -------
    max_im : ndarray of shape (height, width)
        Pixel-wise maximum image over the first axis.
    """
    max_im = np.zeros(data.shape[1:])
    for im in data: 
        im = np.array(im)
        max_im = np.maximum(max_im, np.array(im))
    return max_im




def read_image(filename):
    """
    Read an image file as a NumPy float32 array.

    Parameters
    ----------
    filename : str or path-like
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
    meta_path : str or path-like
        Path to the metadata file.
    regex : re.Pattern
        Compiled regular expression used to extract values from the selected
        line.
    line_number : int, default 25
        One-based line number read from the metadata file.

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




def plot_max_pxl(data, vmin=0, vmax=400):
    """
    Plot and return the maximum-intensity projection of an image stack.

    Parameters
    ----------
    data : array-like of shape (n_images, height, width)
        Image stack.
    vmin, vmax : float, default 0 and 400
        Color-scale limits passed to ``imshow``.

    Returns
    -------
    max_data : ndarray of shape (height, width)
        Pixel-wise maximum projection over the first axis.
    """
    max_data = data.copy()
    max_data = max_data.max(axis = 0)
    plt.figure()
    plt.imshow(max_data, vmin=vmin , vmax=vmax)
    plt.show()
    return max_data



def read_P07_imgs_with_metadata(path, roix=None, roiy=None, metadata_line=25):
    """
    Read DESY P07 beamline detector images and corresponding metadata files.

    The function loads all image files matching `path`, reads the metadata file
    associated with each image, rotates the image stack by 180 degrees, and
    optionally applies a detector ROI.

    Parameters
    ----------
    path : str or path-like
        Glob pattern matching image files.
    roix : slice, ndarray, or sequence, optional
        Delta/x-pixel ROI applied after loading and rotating the image stack.
        The ROI is applied only when both `roix` and `roiy` are provided.
    roiy : slice, ndarray, or sequence, optional
        Gamma/y-pixel ROI applied after loading and rotating the image stack.
        The ROI is applied only when both `roix` and `roiy` are provided.
    metadata_line : int, default 25
        One-based line number read from each ``.metadata`` file.

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
        raise ValueError(f'Directory {path} is empty!')

    regex = re.compile(r'(-?\d+\.\d+)')
    num_images = len(filenames)

    # Read the first image to get shape
    with Image.open(filenames[0]) as first_image:
        img_height, img_width = first_image.size[::-1]

    img_arr_shape = (num_images, img_height, img_width)
    img_arr = np.empty(img_arr_shape, dtype=np.float32)
    omes_list = []

    with ThreadPoolExecutor() as executor:
        # Read images and metadata in parallel
        img_futures = [executor.submit(read_image, f) for f in filenames]
        meta_paths = [f + '.metadata' for f in filenames]
        meta_futures = [executor.submit(read_metadata, m, regex) for m in meta_paths]

        for i, (img_future, meta_future) in enumerate(zip(img_futures, meta_futures)):
            img_arr[i] = img_future.result()
            # omes_list.append(meta_future.result())
            omes_list.append([float(v) for v in meta_future.result()])

    # Rotate all images at once by flipping both axes
    img_arr = np.rot90(img_arr, k=2)

    if roiy is not None and roix is not None:
        return img_arr[:, roiy, roix], omes_list
    else:
        return img_arr, omes_list



def om_profile_slider(data_raw, data_sbt, data_bckgd, bin_omega_rate, bin_gamma_rate, bin_delta_rate,
                      l_bin_rate, omega_values, omega_windows, L_values, gamma_windows,
                      fig_size=(13, 7), save_animate=False, movie_path=None):
    """
    Display raw, background, and background-subtracted omega profiles.

    The function creates an interactive slider over binned gamma positions.
    For each slider position, it plots the raw omega profile, estimated
    background profile, and background-subtracted profile. 
    Optionally, the slider frames can be saved as an animated GIF.

    Parameters
    ----------
    data_raw : sequence of ndarray
        Raw omega profiles, one per plotted gamma bin.
    data_sbt : sequence of ndarray
        Background-subtracted omega profiles, one per plotted gamma bin.
    data_bckgd : sequence of ndarray
        Estimated background profiles, one per plotted gamma bin.
    bin_omega_rate : int or float
        Omega binning parameter shown in the plot legend.
    bin_gamma_rate : int or float
        Gamma binning parameter shown in the plot legend.
    bin_delta_rate : int or float
        Delta binning parameter shown in the plot legend.
    l_bin_rate : int or float
        L-bin width shown in the subtracted-profile title.
    omega_values : ndarray
        Omega values indexed by the slices in `omega_windows`.
    omega_windows : sequence of slice
        Omega windows corresponding to each plotted profile.
    L_values : array-like
        L values or labels shown in the raw-profile title.
    gamma_windows : sequence
        Gamma windows associated with the plotted profiles. Currently used
        only for diagnostic printing during animation errors.
    fig_size : tuple of float, default (13, 7)
        Figure size.
    save_animate : bool, default False
        If True, save all slider frames to `movie_path`.
    movie_path : str or path-like, optional
        Output path for the saved animation. Required when
        ``save_animate=True``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    ax : dict of matplotlib.axes.Axes
        Axes dictionary with keys ``'raw'`` and ``'sbt'``.
    """
    _save = False
    raw_ = data_raw
    sbt_ = data_sbt
    bckgd_ = data_bckgd
    fig, ax = plt.subplot_mosaic([
            ['raw', 'sbt']  ], figsize=fig_size)

    initial_raw = raw_[0]
    omega_win_curr = omega_values[omega_windows[0]]
    delta_omega_win_curr = omega_win_curr[0] - omega_win_curr[-1]
    # L_value = np.round((L_values[gamma_windows[0][1]] + L_values[gamma_windows[0][0]])/2, 3)
    L_value = L_values
    
    raw_plot = ax['raw'].plot(omega_win_curr, initial_raw, '*')

    tmp_omega = np.round(delta_omega_win_curr, 4)
    l_bin_rate = round(l_bin_rate, 4)
    
    # legend_text = ax['raw'].set_title(f'Raw int, L, $\Delta \omega$ = { np.round(L_value[0], 4), tmp_omega }')
    ax['raw'].set_title(
            f'Raw int; $L$, $\Delta \omega$ = ({float(L_value[0]):.4f}, {float(tmp_omega):.4f})'
    )

    initial_sbt = sbt_[0]
    sbt_plot = ax['sbt'].plot(omega_win_curr, initial_sbt, '*', label=f'bin $\omega$ ={bin_omega_rate}\nbin $\gamma$ ={bin_gamma_rate}\nbin $\delta$ ={bin_delta_rate}')
    ax['sbt'].set_title(f'Subtracted int, $\Delta L$ = {l_bin_rate}')

    initial_bckgd = bckgd_[0]
    bckgd_plot = ax['raw'].plot(omega_win_curr, initial_bckgd, 'v', color='y', label='background')

    ax_slider_image = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_image = Slider(ax_slider_image, 'binned gamma pxl', 0, len(raw_) - 1, valinit=0, valstep=1)

    ax['raw'].legend()
    ax['sbt'].legend()
    
    ax['sbt'].set_ylabel('img/$\omega$ window')
    ax['sbt'].set_xlabel('$I$')
        
    ax['raw'].set_ylabel('img/$\omega$ window')
    ax['raw'].set_xlabel('$I$')

    def update(val, l_bin_rate = l_bin_rate):
        if _save:
           index = int(val) 
        else:
            index = int(slider_image.val)
        ax['raw'].clear()
        ax['sbt'].clear()
        
        omega_win_curr = omega_values[omega_windows[index]]
        delta_omega_win_curr = omega_win_curr[0] - omega_win_curr[-1]
        tmp_omega = np.round(delta_omega_win_curr, 4)
        
        # L_value = np.round((L_values[gamma_windows[index][1]] + L_values[gamma_windows[index][0]])/2, 3)
        
        raw_line = ax['raw'].plot(omega_win_curr, raw_[index], '*')[0]
        bckgd_line = ax['raw'].plot(omega_win_curr, bckgd_[index], 'v', color='y', label='background')[0]
        sbt_line = ax['sbt'].plot(omega_win_curr, sbt_[index], '*', label=f'bin $\omega$ ={bin_omega_rate}\nbin $\gamma$ ={bin_gamma_rate}\nbin $\delta$ ={bin_delta_rate}')[0]
        
        #vax['raw'].set_title(f'Raw int, L, $\Delta \omega$ = { np.round(L_value[index],4), tmp_omega }')
        ax['raw'].set_title(
                        f'Raw int, $L$, $\Delta \omega$ = ({float(L_value[index]):.4f}, {float(tmp_omega):.4f})'
        )
        ax['sbt'].set_title(f'Subtracted int, $\Delta L$ = {l_bin_rate}')
        ax['raw'].legend()
        ax['sbt'].legend()

        ax['sbt'].set_xlabel('img/$\omega$ window')
        ax['sbt'].set_ylabel('$Int$')
        
        ax['raw'].set_xlabel('img/$\omega$ window')
        ax['raw'].set_ylabel('$Int$')
        fig.canvas.draw_idle()
    
    if save_animate:
        _save = True 
        anim = FuncAnimation(fig, update, frames=len(raw_), interval=20, repeat=False)
        if movie_path is not None:
            images = []
            print('\n... saving gif ... \n')

            # for i in range(len(raw_)-1):
            for i in range(len(raw_)):
                try:
                    update(i)
                except IndexError as exp:
                    # print(f'{gamma_windows[i-1] = }, {np.round(L_values[i-1],5) = }, {np.round(L_values[i-1],5) = }')
                    print(f"{i = }, {gamma_windows[i] = }, {np.round(L_values[i], 5) = }")
                    # print(f'{i = }, {gamma_windows[index] = }, {np.round(L_values[i],5) = }')
                    raise exp
                fig.canvas.draw()  # Draw the figure
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(image)
                
            imageio.mimsave(movie_path, images, fps=1)
            print('... done ... \n')
        else:
            raise ValueError('path should not be None')
        _save = False

    slider_image.on_changed(update)
    plt.tight_layout()
    plt.show()
    return fig, ax



def image_slider(images, fig_size=(7, 7), vmax1=20, vmin1=None, cmap1='viridis'):
    """
    Display an interactive slider for browsing an image stack.

    Parameters
    ----------
    images : array-like of shape (n_images, height, width)
        Image stack to display.
    fig_size : tuple of float, default (7, 7)
        Figure size.
    vmax1 : float, default 20
        Upper color-scale limit.
    vmin1 : float, optional
        Lower color-scale limit. If None, Matplotlib chooses the lower limit.
    cmap1 : str, default 'viridis'
        Colormap passed to ``imshow``.

    Returns
    -------
    None
        The function displays the interactive figure but does not currently
        return ``fig`` or ``ax``.
    """
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    initial_image = images[0]  # Use one channel for original image
    if vmin1 is None:
        image_plot = ax.imshow(initial_image, cmap=cmap1, vmax=vmax1)
    else:
        image_plot = ax.imshow(initial_image, cmap=cmap1, vmax=vmax1, vmin=vmin1)

    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Image Index', 0, len(images) - 1, valinit=0, valstep=1)

    def update(val):
        index = int(slider.val)
        image_plot.set_array(images[index])  # Use one channel for original image
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)  # Ensure the images are plotted for the initial index
    plt.show()



def image_mask_slider(images, masks, vmax_im1=250, vmax_im2=1,
                      fig_size=(12, 6), cmap1='viridis',
                      vmin_im1=0, vmin_im2=0):
    """
    Display an interactive slider comparing images and masks frame by frame.

    Parameters
    ----------
    images : array-like of shape (n_images, height, width)
        Image stack.
    masks : array-like of shape (n_images, height, width)
        Mask stack aligned with `images`.
    vmax_im1, vmin_im1 : float
        Color-scale limits for the image panel.
    vmax_im2, vmin_im2 : float
        Color-scale limits for the mask panel.
    fig_size : tuple of float, default (12, 6)
        Figure size.
    cmap1 : str, default 'viridis'
        Colormap used for the image panel.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.
    ax : dict of matplotlib.axes.Axes
        Axes dictionary with keys ``'img'`` and ``'mask'``.

    Raises
    ------
    ValueError
        If `images` and `masks` do not contain the same number of frames.
    """
    image_stack = images
    mask_stack = masks

    if len(images) != len(masks):
        raise ValueError("images and masks must have the same number of frames")
        
    fig, ax = plt.subplot_mosaic([
    ['img', 'mask']  ], figsize=fig_size)

    initial_image = image_stack[0, :, :]
    img_plot = ax['img'].imshow(initial_image, vmin=vmin_im1, vmax=vmax_im1, cmap=cmap1)

    initial_mask = mask_stack[0, :, :]
    mask_plot = ax['mask'].imshow(initial_mask, vmin=vmin_im2, vmax=vmax_im2)

    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    # slider = Slider(ax_slider, 'Image Index', 1, len(image_stack), valinit=0, valstep=1)
    slider = Slider(
        ax_slider,
        "Image Index",
        0,
        len(image_stack) - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        index = int(slider.val)
        img_plot.set_array(image_stack[index, :, :])
        mask_plot.set_array(mask_stack[index, :, :])
        fig.canvas.draw_idle()

    slider.on_changed(update)    
    return fig, ax



def compute_max_pixel_image(path, roix=None, roiy=None):
    """
    Compute a streaming maximum-intensity projection from image files.

    Images are loaded one by one from a glob pattern, rotated by 180 degrees,
    optionally cropped, and accumulated into a pixel-wise maximum image.

    Parameters
    ----------
    path : str or path-like
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
        Pixel-wise maximum image over all matched files.

    Raises
    ------
    ValueError
        If no files match `path`.
    """
    file_list = sorted(glob.glob(path))
    if not file_list:
        raise ValueError(f'Directory {path} is empty or no files match the pattern!')
    max_image = None
    for idx, filename in enumerate(file_list):
        with Image.open(filename) as img:
            # Rotate the image by 180 degrees
            img_array = np.rot90(img, k = -2)
            img_array = np.array(img_array)
            # Apply ROI if specified
            if roiy is not None and roix is not None:
                img_array = img_array[roiy, roix][:, ::-1]
            if max_image is None:
                max_image = img_array[:, ::-1]
            else:
                np.maximum(max_image, img_array, out=max_image)
    
    return max_image 



def load_and_sum_npy_files(dir_path, verbose=False):
    """
    Load all `.npy` files from a directory and return their element-wise sum.

    Files are loaded one by one and summed in memory. All successfully loaded
    arrays must have the same shape. Files that cannot be loaded, or whose
    shapes do not match the accumulated sum, are skipped.

    Parameters
    ----------
    dir_path : str or path-like
        Path to the directory containing `.npy` files.
    verbose : bool, default False
        If True, print progress messages for loaded, skipped, and missing
        files.

    Returns
    -------
    arr_sum : ndarray or None
        Element-wise sum of all successfully loaded `.npy` arrays. Returns
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
    Create the standard folder structure for rod-integration outputs.

    The function creates separate folders for data and images before and after
    correction. Optionally, it also creates a `gamma_check` folder inside the
    before-correction image directory.

    Parameters
    ----------
    dir_n : str
        Name of the rod/output directory to create inside `path`.
    path : str or path-like, default "."
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
        If `return_paths=True`, a dictionary with the created paths:

        - ``'base'``
        - ``'after_correction'``
        - ``'before_correction'``
        - ``'data_after_correction'``
        - ``'images_after_correction'``
        - ``'data_before_correction'``
        - ``'images_before_correction'``
        - ``'gamma_check'``

        The ``'gamma_check'`` value is None when `gamma_check=False`.
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
    


def I_error(intensity, f_low=0.17, f_high=0.05, gamma=1, mode="poisson"):
    """
    Estimate absolute intensity errors for CTR intensity data.

    Invalid, non-finite, and non-positive intensity values are assigned NaN in
    the returned error array. Two empirical models are supported: a saturating
    fractional-error model and a Poisson fractional error model.

    Parameters
    ----------
    intensity : array-like
        Intensity values. May contain ``None`` or ``NaN``.
    f_low : float, default 0.17
        Target fractional error at the lowest valid intensity scale.
    f_high : float, default 0.05
        Target fractional error at high intensity.
    gamma : float, default 1
        Shape exponent used by the ``'saturating'`` model.
    mode : {'poisson', 'saturating'}, default 'poisson'
        Error model. Matching is case-insensitive.

    Returns
    -------
    sigma : ndarray
        Absolute error estimates with the same shape as `intensity`. Invalid
        input values are returned as NaN.

    Raises
    ------
    ValueError
        If ``f_low <= f_high``, if `gamma` is not positive, if either
        fractional-error parameter is non-positive, if no positive finite
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


def image_slider_Q(images, q_r, q_z, omes, vmin_=0, vmax_=250,
                   fig_size=(10, 6), show_axis=True, equal_aspect=True,
                   return_fig=False):
    """
    Display an interactive slider for a stack of reciprocal-space images.

    Parameters
    ----------
    images : array-like of shape (n_images, n_qz, n_qr)
        Stack of images to display.
    q_r : ndarray of shape (n_qr,)
        Reciprocal-space x-axis values.
    q_z : ndarray of shape (n_qz,)
        Reciprocal-space y-axis values.
    omes : array-like of shape (n_images,)
        Omega values or labels shown in the title for each image.
    vmin_ : float, default 0
        Lower color-scale limit.
    vmax_ : float, default 250
        Upper color-scale limit.
    fig_size : tuple of float, default (10, 6)
        Figure size.
    show_axis : bool, default True
        If False, hide tick marks and tick labels.
    equal_aspect : bool, default True
        If True, use equal aspect ratio. If False, use automatic aspect ratio.
    return_fig : bool, default False
        If True, return ``(fig, ax)``. If False, return None.

    Returns
    -------
    fig, ax : tuple, optional
        Returned only when `return_fig=True`.
    """
    images = np.asarray(images)
    num_images = images.shape[0]

    fig, ax = plt.subplots(figsize=fig_size)
    plt.subplots_adjust(bottom=0.25)

    aspect = 'equal' if equal_aspect else 'auto'
    img_plot = ax.imshow(
        images[0],
        extent=[q_r.min(), q_r.max(), q_z.min(), q_z.max()],
        origin='lower',
        aspect=aspect,
        cmap='viridis',
        vmin=vmin_,
        vmax=vmax_
    )

    font1 = {'family': 'sans-serif', 'color': 'black', 'size': 14}
    ax.set_xlabel('$q_{xy}$ [$\mathrm{\AA}^{-1}$]', fontdict=font1)
    ax.set_ylabel('$q_{z}$ [$\mathrm{\AA}^{-1}$]', fontdict=font1)

    cbar = fig.colorbar(img_plot, ax=ax, orientation='vertical')
    cbar.ax.set_ylabel('Intensity', fontdict=font1)

    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(
        ax=ax_slider,
        label='Image Index',
        valmin=0,
        valmax=num_images - 1,
        valinit=0,
        valstep=1,
        color='blue'
    )

    if not show_axis:
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)

    def update(val):
        index = int(slider.val)
        img_plot.set_data(images[index])
        ax.set_title(f'omes -  {omes[index]}', fontdict=font1)
        fig.canvas.draw_idle()

    ax.set_title(f'omes -  {omes[0]}', fontdict=font1)
    slider.on_changed(update)

    plt.show()

    if return_fig:
        return fig, ax

    return None


def interpolate_image(im, x, y, RR_r, RR_z):
    """
    Interpolate one detector image onto a precomputed reciprocal-space grid.

    Parameters
    ----------
    im : ndarray of shape (n_y, n_x)
        Detector image to interpolate.
    x, y : ndarray
        Detector-coordinate axes returned by `Q_grid` or `Q_grid2`.
    RR_r, RR_z : ndarray
        Reciprocal-grid detector-coordinate lookup arrays returned by
        `Q_grid` or `Q_grid2`.

    Returns
    -------
    q_image : ndarray
        Image interpolated onto the reciprocal-space grid.
    """
    return interpn(
        (x, y),
        im.T,
        (RR_r, RR_z),
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

