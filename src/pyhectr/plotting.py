import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg

__all__ = [
    "plot_max_pxl",
    "om_profile_slider",
    "image_slider",
    "image_mask_slider",
    "image_slider_Q",
    "image_movie",
    "image_mask_movie",
]


def plot_max_pxl(data, vmin=0, vmax=400):
    """
    Plot and return the maximum intensity projection of an image stack.

    Parameters
    ----------
    data : array of shape (n_images, height, width)
        Image stack.
    vmin, vmax : float, default 0 and 400
        Color scale limits passed to ``imshow``.

    Returns
    -------
    max_data : ndarray of shape (height, width)
        Pixel maximum projection over the first axis.
    """
    max_data = data.copy()
    max_data = max_data.max(axis=0)
    plt.figure()
    plt.imshow(max_data, vmin=vmin, vmax=vmax)
    plt.show()
    return max_data


def om_profile_slider(data_raw, data_sbt, data_bckgd, bin_omega_rate, bin_gamma_rate, bin_delta_rate,
                      l_bin_rate, omega_values, omega_windows, L_values, gamma_windows,
                      fig_size=(13, 7), save_animate=False, movie_path=None):
    """
    Display raw, background, and background subtracted omega profiles.

    The function creates an interactive slider over binned gamma positions.
    For each slider position, it plots the raw omega profile, estimated
    background profile, and background subtracted profile.
    Optionally, the slider frames can be saved as an animated GIF.

    Parameters
    ----------
    data_raw : sequence of ndarray
        Raw omega profiles, one per plotted gamma bin.
    data_sbt : sequence of ndarray
        Background subtracted omega profiles, one per plotted gamma bin.
    data_bckgd : sequence of ndarray
        Estimated background profiles, one per plotted gamma bin.
    bin_omega_rate : int or float
        Omega binning parameter shown in the plot legend.
    bin_gamma_rate : int or float
        Gamma binning parameter shown in the plot legend.
    bin_delta_rate : int or float
        Delta binning parameter shown in the plot legend.
    l_bin_rate : int or float
        L bin width shown in the subtracted profile title.
    omega_values : ndarray
        Omega values indexed by the slices in `omega_windows`.
    omega_windows : sequence of slice
        Omega windows corresponding to each plotted profile.
    L_values : array
        L values or labels shown in the raw profile title.
    gamma_windows : sequence
        Gamma windows associated with the plotted profiles. Currently used
        only for diagnostic printing during animation errors.
    fig_size : tuple of float, default (13, 7)
        Figure size.
    save_animate : bool, default False
        If True, save all slider frames to `movie_path`.
    movie_path : str or path, optional
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
        ["raw", "sbt"]
    ], figsize=fig_size)

    initial_raw = raw_[0]
    omega_win_curr = omega_values[omega_windows[0]]
    delta_omega_win_curr = omega_win_curr[0] - omega_win_curr[-1]
    L_value = L_values

    raw_plot = ax["raw"].plot(omega_win_curr, initial_raw, "*")

    tmp_omega = np.round(delta_omega_win_curr, 4)
    l_bin_rate = round(l_bin_rate, 4)

    ax["raw"].set_title(
        f"Raw int; $L$, $\\Delta \\omega$ = ({float(L_value[0]):.4f}, {float(tmp_omega):.4f})"
    )

    initial_sbt = sbt_[0]
    sbt_plot = ax["sbt"].plot(
        omega_win_curr,
        initial_sbt,
        "*",
        label=f"bin $\\omega$ ={bin_omega_rate}\nbin $\\gamma$ ={bin_gamma_rate}\nbin $\\delta$ ={bin_delta_rate}",
    )
    ax["sbt"].set_title(f"Subtracted int, $\\Delta L$ = {l_bin_rate}")

    initial_bckgd = bckgd_[0]
    bckgd_plot = ax["raw"].plot(
        omega_win_curr,
        initial_bckgd,
        "v",
        color="y",
        label="background",
    )

    ax_slider_image = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider_image = Slider(ax_slider_image, "binned gamma pxl", 0, len(raw_) - 1, valinit=0, valstep=1)

    ax["raw"].legend()
    ax["sbt"].legend()

    ax["sbt"].set_ylabel("img/$\\omega$ window")
    ax["sbt"].set_xlabel("$I$")

    ax["raw"].set_ylabel("img/$\\omega$ window")
    ax["raw"].set_xlabel("$I$")

    def update(val, l_bin_rate=l_bin_rate):
        if _save:
            index = int(val)
        else:
            index = int(slider_image.val)
        ax["raw"].clear()
        ax["sbt"].clear()

        omega_win_curr = omega_values[omega_windows[index]]
        delta_omega_win_curr = omega_win_curr[0] - omega_win_curr[-1]
        tmp_omega = np.round(delta_omega_win_curr, 4)

        raw_line = ax["raw"].plot(omega_win_curr, raw_[index], "*")[0]
        bckgd_line = ax["raw"].plot(
            omega_win_curr,
            bckgd_[index],
            "v",
            color="y",
            label="background",
        )[0]
        sbt_line = ax["sbt"].plot(
            omega_win_curr,
            sbt_[index],
            "*",
            label=f"bin $\\omega$ ={bin_omega_rate}\nbin $\\gamma$ ={bin_gamma_rate}\nbin $\\delta$ ={bin_delta_rate}",
        )[0]

        ax["raw"].set_title(
            f"Raw int, $L$, $\\Delta \\omega$ = ({float(L_value[index]):.4f}, {float(tmp_omega):.4f})"
        )
        ax["sbt"].set_title(f"Subtracted int, $\\Delta L$ = {l_bin_rate}")
        ax["raw"].legend()
        ax["sbt"].legend()

        ax["sbt"].set_xlabel("img/$\\omega$ window")
        ax["sbt"].set_ylabel("$Int$")

        ax["raw"].set_xlabel("img/$\\omega$ window")
        ax["raw"].set_ylabel("$Int$")
        fig.canvas.draw_idle()

    if save_animate:
        _save = True
        anim = FuncAnimation(fig, update, frames=len(raw_), interval=20, repeat=False)
        if movie_path is not None:
            images = []
            print("\n... saving gif ... \n")

            for i in range(len(raw_)):
                try:
                    update(i)
                except IndexError as exp:
                    print(f"{i = }, {gamma_windows[i] = }, {np.round(L_values[i], 5) = }")
                    raise exp
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(image)

            imageio.mimsave(movie_path, images, fps=1)
            print("... done ... \n")
        else:
            raise ValueError("path should not be None")
        _save = False

    slider_image.on_changed(update)
    plt.tight_layout()
    plt.show()
    return fig, ax


def image_slider(images, fig_size=(7, 7), vmax1=20, vmin1=None, cmap1="viridis"):
    """
    Display an interactive slider for browsing an image stack.

    Parameters
    ----------
    images : array of shape (n_images, height, width)
        Image stack to display.
    fig_size : tuple of float, default (7, 7)
        Figure size.
    vmax1 : float, default 20
        Upper color scale limit.
    vmin1 : float, optional
        Lower color scale limit. If None, Matplotlib chooses the lower limit.
    cmap1 : str, default 'viridis'
        Colormap passed to ``imshow``.

    Returns
    -------
    None
        The function displays the interactive figure but does not currently
        return ``fig`` or ``ax``.
    """
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    initial_image = images[0]
    if vmin1 is None:
        image_plot = ax.imshow(initial_image, cmap=cmap1, vmax=vmax1)
    else:
        image_plot = ax.imshow(initial_image, cmap=cmap1, vmax=vmax1, vmin=vmin1)

    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, "Image Index", 0, len(images) - 1, valinit=0, valstep=1)

    def update(val):
        index = int(slider.val)
        image_plot.set_array(images[index])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()


def image_mask_slider(images, masks, vmax_im1=250, vmax_im2=1,
                      fig_size=(12, 6), cmap1="viridis",
                      vmin_im1=0, vmin_im2=0):
    """
    Display an interactive slider comparing images and masks frame by frame.

    Parameters
    ----------
    images : array of shape (n_images, height, width)
        Image stack.
    masks : array of shape (n_images, height, width)
        Mask stack aligned with `images`.
    vmax_im1, vmin_im1 : float
        Color scale limits for the image panel.
    vmax_im2, vmin_im2 : float
        Color scale limits for the mask panel.
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
        ["img", "mask"]
    ], figsize=fig_size)

    initial_image = image_stack[0, :, :]
    img_plot = ax["img"].imshow(initial_image, vmin=vmin_im1, vmax=vmax_im1, cmap=cmap1)

    initial_mask = mask_stack[0, :, :]
    mask_plot = ax["mask"].imshow(initial_mask, vmin=vmin_im2, vmax=vmax_im2)

    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor="lightgoldenrodyellow")
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


def image_slider_Q(images, q_r, q_z, omes, vmin_=0, vmax_=250,
                   fig_size=(10, 6), show_axis=True, equal_aspect=True,
                   return_fig=False):
    """
    Display an interactive slider for a stack of reciprocal space images.

    Parameters
    ----------
    images : array of shape (n_images, n_qz, n_qr)
        Stack of images to display.
    q_r : ndarray of shape (n_qr,)
        Reciprocal space x-axis values.
    q_z : ndarray of shape (n_qz,)
        Reciprocal-space y-axis values.
    omes : array of shape (n_images,)
        Omega values or labels shown in the title for each image.
    vmin_ : float, default 0
        Lower color scale limit.
    vmax_ : float, default 250
        Upper color scale limit.
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

    aspect = "equal" if equal_aspect else "auto"
    img_plot = ax.imshow(
        images[0],
        extent=[q_r.min(), q_r.max(), q_z.min(), q_z.max()],
        origin="lower",
        aspect=aspect,
        cmap="viridis",
        vmin=vmin_,
        vmax=vmax_,
    )

    font1 = {"family": "sans-serif", "color": "black", "size": 14}
    ax.set_xlabel("$q_{xy}$ [$\\mathrm{\\AA}^{-1}$]", fontdict=font1)
    ax.set_ylabel("$q_{z}$ [$\\mathrm{\\AA}^{-1}$]", fontdict=font1)

    cbar = fig.colorbar(img_plot, ax=ax, orientation="vertical")
    cbar.ax.set_ylabel("Intensity", fontdict=font1)

    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(
        ax=ax_slider,
        label="Image Index",
        valmin=0,
        valmax=num_images - 1,
        valinit=0,
        valstep=1,
        color="blue",
    )

    if not show_axis:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def update(val):
        index = int(slider.val)
        img_plot.set_data(images[index])
        ax.set_title(f"omes -  {omes[index]}", fontdict=font1)
        fig.canvas.draw_idle()

    ax.set_title(f"omes -  {omes[0]}", fontdict=font1)
    slider.on_changed(update)

    plt.show()

    if return_fig:
        return fig, ax

    return None




def image_movie(images, output_file="image_movie.mp4",
                fps=10, dpi=100, fig_size=(7, 7),
                vmax1=20, vmin1=None, cmap1="viridis"):
    """
    Save an image stack as an MP4 movie.

    Parameters
    ----------
    images : array of shape (n_images, height, width)
        Image stack to save as a movie.
    output_file : str, default 'image_movie.mp4'
        Path of the output MP4 file.
    fps : int or float, default 10
        Number of frames per second in the output movie.
    dpi : int, default 100
        Resolution of the saved movie.
    fig_size : tuple of float, default (7, 7)
        Figure size.
    vmax1 : float, default 20
        Upper color scale limit.
    vmin1 : float, optional
        Lower color scale limit. If None, Matplotlib chooses the lower limit.
    cmap1 : str, default 'viridis'
        Colormap passed to ``imshow``.

    Returns
    -------
    None
        The function saves the image stack as an MP4 movie.

    Raises
    ------
    ValueError
        If `images` contains no frames.
    """
    if len(images) == 0:
        raise ValueError("images must contain at least one frame")

    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    initial_image = images[0]

    if vmin1 is None:
        image_plot = ax.imshow(
            initial_image,
            cmap=cmap1,
            vmax=vmax1,
        )
    else:
        image_plot = ax.imshow(
            initial_image,
            cmap=cmap1,
            vmax=vmax1,
            vmin=vmin1,
        )

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    with mpl.rc_context({"animation.ffmpeg_path": ffmpeg_path}):
        writer = FFMpegWriter(fps=fps)
    
        with writer.saving(fig, output_file, dpi=dpi):
            for image in images:
                image_plot.set_array(image)
                writer.grab_frame()

    plt.close(fig)



def image_mask_movie(images, masks, output_file="image_mask_movie.mp4",
                     fps=10, dpi=100, vmax_im1=250, vmax_im2=1,
                     fig_size=(12, 6), cmap1="viridis",
                     vmin_im1=0, vmin_im2=0):
    """
    Save aligned image and mask stacks as a side-by-side MP4 movie.

    Parameters
    ----------
    images : array of shape (n_images, height, width)
        Image stack.
    masks : array of shape (n_images, height, width)
        Mask stack aligned with `images`.
    output_file : str, default 'image_mask_movie.mp4'
        Path of the output MP4 file.
    fps : int or float, default 10
        Number of frames per second in the output movie.
    dpi : int, default 100
        Resolution of the saved movie.
    vmax_im1, vmin_im1 : float
        Color scale limits for the image panel.
    vmax_im2, vmin_im2 : float
        Color scale limits for the mask panel.
    fig_size : tuple of float, default (12, 6)
        Figure size.
    cmap1 : str, default 'viridis'
        Colormap used for the image panel.

    Returns
    -------
    None
        The function saves the image and mask stacks as an MP4 movie.

    Raises
    ------
    ValueError
        If `images` and `masks` do not contain the same number of frames,
        or if they contain no frames.
    """
    if len(images) != len(masks):
        raise ValueError("images and masks must have the same number of frames")

    if len(images) == 0:
        raise ValueError("images and masks must contain at least one frame")

    image_stack = images
    mask_stack = masks

    fig, ax = plt.subplot_mosaic([
        ["img", "mask"]
    ], figsize=fig_size)

    initial_image = image_stack[0, :, :]
    img_plot = ax["img"].imshow(
        initial_image,
        vmin=vmin_im1,
        vmax=vmax_im1,
        cmap=cmap1,
    )

    initial_mask = mask_stack[0, :, :]
    mask_plot = ax["mask"].imshow(
        initial_mask,
        vmin=vmin_im2,
        vmax=vmax_im2,
    )

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    with mpl.rc_context({"animation.ffmpeg_path": ffmpeg_path}):
        writer = FFMpegWriter(fps=fps)

        with writer.saving(fig, output_file, dpi=dpi):
            for image, mask in zip(image_stack, mask_stack):
                img_plot.set_array(image)
                mask_plot.set_array(mask)
                writer.grab_frame()

    plt.close(fig)
