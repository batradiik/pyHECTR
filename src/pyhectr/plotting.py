import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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
    "image_slider_with_spt",
    "grain_slider_on_max_image"
    
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




def image_slider_with_spt(images, omegas, peaks_by_frame,
                          fig_size=(7,7), vmax=20, cmap='viridis',
                          size_by='npix', annotate=False,
                          origin = 'lower'):
    """
    Improved interactive image slider with stable callbacks and Next/Prev buttons.
    Returns a state dict (keeps references alive).
    """
    N, H, W = images.shape
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    # show first image
    im = ax.imshow(images[0], cmap=cmap, vmax=vmax, origin=origin)
    scat = ax.scatter([], [], s=30, facecolors='none', edgecolors='r', linewidths=0.8)
    txt = ax.text(0.01, 0.99, '', transform=ax.transAxes, ha='left', va='top',
                  fontsize=9, bbox=dict(fc='white', alpha=0.6, ec='none'))
    annots = []

    # Slider axes
    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, 'Image Index', 0, N-1, valinit=0, valstep=1, valfmt='%0.0f')

    # Helper to pick coordinates (prefer corrected fc/sc; else raw f/s; else detz/dety)
    def get_xy(p):
        if 'fc' in p and 'sc' in p:
            return p['fc'], p['sc']
        if 'f_raw' in p and 's_raw' in p:
            return p['f_raw'], p['s_raw']
        if 'detz' in p and 'dety' in p:
            return p['detz'], p['dety']
        if 'f' in p and 's' in p:
            return p['f'], p['s']
        return None, None

    def update(val):
        i = int(np.round(val))
        # update image reliably
        im.set_data(images[i])

        # remove previous annotations
        if annots:
            for a in annots:
                try:
                    a.remove()
                except Exception:
                    pass
            annots.clear()

        peaks = peaks_by_frame.get(i, [])
        if peaks:
            xs, ys, sizes, valid_peaks = [], [], [], []
            for p in peaks:
                x, y = get_xy(p)
                if x is None:
                    continue
                xs.append(x)
                ys.append(y)
                valid_peaks.append(p)
                if size_by == 'npix':
                    npix = p.get('number_of_pixels', p.get('npix', None))
                    if npix is None:
                        npix = 10.0
                    sizes.append(10.0 + 2.0*np.sqrt(float(npix)))
                elif size_by == 'avg':
                    avg = p.get('avg_intensity', p.get('average_counts', 0.0))
                    sizes.append(10.0 + 0.02*float(avg))
                else:
                    sizes.append(30.0)
            if len(xs) > 0:
                scat.set_offsets(np.c_[xs, ys])
                scat.set_sizes(sizes)
            else:
                scat.set_offsets(np.empty((0, 2)))
                scat.set_sizes([])
            txt.set_text(f'frame {i}   omega≈{omegas[i]:.3f}°   peaks: {len(xs)}')
            if annotate:
                for x, y, p in zip(xs, ys, valid_peaks):
                    sid = p.get('spot3d_id', None)
                    if sid is not None:
                        annots.append(ax.text(x+2, y+2, str(int(sid)), fontsize=6, color='r'))
        else:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_sizes([])
            txt.set_text(f'frame {i}   omega≈{omegas[i]:.3f}°   peaks: 0')

        #ax.set_xlim(-0.5, W-0.5)
        #ax.set_ylim(H-0.5, -0.5)  # invert Y to match imshow
        fig.canvas.draw_idle()

    # Buttons: Prev / Next
    ax_prev = plt.axes([0.78, 0.01, 0.06, 0.03])
    ax_next = plt.axes([0.85, 0.01, 0.06, 0.03])
    bprev = Button(ax_prev, '<')
    bnext = Button(ax_next, '>')

    def _next(event=None):
        v = int(np.round(slider.val))
        slider.set_val(min(N-1, v + 1))

    def _prev(event=None):
        v = int(np.round(slider.val))
        slider.set_val(max(0, v - 1))

    bnext.on_clicked(lambda ev: _next(ev))
    bprev.on_clicked(lambda ev: _prev(ev))

    # Keyboard navigation
    def on_key(event):
        if event.key in ('right', 'd', 'pagedown'):
            _next()
        elif event.key in ('left', 'a', 'pageup'):
            _prev()

    cid = fig.canvas.mpl_connect('key_press_event', on_key)

    # Keep references alive (prevent garbage collection)
    state = dict(fig=fig, ax=ax, im=im, scat=scat, txt=txt, annots=annots,
                 slider=slider, buttons=(bprev, bnext), key_connid=cid,
                 images=images, omegas=omegas, peaks_by_frame=peaks_by_frame)
    # attach to figure object
    fig._image_slider_state = state

    # connect slider after state saved (just in case)
    slider.on_changed(update)

    # initial draw
    update(0)
    plt.show()
    return state




def _get_grain_orientation(orient_df, grain_id, preferred_axis='z'):
    """
    Return a dict with orientation info for a given grain_id.

    orient_df is indexed by (grain_id, axis) with columns including
    'phi1', 'PHI', 'phi2', and e.g. 'ub_hkl', 'euler_axisZ_uvw'.
    """
    try:
        # try preferred axis first
        row = orient_df.loc[(grain_id, preferred_axis)]
    except KeyError:
        # fallback: take first axis for this grain
        try:
            row = orient_df.xs(grain_id, level='grain_id').iloc[0]
        except Exception:
            return None

    phi1 = float(row.get('phi1', np.nan))
    PHI  = float(row.get('PHI', np.nan))
    phi2 = float(row.get('phi2', np.nan))

    # optional: some compact Miller notation for surface normal / main direction
    ub_hkl   = row.get('ub_hkl', None)            # e.g. (1,0,0)
    axisZ_uvw = row.get('euler_axisZ_uvw', None)  # e.g. (1,0,0)

    def _fmt_triplet(val):
        if isinstance(val, (tuple, list, np.ndarray)):
            return tuple(int(x) for x in val)
        # some of your values are like "np.int64(0)" in strings; handle that
        try:
            # seq = eval(str(val))
            
            from ast import literal_eval
            seq = literal_eval(str(val))
            if isinstance(seq, (tuple, list)):
                return tuple(int(x) for x in seq)
        except Exception:
            pass
        return None

    ub_hkl_fmt   = _fmt_triplet(ub_hkl)
    axisZ_uvw_fmt = _fmt_triplet(axisZ_uvw)

    return dict(
        phi1=phi1, PHI=PHI, phi2=phi2,
        ub_hkl=ub_hkl_fmt,
        axisZ_uvw=axisZ_uvw_fmt
    )


def grain_slider_on_max_image(max_image,
                              refl_df,
                              spot_df,
                              orient_df=None,
                              vmax=20,
                              cmap='viridis',
                              size_by='npix',
                              origin='lower',
                              annotate=True,
                              max_labels=80,
                              preferred_axis='z'):
    """
    Interactive viewer: slide over grain_id and show its reflections
    on the max-pixel image, with orientation info in the title.

    max_image  : 2D array (H,W)
    refl_df    : DataFrame indexed by (grain_id, peak_id)
    spot_df    : DataFrame indexed by peak_id with columns x, y, npix, avg_intensity
    orient_df  : DataFrame indexed by (grain_id, axis) with orientation info
    """

    # --- list of available grains (sorted) ---
    grain_ids = np.array(sorted(refl_df.index.get_level_values('grain_id').unique()))
    n_grains = len(grain_ids)
    if n_grains == 0:
        raise ValueError("No grains in refl_df")

    # --- base figure / max image ---
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(max_image, cmap=cmap, vmax=vmax, origin=origin)

    # empty scatter; will be filled in `update`
    scat = ax.scatter([], [], s=30,
                      facecolors='none', edgecolors='r', linewidths=0.6)

    # per-point annotations
    annots = []

    # small text box at top with grain/orientation info
    info_txt = ax.text(0.01, 0.99, '',
                       transform=ax.transAxes,
                       ha='left', va='top',
                       fontsize=8,
                       bbox=dict(fc='white', alpha=0.7, ec='none'))

    ax.set_title("Max-pixel image with GrainSpotter peaks")

    # --- helper to get merged table for one grain ---
    def merged_for_grain(gid):
        refl = refl_df.reset_index()
        refl = refl[refl['grain_id'] == gid].copy()
        m = refl.merge(spot_df.reset_index(), on='peak_id', how='inner')
        return m

    # --- slider over grain INDEX (0..n_grains-1) ---
    ax_slider = plt.axes([0.15, 0.01, 0.55, 0.03])
    slider = Slider(ax_slider, 'grain idx', 0, n_grains-1,
                    valinit=0, valstep=1, valfmt='%0.0f')

    # --- buttons for prev/next grain ---
    ax_prev = plt.axes([0.73, 0.01, 0.06, 0.03])
    ax_next = plt.axes([0.81, 0.01, 0.06, 0.03])
    bprev = Button(ax_prev, '<')
    bnext = Button(ax_next, '>')

    def _set_grain_idx(new_idx):
        new_idx = int(np.clip(new_idx, 0, n_grains-1))
        slider.set_val(new_idx)

    def _next(event=None):
        _set_grain_idx(int(slider.val) + 1)

    def _prev(event=None):
        _set_grain_idx(int(slider.val) - 1)

    bnext.on_clicked(lambda ev: _next(ev))
    bprev.on_clicked(lambda ev: _prev(ev))

    # keyboard shortcuts
    def on_key(event):
        if event.key in ('right', 'd', 'pagedown'):
            _next()
        elif event.key in ('left', 'a', 'pageup'):
            _prev()

    cid = fig.canvas.mpl_connect('key_press_event', on_key)

    # --- update function ---
    def update(val):
        idx = int(np.round(val))
        gid = int(grain_ids[idx])

        m = merged_for_grain(gid)

        xs = m['x'].to_numpy()
        ys = m['y'].to_numpy()

        # sizes
        if size_by == 'npix':
            base = m['npix'].fillna(0).to_numpy()
            sizes = 10.0 + 2.0 * np.sqrt(base)
        elif size_by == 'avg':
            base = m['avg_intensity'].fillna(0).to_numpy()
            sizes = 10.0 + 0.02 * base
        else:
            sizes = np.full_like(xs, 30.0, dtype=float)

        scat.set_offsets(np.c_[xs, ys])
        scat.set_sizes(sizes)

        # remove old annotations
        if annots:
            for a in annots:
                try:
                    a.remove()
                except Exception:
                    pass
            annots.clear()

        # (hkl) labels
        if annotate:
            labels_drawn = 0
            for _, row in m.iterrows():
                if labels_drawn >= max_labels:
                    break
                h, k, l = row.get('h', np.nan), row.get('k', np.nan), row.get('l', np.nan)
                try:
                    lab = f"({int(h)}{int(k)}{int(l)})"
                except Exception:
                    continue
                annots.append(
                    ax.text(row['x']+2, row['y']+2, lab,
                            fontsize=9, color='r')
                )
                labels_drawn += 1

        # orientation info
        orient_str = ""
        if orient_df is not None:
            ori = _get_grain_orientation(orient_df, gid, preferred_axis=preferred_axis)
            if ori is not None:
                phi1, PHI, phi2 = ori['phi1'], ori['PHI'], ori['phi2']
                orient_str = f"φ1={phi1:6.2f}°, Φ={PHI:6.2f}°, φ2={phi2:6.2f}°"
                if ori['ub_hkl'] is not None:
                    hkl = ori['ub_hkl']
                    orient_str += f"  n ≈ ({hkl[0]} {hkl[1]} {hkl[2]})"
                elif ori['axisZ_uvw'] is not None:
                    uvw = ori['axisZ_uvw']
                    orient_str += f"  [uvw]_Z ≈ [{uvw[0]} {uvw[1]} {uvw[2]}]"

        info_txt.set_text(f"grain {gid}  (index {idx+1}/{n_grains})\n{orient_str}")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    # initial draw
    update(0)

    # keep state to avoid garbage collection
    state = dict(
        fig=fig, ax=ax, im=im, scat=scat,
        annots=annots, info_txt=info_txt,
        slider=slider, bprev=bprev, bnext=bnext,
        key_connid=cid,
        grain_ids=grain_ids,
        refl_df=refl_df,
        spot_df=spot_df,
        orient_df=orient_df
    )
    fig._grain_slider_state = state

    plt.show()
    return state

