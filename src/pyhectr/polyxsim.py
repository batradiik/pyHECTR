import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = [
    "row_to_g_from_Gcols",
    "bunge_euler_to_g",
    "write_polyxsim_inp_from_ebsd",
]

def row_to_g_from_Gcols(row):
    """Build a 3x3 orientation matrix from G11..G33 dataframe row columns."""
    return np.array([
        [float(row["G11"]), float(row["G12"]), float(row["G13"])],
        [float(row["G21"]), float(row["G22"]), float(row["G23"])],
        [float(row["G31"]), float(row["G32"]), float(row["G33"])],
    ], dtype=float)


def bunge_euler_to_g(phi1_deg, PHI_deg, phi2_deg, degrees = True):
    """Convert Bunge ZXZ Euler angles to a 3x3 orientation matrix."""
    rot = R.from_euler("ZXZ", [phi1_deg, PHI_deg, phi2_deg], degrees=degrees)
    return rot.as_matrix()


def write_polyxsim_inp_from_ebsd(
    out_inp,
    *,
    # --- instrumental ---
    wavelength_A,
    distance_mm,
    dety_center_px, detz_center_px,
    y_size_mm, z_size_mm,
    dety_size_px, detz_size_px,
    omega_start, omega_end, omega_step, omega_sign=1,
    theta_min=0.0, theta_max=25.0,
    o11=1, o12=0, o21=0, o22=-1,
    tilt_x=0.0, tilt_y=0.0, tilt_z=0.0,
    beamflux=1e12,
    # --- structural ---
    unit_cell=(3.3004, 3.3004, 3.3004, 90.0, 90.0, 90.0),
    sgno=229,
    # --- output ---
    direc="polyNb_sim",
    stem="polyNb_from_ebsd",
    make_image=1,
    output_exts=(".tif", ".par", ".gve"),
    bg=0,
    noise=0,
    psf=1.2,
    peakshape=(1, 0.15, 0.3, 0.25),
    # --- grains ---
    region_df=None,
    grain_rows=None,
    use_Gcols=True,
    U_equals="gT",
    pos_mode="zero",
    ebsd_units_to_mm=1e-3,
    footprint_center_xy=None,
    default_size_mm=0.05,
    gen_size_mode=0,
    gen_size_spread_mm=0.01,
    gen_size_max_mm=0.10,
    default_eps6=(0, 0, 0, 0, 0, 0),
):
    """Write a PolyXSim .inp file using EBSD-derived grain orientations."""

    # ---- helper: if user *wants* to reduce to a single rep. point, they can pass grain_rows explicitly ----
    if grain_rows is None:
        if region_df is None or len(region_df) == 0:
            raise ValueError("Provide region_df with >=1 rows or grain_rows list.")
        # IMPORTANT: use *all* rows in region_df
        grain_rows = [row for _, row in region_df.iterrows()]

    n = len(grain_rows)

    lines = []
    lines += ["### Instrumental"]
    lines += [f"wavelength   {wavelength_A}"]
    lines += [f"beamflux     {beamflux}"]
    lines += [f"distance     {distance_mm}"]
    lines += [""]
    lines += [f"dety_center  {dety_center_px}"]
    lines += [f"detz_center  {detz_center_px}"]
    lines += [""]
    lines += [f"y_size       {y_size_mm}"]
    lines += [f"z_size       {z_size_mm}"]
    lines += [""]
    lines += [f"dety_size    {dety_size_px}"]
    lines += [f"detz_size    {detz_size_px}"]
    lines += [""]
    lines += [f"tilt_x       {tilt_x}"]
    lines += [f"tilt_y       {tilt_y}"]
    lines += [f"tilt_z       {tilt_z}"]
    lines += [""]
    lines += [f"omega_start  {omega_start}"]
    lines += [f"omega_end    {omega_end}"]
    lines += [f"omega_step   {omega_step}"]
    lines += [f"omega_sign   {omega_sign}"]
    lines += [""]
    lines += ["beampol_factor  1"]
    lines += ["beampol_direct  0"]
    lines += [""]
    lines += [f"theta_min    {theta_min}"]
    lines += [f"theta_max    {theta_max}"]
    lines += [""]
    lines += [f"o11  {o11}"]
    lines += [f"o12  {o12}"]
    lines += [f"o21  {o21}"]
    lines += [f"o22  {o22}"]
    lines += [""]

    # ---- Grains ----
    lines += ["### Grains"]
    lines += [f"no_grains  {n}"]
    lines += ["gen_U      0"]
    lines += ["gen_pos    0 0"]
    lines += ["gen_eps    0 0 0 0 0"]
    lines += ["sample_cyl 0.5 0.5"]
    lines += [f"gen_size   {gen_size_mode} {default_size_mm} {gen_size_spread_mm} {gen_size_max_mm}"]
    lines += [""]

    if pos_mode == "from_xy" and footprint_center_xy is None:
        raise ValueError("If pos_mode='from_xy', provide footprint_center_xy=(x0,y0).")

    # 0-based indices, like your original inp
    for ig, row in enumerate(grain_rows):
        if use_Gcols and all(k in row.index for k in ("G11", "G12", "G13", "G21", "G22", "G23", "G31", "G32", "G33")):
            g = row_to_g_from_Gcols(row)
        else:
            g = bunge_euler_to_g(float(row["phi1"]), float(row["PHI"]), float(row["phi2"]), degrees=True)

        U = g.T if U_equals.lower() == "gt" else g
        Uflat = " ".join(f"{v:.9f}" for v in U.reshape(-1))

        if pos_mode == "zero":
            pos = (0.0, 0.0, 0.0)
        else:
            cx, cy = footprint_center_xy
            px = (float(row["X"]) - float(cx)) * ebsd_units_to_mm
            py = (float(row["Y"]) - float(cy)) * ebsd_units_to_mm
            pos = (px, py, 0.0)

        lines += [f"U_grains_{ig}   {Uflat}"]
        lines += [f"pos_grains_{ig} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"]
        lines += [f"eps_grains_{ig} " + " ".join(f"{float(v):.6g}" for v in default_eps6)]
        lines += [f"size_grains_{ig} {float(default_size_mm):.6f}"]
        lines += [""]

    lines += ["### Structural"]
    a, b, c, al, be, ga = unit_cell
    lines += [f"unit_cell   {a} {b} {c}  {al} {be} {ga}"]
    lines += [f"sgno        {sgno}"]
    lines += [""]

    lines += ["### Files"]
    lines += [f"direc  '{direc}'"]
    lines += [f"stem   '{stem}'"]
    lines += [""]

    lines += ["### Images"]
    lines += [f"make_image  {make_image}"]
    lines += ["output " + " ".join(f"'{e}'" for e in output_exts)]
    lines += [f"bg     {bg}"]
    lines += [f"noise  {noise}"]
    lines += [f"psf    {psf}"]
    lines += ["peakshape " + " ".join(f"{v:g}" for v in peakshape)]
    lines += [""]

    with open(out_inp, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote PolyXSim input to {out_inp!r}")
