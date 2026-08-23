# PolyXSim input generation from EBSD

`write_polyxsim_inp_from_ebsd` writes a PolyXSim input file from a selected EBSD
orientation region.

## Example

```python
write_polyxsim_inp_from_ebsd(
    "polyNb_from_ebsd_r_13_327.inp",
    wavelength_A=0.664008112030619,
    distance_mm=221.339,
    dety_center_px=1482.502,
    detz_center_px=2450.436,
    y_size_mm=0.075,
    z_size_mm=0.075,
    dety_size_px=3110,
    detz_size_px=2500,
    omega_start=60.0,
    omega_end=340.0,
    omega_step=0.25,
    theta_min=0.0,
    theta_max=25.0,
    o11=1,
    o12=0,
    o21=0,
    o22=-1,
    region_df=roi,
    use_Gcols=True,
    U_equals="gt",
    pos_mode="zero",
    direc="./",
    stem="grain_1_ebsd",
)
```

If all `G11..G33` columns are present and `use_Gcols=True`, those values define the
orientation matrix. Otherwise the Bunge Euler angle columns are used.

`U_equals="gt"` writes the transpose of the constructed orientation matrix. Use the
convention that matches the downstream PolyXSim geometry.

With `pos_mode="zero"`, all simulated orientations are placed at the origin. A
coordinate derived mode can instead be used when the selected EBSD points should
retain local position differences.
