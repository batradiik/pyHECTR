# Data and coordinate conventions

## Detector stack axes

The main pyHECTR detector representation is

```text
data.shape = (n_images, n_gamma, n_delta)
```

with:

- axis 0 — image / rocking / azimuthal index;
- axis 1 — detector gamma pixel;
- axis 2 — detector delta pixel.

A selected signal pixel is therefore stored as

```text
(image_index, gamma_pixel, delta_pixel)
```

This order is used by the integration and theta offset workflows.

## Detector angles

`ROI_to_angle` and `pixel_to_angle` use the sign convention

```text
angle = arctan(-(pixel - beam_center) * pixel_size / SDD)
```

and return degrees.

## Reconstruction convention

The classical branch uses the UB relation

```text
Q_lab = R U B h
```

where `h = (h, k, l)^T`.

The P07 example uses:

- incident beam along the laboratory y direction;
- delta as the horizontal detector scattering rotation;
- gamma as the vertical detector scattering rotation;
- sample azimuth/omega as a rotation around laboratory z.


## Validate orientation before reconstruction

For detector readers that apply a flip/rotation convention, verify the orientation
with:

1. one known detector frame;
2. the maximum intensity projection;
3. direct beam coordinates;
4. the ordering of omega values.

The image stack and omega array must describe the same frame order.
