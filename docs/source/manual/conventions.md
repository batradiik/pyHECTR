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


## ImageD11 and GrainSpotter detector coordinates

ImageD11 peak search must run on the same rotated and cropped image stack that is
later displayed. For the flat `.spt` and `.flt` peak tables, pyHECTR selects
detector coordinates in this order:

```text
(fc, sc) -> (f_raw, s_raw) -> (detz, dety) -> (f, s)
```

The first value is plotted as x / image column and the second as y / image row.
Use `origin="upper"` when row zero is displayed at the top of the image.

The omega value stored for every peak is matched to the nearest measured frame.
The matching tolerance should normally be slightly larger than half the measured
omega step.

## Peak identity through the indexing workflow

The merged ImageD11 `.flt` table is used for detector overlays because its
`spot3d_id` is propagated through the `.gve` file and appears as GrainSpotter
`peak_id` in `grains.log`.

The plotting tables are joined through this identifier:

```text
ImageD11 .flt spot3d_id
        -> .gve peak identifier
        -> GrainSpotter grains.log peak_id
```

If an indexed grain overlay is empty, check the overlap of these identifiers
before changing detector coordinates.

## GrainSpotter orientation matrices

The GrainSpotter parser treats `U` as the orientation matrix and `UBI` as

```text
UBI = inverse(U B)
```

For a specimen unit vector `e`, the current orientation summary uses:

```text
crystal direction       = U.T @ e
fractional plane normal = UBI @ e
```
