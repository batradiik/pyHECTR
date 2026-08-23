# Quick start


## Imports

```python
import numpy as np

from pyhectr import integration
from pyhectr import read_plot
from pyhectr import xrd_geom
```

## Detector ROI to scattering angles

```python
ROIx = slice(1000, 1200)
ROIy = slice(1500, 1800)

x0 = 1035.0
y0 = 1950.0
pix_size = 0.2 # mm
SDD = 1400.0 # mm

delta = xrd_geom.ROI_to_angle(ROIx, x0, pix_size, SDD)
gamma = xrd_geom.ROI_to_angle(ROIy, y0, pix_size, SDD)
```

`delta` and `gamma` are in degrees. `pix_size` and `SDD` must be in mm.

## Detector correction map

```python
correction_map = integration.apply_corrections2D(
    delta_arr=delta,
    gamma_arr=gamma,
    incidence_ang=0.03,
    rocking=True,
    return_map=True,
)

print(correction_map.shape)
# (len(gamma), len(delta))
```

For a detector stack shaped `(n_images, n_gamma, n_delta)` the map broadcasts over
the first dimension:

```python
corrected_data = data / correction_map[None, :, :]
```

This division follows the current DESY P07 beamline  integration example.

## Detector space rod mask

```python
mask = np.zeros((20, len(gamma), len(delta)), dtype=bool)
mask[5:15, 100:140, 60:70] = True

pixel_coord, max_mask = integration.rod_points_prep(
    mask,
    show=False,
)

print(pixel_coord.shape)
```

Rows of `pixel_coord` are

```text
(image_index, gamma_pixel, delta_pixel)
```

## Estimate a moving integration window

```python
half_omega = integration.get_omega_range(
    pixel_coord,
    gamma_pxl_range=mask.shape[1],
    plot_flag=False,
)

half_delta, delta_range = integration.get_delta_range(
    pixel_coord,
    gamma_pxl_range=mask.shape[1],
    plot_flag=False,
)
```

## Maximum projection and visualization

```python
max_image = read_plot.max_pxl_im(data)

read_plot.image_mask_slider(
    data,
    mask,
    vmax_im1=250,
    vmax_im2=1,
)
```

## Next

For the complete single crystal pipeline, continue with:

- [P07 sapphire data preparation](examples/p07_preparation.md)
- [P07 sapphire rocking-scan integration](examples/p07_integration.md)
