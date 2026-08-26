# ImageD11 and GrainSpotter indexing

This workflow converts a detector rotation scan into GrainSpotter input,
indexes the resulting g-vectors, and overlays the indexed reflections on the
detector images.

The responsibilities are separated as follows:

- `ImageD11` detects and merges peaks and creates the `.gve` file;
- the external `GrainSpotter` executable performs multigrain indexing;
- `pyhectr.grainspotter` parses peak and log files and constructs tables;
- `pyhectr.read_plot` loads beamline images and displays interactive overlays.

The complete experimental workflow is available in
`examples/Gvector_GrainSpotter.ipynb`.

## Required software

Image loading and parsing existing files require only pyHECTR. Peak search and
indexing additionally require:

- [ImageD11](https://github.com/FABLE-3DXRD/ImageD11);
- a locally installed GrainSpotter executable;
- an interactive Matplotlib backend for notebook sliders.

ImageD11 should remain an optional dependency: importing
`pyhectr.grainspotter` to parse an existing log must not require ImageD11.

## File pipeline

| File | Producer | Purpose |
| --- | --- | --- |
| detector images | experiment | one detector frame per measured omega value |
| `peaks.spt` | ImageD11 peak search | two dimensional peaks before frame merging |
| `peaks.flt` | ImageD11 peak search | merged three dimensional spots used for overlays |
| `geom.prm` | notebook | detector geometry, wavelength, and unit cell |
| `peaks.gve` | ImageD11 transformer | g-vectors consumed by GrainSpotter |
| `GrainSpotter.ini` | notebook | indexing ranges, tolerances, and filenames |
| `grains.log` | GrainSpotter | grains, orientations, positions, and assigned peaks |

## Imports

Use `pyhectr.read_plot` as the public facade for image I/O and plotting:

```python
from pathlib import Path
import subprocess

import numpy as np
from ImageD11 import labelimage, transformer

from pyhectr import read_plot
from pyhectr.grainspotter import (
    parse_spt_table,
    group_peaks_by_frame,
    build_reflections_table,
    build_spot_coord_table,
    build_multigrain_table,
)
```

In Jupyter, activate an interactive backend before creating a slider:

```python
%matplotlib widget
```

## Load images and omega metadata

```python
data, metadata = read_plot.read_P07_imgs_with_metadata(
    img_path_pattern,
    roix=ROIx,
    roiy=ROIy,
)

metadata = np.asarray(metadata, dtype=float)
omegas = metadata if metadata.ndim == 1 else metadata[:, 0]

if data.ndim != 3:
    raise ValueError("data must have shape (n_frames, rows, columns)")
if omegas.shape != (len(data),):
    raise ValueError("one omega value is required for every detector frame")
if not np.all(np.isfinite(omegas)):
    raise ValueError("omega metadata contains non-finite values")
```

Use the measured metadata rather than generating omega values with `linspace`.
The image and metadata arrays must describe the same frame order.

`read_plot.compute_max_pixel_image` is a streaming alternative when only a
maximum projection is required. It does not return the complete image stack or
omega metadata.

## Run ImageD11 peak search

The peak search writes separate `.spt` and `.flt` files. Process frames in
omega order while keeping the original image to omega association:

```python
output_dir = Path("gvec")
output_dir.mkdir(parents=True, exist_ok=True)

frames = np.asarray(data)
order = np.argsort(omegas)
height, width = frames.shape[1:]
threshold = 9000.0

spt_path = output_dir / "peaks.spt"
flt_path = output_dir / "peaks.flt"

with spt_path.open("w") as spt, flt_path.open("w") as flt:
    lio = labelimage.labelimage(
        (height, width),
        fileout=flt,
        sptfile=spt,
    )
    spt.write(lio.titles)

    total_2d_peaks = 0
    for frame_index in order:
        lio.peaksearch(
            frames[frame_index],
            threshold,
            omegas[frame_index],
        )
        total_2d_peaks += int(lio.npk)
        if lio.npk:
            lio.output2dpeaks(spt)
            lio.mergelast()

    lio.finalise()
```

`lio.npk` is a per frame value. Accumulate it inside the loop when reporting the
total number of two dimensional peaks.

## Associate merged peaks with detector frames

Use the merged `.flt` table so its identifiers remain consistent with the
later `.gve` and GrainSpotter log:

```python
peaks = parse_spt_table(flt_path)

unique_omega = np.sort(np.unique(omegas))
if len(unique_omega) < 2:
    raise ValueError("at least two distinct omega values are required")

omega_step = float(np.median(np.abs(np.diff(unique_omega))))
peaks_by_frame = group_peaks_by_frame(
    peaks,
    omegas,
    tol_deg=0.51 * omega_step,
)
```

Start with a tolerance slightly larger than half the measured frame spacing.

Inspect the result before indexing:

```python
peak_state = read_plot.image_slider_with_spt(
    data,
    omegas,
    peaks_by_frame,
    vmax=950,
    size_by="npix",
    annotate=True,
    origin="upper",
)
```

See [Data and coordinate conventions](conventions.md) for the detector coordinate
and peak identifier conventions.

## Generate g-vectors

The ImageD11 `.prm` file contains the detector geometry, wavelength, unit cell,
and detector orientation. ImageD11 expects detector distance and pixel size in
micrometres in this file. The detector centre must refer to the same rotated and
cropped image convention used for peak search.

After writing `geom.prm`, generate g-vectors from the merged `.flt` table:

```python
transform = transformer.transformer()
transform.loadfiltered(str(flt_path))
transform.loadfileparameters(str(output_dir / "geom.prm"))
transform.compute_tth_eta()
transform.addcellpeaks()
transform.computegv()
transform.savegv(str(output_dir / "peaks.gve"))
```

## Write the GrainSpotter configuration

Calculate `domega` and the omega range from the measured metadata. The other
ranges and uncertainty values remain experiment specific:

```python
ini_text = f"""\
spacegroup 229
etarange 0 360
domega {omega_step:.8g}
tthrange 0 15
omegarange {omegas.min():.8g} {omegas.max():.8g}
filespecs peaks.gve grains.log
cuts 4 0.1 0.0
eulerstep 2
uncertainties 0.12 0.15 0.25
nsigmas 2
positionfit
"""

(output_dir / "GrainSpotter.ini").write_text(ini_text)
```


Run GrainSpotter with the output directory as the working directory so the
relative filenames in `filespecs` resolve correctly:

```python
subprocess.run(
    ["GrainSpotter", "GrainSpotter.ini"],
    cwd=output_dir,
    check=True,
)
```

The executable may be named `GrainSpotter.0.90` on some installations.


## Parse the results

Construct one table for assigned reflections, one for detector coordinates,
and one for grain orientations:

```python
log_path = output_dir / "grains.log"

reflections = build_reflections_table(log_path)
spots = build_spot_coord_table(peaks_by_frame)
orientations = build_multigrain_table(
    log_path,
    axes=("x", "y", "z"),
    max_index=6,
    order_indices=(0, 2, 1),
    a=3.3004,
)
```

The resulting indices are:

- `reflections`: `(grain_id, peak_id)`;
- `spots`: `peak_id`;
- `orientations`: `(grain_id, axis)`.

The reflection and spot tables are joined through `peak_id`.

## Visualize indexed reflections

```python
grain_state = read_plot.grain_slider_on_max_image(
    max_image=data.max(axis=0),
    refl_df=reflections,
    spot_df=spots,
    orient_df=orientations,
    vmax=200,
    origin="upper",
    annotate=True,
    preferred_axis="z",
)
```


For EBSD orientation indexing and coordinate calibration, see
[EBSD and crystallographic direction indexing](ebsd_polycrystal.md).

## Further reading

- [FABLE GrainSpotter 0.90 documentation](https://sourceforge.net/p/fable/wiki/grainspotter/)
- [ImageD11 documentation](https://imaged11.readthedocs.io/)
