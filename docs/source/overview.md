# What pyHECTR is about

## Scope

Python tools for high energy crystal truncation rod (CTR) reconstruction, detector space mask preparation, and rocking scan integration.


The package combines several analysis layers that are often handled separately:

- detector pixels and scattering angles;
- direct and reciprocal lattice geometry;
- HKL reconstruction;
- Bragg reflection and azimuthal offset analysis;
- CTR localization and detector space masks;
- rocking curve background subtraction and integration;
- Lorentz/rod, polarization, transmission, area, and flat field corrections;
- detector image and mask visualization;
- optional Mask R-CNN detector localization;
- optional EBSD/polycrystal orientation analysis and PolyXSim export;
- high index surface cell construction and XTL/BUL writing.

## Modules

| Module | Main role |
|---|---|
| `pyhectr.xrd_geom` | detector geometry, reciprocal lattice, HKL reconstruction, gridding, CTR masks, mask morphology |
| `pyhectr.integration` | signal window estimation, background models, detector corrections, rocking-scan integration |
| `pyhectr.read_plot` | detector/HDF5 I/O, prediction loading, maximum projections, plotting, sliders, movies, uncertainty helpers |
| `pyhectr.bragg_simulations` | theoretical Bragg maps, reflection filtering, symmetry assignments, theta-offset cost scans |
| `pyhectr.theta0_finder` | sparse point conversion of detector pixels to fractional HKL |
| `pyhectr.mask_nn` | intensity preprocessing, pseudo channel construction, Detectron2 setup and Mask R-CNN inference |
| `pyhectr.ebsd_map` | EBSD orientation maps, affine calibration, footprint selection |
| `pyhectr.polycrystal` | nearest crystallographic-direction indexing for EBSD/polycrystal orientations |
| `pyhectr.polyxsim` | PolyXSim `.inp` generation from EBSD orientations |
| `pyhectr.surface` | transformed surface supercells and XTL/BUL structure files |

`pyhectr.read_plot`  Data loading and plotting helpers.

## CTR localization

A detector stack has the working shape

```text
(n_images, n_gamma, n_delta)
```

The branch:

1. converts detector ROIs to `delta` and `gamma`;
2. builds the reciprocal basis and inverse UB matrix;
3. optionally refines a constant azimuthal offset from selected Bragg/CTR points;
4. calculates fractional `(h, k, l)` for every detector pixel and image;
5. grids the intensity in reciprocal space;
6. detects in-plane CTR positions;
7. projects those positions back to a detector space mask.

For large reconstructed datasets, `xrd_geom.grid_hkl_3d` provides chunked
xrayutilities `Gridder3D` calls while retaining one global HKL grid.

## ML CTR localization

The optional neural network branch keeps the analysis in detector space:

1. load the detector stack;
2. normalize/equalize intensity;
3. construct three channels from one or neighboring rocking frames;
4. run a trained Detectron2 Mask R-CNN;
5. save per-frame instance masks or bounding-box/mask-coordinate data;
6. aggregate the prediction into a detector-space mask.


## Integration backend

A mask from either localization branch is converted to

```text
(image_index, gamma_pixel, delta_pixel)
```

coordinates. The integration code estimates the local omega and delta extents, constructs moving detector windows,  subtracts the rocking curve background, applies
the selected detector correction convention.

