# Changelog

All notable changes to `pyHECTR` will be documented in this file.



## [0.2.2]

### Added

- Added GrainSpotter utilities and a complete indexing example workflow.
- Added surface coordinate generation and related surface structure examples.
- Added PolyXSim input and crystallographic structure example files.
- Added manuals, API references, and workflow documentation.
- Added documentation for the  Optuna automated ROD search.

### Changed

- Updated EBSD, PolyXSim, Mask R-CNN, surface, and P07 example notebooks.
- Improved surface, plotting, integration, I/O, and crystallographic utilities.

### Fixed

- Fixed use of `inverse_transpose` before its definition.
- Fixed imports and performed general code cleanup.



## [0.2.1]

### Added

- Added dedicated plotting utilities and expanded plotting functionality.
- Added additional I/O helpers for detector and experimental data.

### Changed

- Improved detector data handling and related analysis utilities.
- Updated example notebooks .
- 
### Fixed

- Fixed int32 dtype handling in calculations involving xrayutilities.
- Fixed plotting issues and improved compatibility of plotting functions.



## [0.2.0]

### Added

- Added optional neural network mask inference helpers in `pyhectr.mask_nn`.
- Added EBSD orientation map and coordinate calibration utilities in `pyhectr.ebsd_map`.
- Added polycrystalline orientation indexing utilities in `pyhectr.polycrystal`.
- Added PolyXSim input file generation helpers in `pyhectr.polyxsim`.
- Added surface cell generation and `.xtl` / `.bul` writing utilities in `pyhectr.surface`.
- Added example notebooks and workflows under `examples/`.
- Added installation notes for using a fresh virtual environment and registering a Jupyter kernel.
- Added Detectron2 installation notes for optional Mask R-CNN workflows.

### Changed

- Updated package metadata for version `0.2.0`.
- Updated README module overview to include the new modules.
- Expanded dependency metadata and optional helper dependencies.
- Improved documentation around optional PyTorch / Detectron2 setup.


## [0.1.0] - Initial release

### Added

- Initial `pyHECTR` package structure.
- Detector geometry utilities.
- CTR mask generation utilities.
- Rocking scan integration utilities.
- Plotting and image-loading helpers.
- Basic Bragg simulation and detector--HKL conversion helpers.
