<p align="center">
  <img src="docs/source/_static/pyhectr_logo.svg" alt="pyHECTR logo" width="260">
</p>

# pyHECTR

Python tools for high energy crystal truncation rod (CTR) reconstruction, detector space mask preparation, and rocking scan integration.


`pyHECTR` supports the classical data reduction branch used for high energy grazing incidence X-ray diffraction (HEGIXRD) measurements with  two dimensional detectors. The package provides utilities for detector geometry, reciprocal space reconstruction, CTR mask generation, and visualization.

The neural network (CNN) mask inference based on Detectron2 is optional and requires a separate PyTorch/Detectron2 installation that matches your CPU/GPU environment.

---

## Recommended environment setup

Use a fresh virtual environment for `pyHECTR`. This avoids conflicts between scientific Python packages, Jupyter widgets, PyTorch, CUDA, and Detectron2.

```bash
python3 -m venv pyhectr
source pyhectr/bin/activate          # Linux/macOS
# pyhectr\Scripts\activate           # Windows PowerShell

python -m pip install --upgrade pip setuptools wheel
```

For the full tools usage, install the  extras.

```bash
python -m pip install "pyhectr[full]"
```

Register the environment as a Jupyter kernel:
```bash
python -m ipykernel install --user --name pyhectr --display-name "pyHECTR"
```

Then open JupyterLab and select the kernel named **pyHECTR**.

---

## Installation

Python 3.9 or newer is recommended.

### Install from PyPI

PyPI:

```bash
python -m pip install pyhectr
```

For optional helper dependencies:

```bash
python -m pip install "pyhectr[full]"
```

### Install directly from GitHub

Install the latest version from the default branch:

```bash
python -m pip install "pyhectr @ git+https://github.com/batradiik/pyHECTR.git"
```

With notebook extras:

```bash
python -m pip install "pyhectr[full] @ git+https://github.com/batradiik/pyHECTR.git"
```



### Clone for development

```bash
git clone https://github.com/batradiik/pyHECTR.git
cd pyHECTR

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -e .
```


```bash
git clone https://github.com/batradiik/pyHECTR.git
cd pyHECTR

python3 -m venv pyhectr
source pyhectr/bin/activate          # Linux/macOS
# pyhectr\Scripts\activate           # Windows PowerShell

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[full,dev]"
python -m ipykernel install --user --name pyhectr-dev --display-name "pyHECTR dev"
```

---


## Documentation 

The documentation for the related package can be find:
https://pyhectr.readthedocs.io



---

## Optional Detectron2 installation

The classical CTR reconstruction and rocking scan integration tools do **not** require Detectron2.

Detectron2 is only needed for the optional Mask R-CNN / neural-network helpers in `pyhectr.mask_nn`. Detectron2 should be installed manually because it must match your installed PyTorch and CUDA setup.

### 1. Install `pyHECTR` with neural-network helper dependencies

```bash
python -m pip install "pyhectr[full]"
```

For a development checkout:

```bash
python -m pip install -e ".[full,dev]"
```

### 2. Install PyTorch and torchvision

Choose the correct command for your machine from the official PyTorch installation selector:

<https://pytorch.org/get-started/locally/>

CPU-only example:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

GPU/CUDA example:

```bash
# Replace the index URL with the CUDA build recommended by the PyTorch selector.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Detectron2

Install Detectron2 after PyTorch is installed:

```bash
python -m pip install "git+https://github.com/facebookresearch/detectron2.git"
```

See the official Detectron2 installation guide for platform-specific notes:

<https://detectron2.readthedocs.io/en/stable/tutorials/install.html>

### 4. Verify the installation

```bash
python - <<'PY'
import torch
import detectron2
from pyhectr import mask_nn

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("detectron2:", detectron2.__version__)
print("pyhectr.mask_nn imported successfully")
PY
```

If Detectron2 import fails after changing PyTorch, reinstall Detectron2 in the same environment.

---

## Package modules

- `pyhectr.xrd_geom`  
  Detector angle conversion, reciprocal space grids, UB matrices, HKL maps, CTR localization, and mask morphology.

- `pyhectr.integration`  
  Background estimation, signal window estimation, correction maps, binary mask preparation, and rocking scan integration.

- `pyhectr.read_plot`  
  DESY P07 image and metadata loading, plotting utilities, interactive sliders, output folder creation, and uncertainty estimates.

- `pyhectr.bragg_simulations`  
  Theoretical reflection handling, peak plotting, symmetry handling, and azimuthal offset scans.

- `pyhectr.theta0_finder`  
  Conversion of detector pixels to fractional HKL coordinates.

- `pyhectr.mask_nn`  
  Optional Detectron2 / Mask R-CNN helper functions for detector space mask inference and prediction post-processing.

- `pyhectr.ebsd_map`  
  EBSD orientation map utilities, footprint selection, and EBSD coordinate calibration.

- `pyhectr.polycrystal`  
  Utilities for assigning low index crystallographic directions to EBSD grain orientations.

- `pyhectr.polyxsim`  
  Helpers for writing PolyXSim input files from EBSD grain orientations.

- `pyhectr.surface`  
  Surface cell generation and writing of `.xtl` and `.bul` structure files.

---


## Dependencies

### Core dependencies

- NumPy
- SciPy
- Matplotlib
- Pillow
- ImageIO
- OpenCV
- pandas
- tqdm
- xrayutilities
- ipython
- jupyterlab
- ipympl

### Extras

- joblib
- scikit-image
- h5py
- hdf5plugin
- imageio-ffmpeg
- yacs
- pycocotools
- fvcore
- iopath
- numba

---


## Citation

If you use `pyHECTR` in research, cite the software using the metadata in `CITATION.cff`.

A paper citation wikk be added as the preferred citation after the associated manuscript has been published and assigned a DOI. For archived software releases, a version specific DOI can also be created through Zenodo.

---


## Contributing

Bug reports and focused pull requests are welcome. Before contributing:

1. open an issue describing the proposed change;
2. add or update tests for behavior changes;
3. update docstrings and documentation where needed;
4. run the test and lint checks locally.


---


## License

This project is distributed under the GNU General Public License v2.0 or later (`GPL-2.0-or-later`). See `LICENSE`.

Parts of the Bragg-simulation functionality are adapted from or closely related to `xrayutilities`, which is distributed under `GPL-2.0-or-later`. Preserve the applicable copyright and attribution notices for adapted code.

