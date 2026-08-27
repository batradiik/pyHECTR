# pyHECTR documentation

`pyHECTR` provides tools for reciprocal space reconstruction, crystal
truncation rod localization, detector space mask preparation, and rocking scan
integration of high energy surface X-ray diffraction (HESXRD) data.


The structure:

- **Getting started** explains what the package does and how to install it.
- **User manual** explains the scientific and data processing steps.
- **Structure search** explains the Optuna search for the ROD surface structure fitting.
- **Examples**  the supplied research notebooks.
- **API reference** is generated from source docstrings.


```{admonition} Project status
:class: note

pyHECTR is research software under active development. Please report any bugs you encounter.
```



```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

overview
installation
quickstart
```


```{toctree}
:maxdepth: 2
:caption: User manual
:hidden:

manual/index
```

```{toctree}
:maxdepth: 2
:caption: Examples
:hidden:

examples/index
```

```{toctree}
:maxdepth: 1
:caption: ROD structure search
:hidden:

structure_search
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api/index
```


## First steps

1. Read [Installation](installation.md).
2. Read [Data and coordinate conventions](manual/conventions.md).
3. Run through [Quick start](quickstart.md).
4. Choose the [workflow example](examples/index.md) closest to your experiment.
5. For automated ROD refinement, see [Automated ROD structure search](structure_search.md).

