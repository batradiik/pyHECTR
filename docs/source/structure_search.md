# Automated ROD structure search


`pyHECTR` provides tools for preparing and analysing high energy CTR data and for generating surface structure files. 

Automated refinement of these structures with ROD is provided through the separate [ROD Optuna Search](https://github.com/batradiik/rod_structure_search) repository.

It uses [Optuna](https://optuna.org/) to explore different combinations of active ROD refinement parameters and evaluate the resulting fits.

Two example environments are provided:

- **Windows:** local searches using `rod.exe`;
- **Linux/SLURM:** parallel searches using `rod_doublePrecision` on a computing cluster.

The repository includes working Nb examples with `.bul`, `.fit`, `.par`, and `.dat` files. 

Installation requirements, execution commands, file layouts, and workflow details are documented in the [ROD Optuna Search README](https://github.com/batradiik/rod_structure_search#readme).
