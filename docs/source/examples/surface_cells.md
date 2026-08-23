# Nb high index surface cell example

This page follows `Nb_surface_cell_generation.ipynb`.

## 1. Define bulk Nb

```python
a0_nb = 3.3004
bulk_lattice = np.eye(3) * a0_nb

bulk_fractional = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
])
```

## 2. Define candidate transforms

The notebook contains integer transforms for:

- Nb(326);
- Nb(438);
- Nb(539);
- Nb(6 4 11).

Before generation it prints:

```text
abs(det(P))
expected atom count = 2 * abs(det(P))
```

for each transformed bcc cell.

## 3. Generate all structures

```python
results = {}

for label, transform in SURFACE_TRANSFORMS.items():
    results[label] = generate_surface_files(
        label=label,
        transform=transform,
        lattice_vectors=bulk_lattice,
        fractional_coordinates=bulk_fractional,
        species="Nb",
        output_dir="surface_cells",
        file_stem=f"Nb_{label}",
        title=f"Niobium ({label})",
        comment=f"# Niobium ({label})",
        save_xtl=True,
        save_bul=True,
        save_positive_bul=True,
    )
```

## 4. Check geometry

For each candidate the notebook prints cell lengths, cell angles, atom count, and the dot products between the integer coefficient vectors.

The transformed surface normal coefficient vector is the third transformation column.

## 5. Inspect Nb(539)

The final cells print:

- transformed lattice vectors;
- six cell parameters;
- total atom count;
- first ten fractional coordinates.

This is a useful numerical check before opening the generated XTL/BUL files.
