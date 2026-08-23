# High-index surface cell generation

## Bulk bcc Nb

```python
a0_nb = 3.3004

bulk_lattice = np.eye(3) * a0_nb

bulk_fractional = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
])
```

Lattice vectors are stored as rows.

## Transformation convention

For transformation matrix `P`:

```text
A_surface = P.T @ A_bulk
```

The columns of `P` contain the coefficients of the transformed lattice vectors in
the original lattice basis.

Fractional coordinates are generated according to

```text
f_surface = (f_bulk + n) @ P^{-T} + origin_shift
```

and reduced to the half open transformed cell.

## Candidate Nb surface transformations

The current example constructs Nb(326), Nb(438), Nb(539), and Nb(6 4 11).

For Nb(539):

```python
transform_539 = np.array([
    [0, -6, 5],
    [3,  1, 3],
    [-1, 3, 9],
])
```

## Generate and write all files

```python
result = generate_surface_files(
    label="539",
    transform=transform_539,
    lattice_vectors=bulk_lattice,
    fractional_coordinates=bulk_fractional,
    species="Nb",
    output_dir="surface_cells",
    file_stem="Nb_539",
    title="Niobium (539)",
    comment="# Niobium (539)",
    save_xtl=True,
    save_bul=True,
    save_positive_bul=True,
)
```

The returned dictionary includes:

- transformed lattice vectors;
- transformed fractional coordinates;
- species;
- cell parameters;
- expected atom count;
- paths of written XTL/BUL files.

## Atom count check

For a periodic integer transform, the expected number of atoms is

```text
N_surface = N_basis * abs(det(P))
```

`generate_surface_cell` checks this by default.

## Inspect transformed axes

The example verifies the integer coefficient vectors directly:

```python
a_coeff = transform[:, 0]
b_coeff = transform[:, 1]
c_coeff = transform[:, 2]

print(np.dot(a_coeff, b_coeff))
print(np.dot(a_coeff, c_coeff))
print(np.dot(b_coeff, c_coeff))
print(c_coeff)
```

This catches accidental row/column convention changes.

## Other materials

One species string can be broadcast to every basis atom. Multi element crystals pass
one label per bulk fractional coordinate:

```python
surface_lattice, surface_frac, species = generate_surface_cell(
    lattice,
    basis,
    transform,
    species=["Cs", "Cl"],
)
```
