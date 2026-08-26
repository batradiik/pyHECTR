# High index surface cell generation

The surface workflow has two distinct stages:

1. use a VESTA `.xtl` cut to search for candidate in-plane repeat vectors;
2. validate the resulting integer transformation and generate the final
   high precision periodic cell directly from the bulk structure.

The VESTA export is an exploratory aid, not the final source of ROD
coordinates. Rounded exported fractional coordinates can introduce numerical
artifacts in large high index cells.

The companion examples are:

- `examples/Nb_single_cystal_surface_coordinates.ipynb` for transformation
  search;
- `examples/Nb_surface_cell_generation.ipynb` for exact cell generation.


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
f_surface = (f_bulk + n - origin_shift) @ P^{-T}
```

where `n` is an integer bulk translation and `origin_shift` is expressed in the
old bulk fractional basis. This has the same role as the origin vector in a
VESTA transformation. 

The transformed `x'` and `y'` directions are in the surface plane.
The transformed `z'` direction follows the third column of `P`, which is chosen
to point toward the positive surface normal.


## Search for a candidate transformation

Start from a VESTA cut or expanded bulk cell:

```python
from pyhectr.surface import (
    parse_xtl,
    compute_pairwise_vectors,
    find_vectors_with_target_angle_minimal_stream,
    find_shortest_normal_vector,
    validate_surface_transform,
)

atoms = parse_xtl("nb_5_3_9_plane.xtl")
pairs, vectors = compute_pairwise_vectors(atoms)

hkl = np.array([5, 3, 9])
match = find_vectors_with_target_angle_minimal_stream(
    pairs,
    vectors,
    target_angle=90.0,
    tol_angle=1e-6,
    surface_normal=hkl,
    surface_tol=1e-6,
    block=100_000,
)
```

The angle search finds two short in plane vectors. A third vector parallel to
the outward normal can be selected with:

```python
normal_match = find_shortest_normal_vector(
    pairs,
    vectors,
    plane_vec1,
    plane_vec2,
    normal=hkl,
)
```

For cubic Nb, fractional vector coefficients can be compared with ordinary dot
products and `[h, k, l]` is parallel to the Cartesian plane normal. For a
non cubic cell, transform candidate vectors to Cartesian space and construct
the normal through the reciprocal basis.

The search is quadratic in the number of parsed atoms because it starts from
pairwise displacements. Expand the VESTA cell only as far as needed and tune
`block` to the available memory.


Place the selected in plane vectors in the first two columns and the outward
normal vector in the third column. Round only after confirming that the values
are numerically close to integers:

```python
P = np.rint(M).astype(int)
if not np.allclose(M, P, atol=1e-8, rtol=0.0):
    raise ValueError("candidate transformation is not integer-valued")

validate_surface_transform(P, hkl)
print("det(P) =", round(np.linalg.det(P)))
print("hkl @ P =", hkl @ P)
```

For Nb(539), a valid result is:

```text
det(P) = 230
hkl @ P = [0, 0, 115]
```

`hkl @ P` contains the dot product of the Miller-index row vector with each
column of `P`. The two leading zeros show that the first two vectors lie in the
surface plane. A positive third value shows that the third vector points toward
the selected `+hkl` side. `validate_surface_transform` also requires a positive
integer determinant so the transformed basis is right handed.



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

For the supplied bcc Nb transformations, the expected atom counts are 490 for
Nb(326), 890 for Nb(438), 460 for Nb(539), and 4498 for Nb(6 4 11).

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
