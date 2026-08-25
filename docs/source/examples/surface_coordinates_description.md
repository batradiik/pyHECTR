# Surface Coordinate Convention

`pyhectr.surface` generates complete periodic bulk cells in a surface oriented
coordinate system. Its first two lattice vectors lie in the selected surface plane, and its third lattice vector points toward the chosen outward surface normal.

## Transformation Matrix

Cartesian lattice vectors are stored as rows:

```python
A_bulk = np.array([a_vector, b_vector, c_vector])
```

For an integer transformation matrix `P`, the columns of `P` contain the new
surface cell vectors expressed in the original bulk basis. With this convention,

```python
A_surface = P.T @ A_bulk
```

Fractional coordinates are transformed as

```python
f_surface = (f_bulk + n - p) @ inv(P).T
```

where `n` is an integer bulk lattice translation and `p` is an optional origin
shift in the original bulk fractional basis. This is the same role as the
origin vector in a VESTA transformation.

## Surface Validation

For a cubic parent cell, the direct space normal to the `(hkl)` plane is
parallel to `[h, k, l]`. A valid surface transform should satisfy

```python
hkl @ P == [0, 0, positive]
```

The first two columns of `P` are then in-plane vectors. The third column points
toward the positive side of the surface normal. A positive determinant is used
to keep the transformed cell right handed.

## Nb High-Index Workflow

The Nb example notebooks use VESTA only as an exploratory visualization step.
An expanded bcc Nb cell is cut along a candidate surface such as Nb(326), and
the exported `.xtl` coordinates are searched for two short in-plane repeat
vectors. Together with the outward normal `[h, k, l]`, these vectors define the
integer transformation matrix `P`.

The final `.xtl` and ROD `.bul` files are generated directly from `P` in Python.
This avoids rounded VESTA exported fractional coordinates, which can cause
numerical artifacts in large high-index structure-factor calculations.

For conventional bcc Nb with two basis atoms, the generated atom count is

```python
N_atoms = 2 * abs(det(P))
```

The rectangular cells are larger than primitive oblique surface cells, but they
give one consistent coordinate convention for file generation, layer grouping,
displacement parameters, and comparison between candidate orientations.
