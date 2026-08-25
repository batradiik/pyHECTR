from __future__ import annotations
from pathlib import Path
import numpy as np
import xrayutilities as xu
from numba import njit, prange


__all__ = [
    "cell_parameters_from_vectors",
    "generate_surface_cell",
    "generate_surface_files",
    "bul_file_string",
    "write_bul",
    "write_xtl",
    "validate_surface_transform",
    "parse_xtl",
    "compute_pairwise_vectors",
    "find_vectors_with_target_angle_minimal_stream",
    "find_shortest_normal_vector"
]




def validate_surface_transform(transform, hkl):
    """Validate that the transform represents the requested surface."""
    matrix = _validate_transform(transform)

    indices = np.asarray(hkl, dtype=float)
    if indices.shape != (3,) or not np.all(np.isfinite(indices)):
        raise ValueError("hkl must be a finite vector with shape (3,).")

    rounded = np.rint(indices)
    if not np.allclose(indices, rounded, atol=1e-12, rtol=0.0):
        raise ValueError("hkl must contain integer Miller indices.")

    indices = rounded.astype(np.int64)
    if not np.any(indices):
        raise ValueError("hkl must not be (0, 0, 0).")

    projections = indices @ matrix

    if projections[0] != 0 or projections[1] != 0:
        raise ValueError(
            "The first two transform columns do not lie in the (hkl) plane."
        )

    if projections[2] <= 0:
        raise ValueError(
            "The third transform column must point toward the +hkl side."
        )


def _validate_lattice_vectors(lattice_vectors):
    """Return lattice vectors as a validated ``(3, 3)`` float array."""
    lattice = np.asarray(lattice_vectors, dtype=float)

    if lattice.shape != (3, 3):
        raise ValueError(
            "lattice_vectors must have shape (3, 3), with rows equal to "
            "the Cartesian a, b, and c lattice vectors."
        )

    if not np.all(np.isfinite(lattice)):
        raise ValueError("lattice_vectors must contain only finite values.")

    if abs(np.linalg.det(lattice)) < 1e-15:
        raise ValueError("lattice_vectors must define a non-singular cell.")

    return lattice


def _validate_fractional_coordinates(fractional_coordinates,):
    """Return fractional coordinates as a validated (N, 3) float array."""
    coordinates = np.asarray(fractional_coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("fractional coordinates must have shape (N, 3).")

    if not np.all(np.isfinite(coordinates)):
        raise ValueError("fractional coordinates must contain only finite values.")

    return coordinates


def _normalize_species(species, n_atoms,):
    """Expand or validate species labels for ``n_atoms`` basis positions."""
    if isinstance(species, str):
        return [species] * n_atoms

    labels = list(species)
    if len(labels) != n_atoms:
        raise ValueError(
            "species must be a single label or contain one label for each "
            f"fractional coordinate ({n_atoms} expected, {len(labels)} given)."
        )

    if not all(isinstance(label, str) and label for label in labels):
        raise ValueError("All species labels must be non empty strings.")

    return labels


# def _validate_transform(transform):
#     """Return a validated integer-valued ``(3, 3)`` transformation matrix."""
#     matrix = np.asarray(transform, dtype=float)

#     if matrix.shape != (3, 3):
#         raise ValueError("transform must have shape (3, 3).")

#     if not np.all(np.isfinite(matrix)):
#         raise ValueError("transform must contain only finite values.")

#     if not np.allclose(matrix, np.rint(matrix), atol=1e-12, rtol=0.0):
#         raise ValueError(
#             "transform must be integer to define a periodic supercell."
#         )

#     matrix = np.rint(matrix).astype(float)
#     determinant = np.linalg.det(matrix)

#     if abs(determinant) < 1e-12:
#         raise ValueError("transform must be non singular.")

#     volume_factor = abs(determinant)
#     if not np.isclose(volume_factor, round(volume_factor), atol=1e-10):
#         raise ValueError(
#             "The absolute determinant of transform must be an integer."
#         )

#     return matrix


def _determinant_3x3(matrix):
    """Return the exact determinant of a 3×3 integer matrix."""
    a, b, c = (int(value) for value in matrix[0])
    d, e, f = (int(value) for value in matrix[1])
    g, h, i = (int(value) for value in matrix[2])

    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )


def _validate_transform(transform):
    """Return a validated right handed integer transformation matrix."""
    values = np.asarray(transform, dtype=float)

    if values.shape != (3, 3):
        raise ValueError("Transformation must have shape (3, 3).")

    if not np.all(np.isfinite(values)):
        raise ValueError("Transformation must contain only finite values.")

    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-12, rtol=0.0):
        raise ValueError("Transformation must be integer valued.")

    matrix = rounded.astype(np.int64)
    determinant = _determinant_3x3(matrix)

    if determinant == 0:
        raise ValueError("Transformation must be non singular.")

    if determinant < 0:
        raise ValueError(
            "Transformation must have positive determinant; "
            "flip one in-plane column."
        )

    return matrix


def cell_parameters_from_vectors(lattice_vectors,):
    """Calculate lattice parameters from Cartesian lattice vectors.

    Parameters
    ----------
    lattice_vectors : array-like, shape (3, 3)
        Cartesian lattice vectors in angstrom, stored as rows
        ``[a_vector, b_vector, c_vector]``.

    Returns
    -------
    a, b, c : float
        Lattice vector lengths in angstrom.
    alpha, beta, gamma : float
        Lattice angles in degrees. ``alpha`` is the angle between ``b`` and
        ``c``, ``beta`` between ``a`` and ``c``, and ``gamma`` between ``a``
        and ``b``.

    Examples
    --------
    >>> lattice = np.diag([3.0, 4.0, 5.0])
    >>> cell_parameters_from_vectors(lattice)
    (3.0, 4.0, 5.0, 90.0, 90.0, 90.0)
    """
    lattice = _validate_lattice_vectors(lattice_vectors)
    a_vector, b_vector, c_vector = lattice

    a = float(np.linalg.norm(a_vector))
    b = float(np.linalg.norm(b_vector))
    c = float(np.linalg.norm(c_vector))

    def angle(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
        cosine = np.dot(vector_1, vector_2) / (
            np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
        )
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    alpha = angle(b_vector, c_vector)
    beta = angle(a_vector, c_vector)
    gamma = angle(a_vector, b_vector)

    return a, b, c, alpha, beta, gamma


def generate_surface_cell(lattice_vectors, fractional_coordinates, 
                          transform, *, species, origin_shift = None, 
                          tolerance = 1e-12, check_atom_count = True,):
    """Generate a complete surface supercell.

    The function expands every basis atom through the integer lattice
    translations required to populate the transformed cell.

    The transformation follows the row vector convention used by pyHECTR.
    If ``P`` is ``transform``, its columns contain the coefficients of the
    new lattice vectors in the original basis:

    ``A_surface = P.T @ A_bulk``

    Fractional coordinates are transformed as

    ``f_surface = (f_bulk + n - origin_shift) @ P^{-T}``

    where ``n`` is an integer lattice translation. The generated coordinates
    are restricted to the half open interval ``[0, 1)`` in all three
    fractional directions.

    Parameters
    ----------
    lattice_vectors : array, shape (3, 3)
        Bulk Cartesian lattice vectors in angstrom, stored as rows
        ``[a_vector, b_vector, c_vector]``. 
        The cell does not need to be cubic or orthogonal.
    fractional_coordinates : array, shape (N, 3)
        Fractional coordinates of the basis atoms in the bulk cell.
    transform : array, shape (3, 3)
        Non singular integer transformation matrix. Its columns define the
        new lattice vectors in the original lattice basis.
    species : str or sequence of str
        Species labels associated with ``fractional_coordinates``. 
        A single string is broadcast to every basis atom. 
        For multi element structures, provide one label per basis atom.
    origin_shift : array, shape (3,), optional
        Origin shift in the new fractional basis. This corresponds to the
        ``p`` vector in the VESTA transformation. 
        The default is ``[0, 0, 0]``.
    tolerance : float, default=1e-12
        Numerical tolerance used for cell boundary tests and duplicate
        removal.
    check_atom_count : bool, default=True
        If ``True``, verify that the generated number of atoms equals
        ``N * abs(det(transform))``.

    Returns
    -------
    surface_lattice : ndarray, shape (3, 3)
        Cartesian lattice vectors of the transformed cell, stored as rows.
    surface_coordinates : ndarray, shape (M, 3)
        Fractional coordinates of all atoms in the transformed cell.
    surface_species : list of str
        Species labels corresponding to ``surface_coordinates``.

    Raises
    ------
    ValueError
        If input shapes are inconsistent, the transformation is singular or
        non integer, or ``tolerance`` is invalid.
    RuntimeError
        If ``check_atom_count`` is enabled and the generated atom count does
        not match the expected supercell multiplicity.

    Examples
    --------
    One element bcc cell can use one species label:

    >>> a0 = 3.3004
    >>> lattice = np.eye(3) * a0
    >>> basis = np.array([[0.0, 0.0, 0.0],
    ...                   [0.5, 0.5, 0.5]])
    >>> transform = np.array([[0, -6, 5],
    ...                       [3,  1, 3],
    ...                       [-1, 3, 9]])
    >>> surface_lattice, surface_frac, surface_species = generate_surface_cell(
    ...     lattice, basis, transform, species="Nb"
    ... )
    >>> len(surface_species)
    460

    Multi-element structures use one species label per basis position:

    >>> lattice = np.eye(3) * 4.1
    >>> basis = np.array([[0.0, 0.0, 0.0],
    ...                   [0.5, 0.5, 0.5]])
    >>> transform = np.diag([2, 1, 1])
    >>> _, surface_frac, surface_species = generate_surface_cell(
    ...     lattice, basis, transform, species=["Cs", "Cl"]
    ... )
    >>> len(surface_frac)
    4
    """
    lattice = _validate_lattice_vectors(lattice_vectors)
    basis   = _validate_fractional_coordinates(fractional_coordinates)
    labels  = _normalize_species(species, len(basis))
    matrix  = _validate_transform(transform)

    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be a finite number between 0 and 1.")

    min_tolerance = 100 * np.finfo(float).eps
    if (
        not np.isfinite(tolerance)
        or not min_tolerance <= tolerance < 0.5
    ):
        raise ValueError(
            f"tolerance must be between {min_tolerance:g} and 0.5."
        )
    # if origin_shift is None:
    #     shift_origin = np.zeros(3, dtype=float)
    # else:
    #     shift_origin = np.asarray(origin_shift, dtype=float)
    #     if shift_origin.shape != (3,):
    #         raise ValueError("origin shift must have shape (3,).")
    #     if not np.all(np.isfinite(shift_origin)):
    #         raise ValueError("origin shift must contain only finite values.")
    if origin_shift is None:
        origin_old = np.zeros(3, dtype=float)
    else:
        origin_old = np.asarray(origin_shift, dtype=float)
    
        if origin_old.shape != (3,):
            raise ValueError("origin_shift must have shape (3,).")
    
        if not np.all(np.isfinite(origin_old)):
            raise ValueError("origin_shift must contain only finite values.")

    shift_surface = -origin_old @ inverse_transpose

    surface_lattice = matrix.T @ lattice
    inverse_transpose = np.linalg.inv(matrix).T

    # f_surface lies in [0, 1). Therefore g = f_surface - origin_shift
    # lies in [-origin_shift, 1-origin_shift). Mapping the eight corners of
    # this region back through P.T gives tight bounds for the old cell lattice
    # translations that can contribute atoms to the transformed cell.
    # corner_values = [
    #     (-shift_origin[index], 1.0 - shift_origin[index])
    #     for index in range(3)
    # ]
    corner_values = [
        (-shift_surface[index], 1.0 - shift_surface[index])
        for index in range(3)
        ]
    new_frame_corners = np.array(
        [
            [x, y, z]
            for x in corner_values[0]
            for y in corner_values[1]
            for z in corner_values[2]
        ],
        dtype=float,
    )
    old_frame_corners = new_frame_corners @ matrix.T
    lower_corner = old_frame_corners.min(axis=0)
    upper_corner = old_frame_corners.max(axis=0)

    generated_coordinates: list[np.ndarray] = []
    generated_species: list[str] = []

    for basis_coordinate, label in zip(basis, labels):
        lower_translation = np.ceil(
            lower_corner - basis_coordinate - tolerance
        ).astype(int)
        upper_translation = np.floor(
            upper_corner - basis_coordinate + tolerance
        ).astype(int)

        translation_axes = [
            np.arange(lower_translation[index], upper_translation[index] + 1)
            for index in range(3)
        ]

        mesh = np.meshgrid(*translation_axes, indexing="ij")
        translations = np.stack(mesh, axis=-1).reshape(-1, 3)

        candidates = (
            (basis_coordinate + translations) @ inverse_transpose
            + shift_surface
        )
        

        inside = np.all(
            (candidates >= -tolerance) & (candidates < 1.0 - tolerance),
            axis=1,
        )

        accepted = candidates[inside]
        accepted[np.abs(accepted) < tolerance] = 0.0

        generated_coordinates.extend(accepted)
        generated_species.extend([label] * len(accepted))

    if not generated_coordinates:
        surface_coordinates = np.empty((0, 3), dtype=float)
        surface_species: list[str] = []
    else:
        coordinates_array = np.asarray(generated_coordinates, dtype=float)

        # Remove only numerical duplicates of the same species. Different
        # species at the same coordinates are preserved
        quantized = np.rint(coordinates_array / tolerance).astype(np.int64)
        keep_indices: list[int] = []
        seen: set[tuple[object, ...]] = set()

        for index, (coordinate_key, label) in enumerate(
            zip(quantized, generated_species)
        ):
            key = (label, *coordinate_key.tolist())
            if key not in seen:
                seen.add(key)
                keep_indices.append(index)

        surface_coordinates = coordinates_array[keep_indices]
        surface_species = [generated_species[index] for index in keep_indices]

    if check_atom_count:
        volume_factor = int(round(abs(np.linalg.det(matrix))))
        expected_atoms = len(basis) * volume_factor
        actual_atoms = len(surface_coordinates)

        if actual_atoms != expected_atoms:
            raise RuntimeError(
                "Generated atom count does not match the expected supercell "
                f"multiplicity: got {actual_atoms}, expected {expected_atoms} "
                f"({len(basis)} basis atoms × |det(P)|={volume_factor})."
            )

    return surface_lattice, surface_coordinates, surface_species


def bul_file_string( lattice_vectors, fractional_coordinates, species, *,
    comment = "# Bulk structure", z_shift = 0.0,
    sort_by_z = False, precision = 15, ):
    """Build the text representation of a ROD ``.bul`` structure file.

    Parameters
    ----------
    lattice_vectors : array, shape (3, 3)
        Cartesian lattice vectors in angstrom, stored as rows.
    fractional_coordinates : array, shape (N, 3)
        Fractional atomic coordinates.
    species : str or sequence of str
        Species labels. A single string is broadcast to all coordinates.
    comment : str, default="# Bulk structure"
        First line of the output file.
    z_shift : float, default=0.0
        Uniform shift applied to the fractional ``z`` coordinates before
        writing. Use ``z_shift=-1.0`` when the chosen ROD bulk convention
        requires the bulk cell to be placed one repeat below the surface.
    sort_by_z : bool, default=False
        If ``True``, write atoms from largest to smallest fractional ``z``.
        Species labels are reordered together with their coordinates.
    precision : int, default=15
        Number of digits written after the decimal point.

    Returns
    -------
    str
        Complete ``.bul`` file contents without a trailing newline.

    Examples
    --------
    >>> lattice = np.eye(3) * 3.0
    >>> coords = np.array([[0.0, 0.0, 0.0]])
    >>> text = bul_file_string(lattice, coords, "Nb", z_shift=-1.0)
    """
    lattice = _validate_lattice_vectors(lattice_vectors)
    coordinates = _validate_fractional_coordinates(
        fractional_coordinates
    ).copy()
    labels = _normalize_species(species, len(coordinates))

    if not isinstance(precision, int) or precision < 0:
        raise ValueError("precision must be a non-negative integer.")

    if not np.isfinite(z_shift):
        raise ValueError("z_shift must be finite.")

    coordinates[:, 2] += z_shift

    if sort_by_z:
        order = np.argsort(coordinates[:, 2])[::-1]
        coordinates = coordinates[order]
        labels = [labels[index] for index in order]

    a, b, c, alpha, beta, gamma = cell_parameters_from_vectors(lattice)

    parameter_line = (
        f"{a:.{precision}f} {b:.{precision}f} {c:.{precision}f} "
        f"{alpha:.{precision}f} {beta:.{precision}f} "
        f"{gamma:.{precision}f}"
    )

    atom_lines = [
        (
            f"{label} "
            f"{x:.{precision}f} {y:.{precision}f} {z:.{precision}f}"
        )
        for label, (x, y, z) in zip(labels, coordinates)
    ]

    return "\n".join([comment, parameter_line, *atom_lines])


def write_bul(filename, lattice_vectors, fractional_coordinates,
    species, *, comment = "# Bulk structure", z_shift = 0.0,
    sort_by_z = False, precision = 15,):
    """Write a structure to a ROD ``.bul`` file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Destination file.
    lattice_vectors : array, shape (3, 3)
        Cartesian lattice vectors in angstrom, stored as rows.
    fractional_coordinates : array, shape (N, 3)
        Fractional atomic coordinates.
    species : str or sequence of str
        Species labels. A single string is broadcast to all coordinates.
    comment : str, default="# Bulk structure"
        First line of the file.
    z_shift : float, default=0.0
        Uniform fractional ``z`` shift applied before writing.
    sort_by_z : bool, default=False
        If ``True``, write atoms from largest to smallest fractional ``z``.
    precision : int, default=15
        Number of digits written after the decimal point.

    Returns
    -------
    pathlib.Path
        Path to the written file.
    """
    path = Path(filename)
    text = bul_file_string(
        lattice_vectors,
        fractional_coordinates,
        species,
        comment=comment,
        z_shift=z_shift,
        sort_by_z=sort_by_z,
        precision=precision,
    )
    path.write_text(text + "\n", encoding="utf-8")
    return path


def write_xtl(filename, lattice_vectors,
    fractional_coordinates, 
    species, *, title = "Crystal structure",
    precision = 16,):
    """Write a P1 structure in VESTA XTL format.
    
    Parameters
    ----------
    filename : str or pathlib.Path
        Destination file.
    lattice_vectors : array-like, shape (3, 3)
        Cartesian lattice vectors in angstrom, stored as rows.
    fractional_coordinates : array, shape (N, 3)
        Fractional atomic coordinates.
    species : str or sequence of str
        Species labels. A single string is broadcast to all coordinates.
    title : str, default="Crystal structure"
        Structure title written to the XTL header.
    precision : int, default=16
        Number of digits written after the decimal point.

    Returns
    -------
    pathlib.Path
        Path to the written file.
    """
    lattice     = _validate_lattice_vectors(lattice_vectors)
    coordinates = _validate_fractional_coordinates(fractional_coordinates)
    labels      = _normalize_species(species, len(coordinates))

    if not isinstance(precision, int) or precision < 0:
        raise ValueError("precision must be a non-negative integer.")

    a, b, c, alpha, beta, gamma = cell_parameters_from_vectors(lattice)

    lines = [
        f"TITLE {title}",
        "CELL",
        (
            f"  {a:.{precision}f}  {b:.{precision}f}  "
            f"{c:.{precision}f}  {alpha:.{precision}f}  "
            f"{beta:.{precision}f}  {gamma:.{precision}f}"
        ),
        "SYMMETRY NUMBER 1",
        "SYMMETRY LABEL  P1",
        "ATOMS",
        "NAME         X           Y           Z",
    ]

    lines.extend(
        (
            f"{label:<2s} "
            f"{x:.{precision}f} "
            f"{y:.{precision}f} "
            f"{z:.{precision}f}"
        )
        for label, (x, y, z) in zip(labels, coordinates)
    )

    path = Path(filename)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_surface_files(
    label, transform, lattice_vectors,
    fractional_coordinates, *,
    species, output_dir = ".",
    file_stem = None, title = None,
    comment = None, save_xtl = True,
    save_bul = True, save_positive_bul = True,
    bul_z_shift = -1.0, sort_by_z = True,):
    """Generate a surface cell and optionally write XTL/BUL files.

    This is a wrapper around ``generate_surface_cell``,
    ``write_xtl`` and ``write_bul``. 
    It is intended for notebooks and scripts where the same
    transformed cell should be generated and written in one step.

    Parameters
    ----------
    label : str
        surface label, for example ``"539"`` or ``"6_4_11"``.
    transform : array-like, shape (3, 3)
        Integer transformation matrix passed to ``generate_surface_cell``.
    lattice_vectors : array-like, shape (3, 3)
        Bulk Cartesian lattice vectors in angstrom, stored as rows.
    fractional_coordinates : array-like, shape (N, 3)
        Bulk fractional basis coordinates.
    species : str or sequence of str
        Species labels. A single string is broadcast to all basis atoms.
    output_dir : str or pathlib.Path, default="."
        Directory where output files are written.
    file_stem : str, optional
        Base filename without extension. If ``None``, uses
        ``"surface_{label}"``.
    title : str, optional
        Title used in the XTL file. If ``None``, uses ``"Surface ({label})"``.
    comment : str, optional
        First line used in the BUL files. If ``None``, uses
        ``"# Surface ({label})"``.
    save_xtl : bool, default=True
        If ``True``, write ``<file_stem>.xtl``.
    save_bul : bool, default=True
        If ``True``, write shifted ROD bulk file ``<file_stem>.bul``.
    save_positive_bul : bool, default=True
        If ``True``, write unshifted file ``<file_stem>_positive.bul``.
    bul_z_shift : float, default=-1.0
        Fractional z-shift applied to the shifted BUL file.
    sort_by_z : bool, default=True
        If ``True``, write BUL atoms from largest to smallest fractional z.

    Returns
    -------
    dict
        Dictionary containing the generated lattice, fractional coordinates,
        species labels, cell parameters, expected atom count, and paths of the
        files that were written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_stem is None:
        file_stem = f"surface_{label}"

    if title is None:
        title = f"Surface ({label})"

    if comment is None:
        comment = f"# Surface ({label})"

    surface_lattice, surface_fractional, surface_species = generate_surface_cell(
        lattice_vectors=lattice_vectors,
        fractional_coordinates=fractional_coordinates,
        transform=transform,
        species=species,
    )

    expected_atoms = (
        len(fractional_coordinates)
        * int(round(abs(np.linalg.det(transform))))
    )

    if len(surface_fractional) != expected_atoms:
        raise RuntimeError(
            f"Generated atom count {len(surface_fractional)} does not match "
            f"expected count {expected_atoms}."
        )

    parameters = cell_parameters_from_vectors(surface_lattice)

    files: dict[str, Path] = {}

    if save_xtl:
        files["xtl"] = write_xtl(
            output_dir / f"{file_stem}.xtl",
            surface_lattice,
            surface_fractional,
            surface_species,
            title=title,
        )

    if save_bul:
        files["bul"] = write_bul(
            output_dir / f"{file_stem}.bul",
            surface_lattice,
            surface_fractional,
            surface_species,
            comment=comment,
            z_shift=bul_z_shift,
            sort_by_z=sort_by_z,
        )

    if save_positive_bul:
        files["positive_bul"] = write_bul(
            output_dir / f"{file_stem}_positive.bul",
            surface_lattice,
            surface_fractional,
            surface_species,
            comment=comment,
            z_shift=0.0,
            sort_by_z=sort_by_z,
        )

    return {
        "lattice": surface_lattice,
        "fractional": surface_fractional,
        "species": surface_species,
        "cell_parameters": parameters,
        "expected_atoms": expected_atoms,
        "files": files,
    }




def parse_xtl(filename):
    """
    Read fractional atomic coordinates from a VESTA XTL file.

    The parser searches for the ``ATOMS`` section and reads coordinates from
    the following atom table until an empty line or ``EOF`` is reached.

    Parameters
    ----------
    filename : str or path-like
        Path to the ``.xtl`` file.

    Returns
    -------
    atoms : ndarray, shape (N, 3)
        Atomic positions as stored in the XTL file.

    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    atoms = []
    start_index = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("ATOMS"):
            start_index = i + 2  # skip header line after "ATOMS"
            break
    if start_index is None:
        raise ValueError("Could not find the ATOMS section in the file.")
    
    for line in lines[start_index:]:
        if line.strip() == "" or line.strip().startswith("EOF"):
            break
        parts = line.split()
        if len(parts) < 4:
            continue
        pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        atoms.append(pos)
    
    return np.array(atoms)


def compute_pairwise_vectors(atoms):
    """
    Compute all unique pairwise displacement vectors.

    Parameters
    ----------
    atoms : ndarray, shape (N, 3)
        Atomic coordinates. These may be fractional or Cartesian, but the output
        vectors and areas will use the same coordinate system.

    Returns
    -------
    pairs : ndarray, shape (M, 2)
        Atom index pairs ``(i, j)`` with ``i < j``.

    vecs : ndarray, shape (M, 3)
        Displacement vectors ``atoms[j] - atoms[i]``.

    Notes
    -----
    The number of returned vectors is ``M = N * (N - 1) / 2``. The later
    angle search step can still be expensive because it compares vector pairs.
    """
    n = len(atoms)
    idx_i, idx_j = np.triu_indices(n, 1)        # upper-triangle indices
    vecs = atoms[idx_j] - atoms[idx_i]
    pairs = np.stack((idx_i, idx_j), axis=1)
    return pairs, vecs



@njit(fastmath=True, parallel=True)
def _scan_block(u_unit,  v_unit,    # (p,3) and (q,3)  –‑ normalised vectors
                u_full,  v_full,    # (p,3) and (q,3)  –‑ original vectors
                cos_t, tol,         # target cos θ and tolerance
                ns, surface_tol,    # surface‑normal (unit) & tolerance
                want_surface,       # bool
                same_block):        # True if this is a diagonal tile
    p, q = u_unit.shape[0], v_unit.shape[0]
    # thread‑local best
    best_area_i = np.full(p, 1e30)
    best_j_i    = np.full(p, -1,  dtype=np.int64)

    for i in prange(p):                       # ← parallel over i
        
        best_a  = 1e30
        best_j  = -1

        uu0, uu1, uu2 = u_unit[i]     # unit vector components
        uf0, uf1, uf2 = u_full[i]     # full‑length components
        
        for j in range(q):
            if same_block and j <= i:         # skip lower triangle
                continue
            dot = uu0 * v_unit[j,0] + uu1 * v_unit[j,1] + uu2 * v_unit[j,2]
            if abs(dot - cos_t) >= tol:
                continue

            # optional surface‑normal constraint
            if want_surface:
                cx = uu1 * v_unit[j,2] - uu2 * v_unit[j,1]
                cy = uu2 * v_unit[j,0] - uu0 * v_unit[j,2]
                cz = uu0 * v_unit[j,1] - uu1 * v_unit[j,0]
                mag = (cx*cx + cy*cy + cz*cz)**0.5
                if mag < 1e-12:
                    continue
                if abs((cx*ns[0]+cy*ns[1]+cz*ns[2]) / mag) < 1.0 - surface_tol:
                    continue

            # area with *original* (unnormalised) vectors
            vf0, vf1, vf2 = v_full[j]
            cx = uf1 * vf2 - uf2 * vf1
            cy = uf2 * vf0 - uf0 * vf2
            cz = uf0 * vf1 - uf1 * vf0
            area = (cx*cx + cy*cy + cz*cz)**0.5

            if area < best_a:
                best_a, best_j = area, j
        best_area_i[i] = best_a
        best_j_i[i]    = best_j

    # sequential reduction (tiny cost, outside prange)
    g_best = 1e30
    gi = gj = -1
    for i in range(p):
        if best_j_i[i] != -1 and best_area_i[i] < g_best:
            g_best, gi, gj = best_area_i[i], i, best_j_i[i]
    return gi, gj, g_best


def find_vectors_with_target_angle_minimal_stream(
        pairs, vecs,
        target_angle=90.0,
        tol_angle=1e-5,
        surface_normal=None,
        surface_tol=1e-5,
        block=10_000):
    """
    Find the matching vector pair with the smallest parallelogram area.

    The function scans the upper triangle of the vector-vector comparison
    matrix in blocks, avoiding construction of the full dense dot-product
    matrix in memory.

    Parameters
    ----------
    pairs : ndarray, shape (M, 2)
        Atom index pairs returned by `compute_pairwise_vectors`.

    vecs : ndarray, shape (M, 3)
        Displacement vectors returned by `compute_pairwise_vectors`.

    target_angle : float, default=90.0
        Target angle between two vectors, in degrees.

    tol_angle : float, default=1e-5
        Tolerance in cosine space. A pair is accepted when
        ``abs(dot(unit_vec1, unit_vec2) - cos(target_angle)) < tol_angle``.

    surface_normal : array, shape (3,), optional
        If given, the cross product of the two vectors must be collinear with
        this normal.

    surface_tol : float, default=1e-5
        Collinearity tolerance for the surface-normal constraint. A pair is
        accepted when ``abs(dot(cross_hat, normal_hat)) >= 1 - surface_tol``.

    block : int, default=10_000
        Tile size used during the blocked scan.

    Returns
    -------
    matches : list
        Empty list if no match is found. Otherwise returns one tuple:
        ``(pair1, pair2, vec1, vec2, actual_angle, area)``.
    """
    # ── pre‑compute unit vectors & constants ────────────────────────────
    cos_t   = np.cos(np.deg2rad(target_angle))
    norms   = np.linalg.norm(vecs, axis=1)
    valid = norms > 1e-12
    pairs = pairs[valid]
    vecs = vecs[valid]
    norms = norms[valid]
    
    v_unit  = vecs / norms[:, None]

    want_surface = surface_normal is not None
    if want_surface:
        ns = np.asarray(surface_normal, float)
        ns /= np.linalg.norm(ns)
    else:
        ns = np.zeros(3)

    best_area      = np.inf
    best_global_i  = -1
    best_global_j  = -1

    N = len(vecs)
    nvec = len(vecs)
    border = "="*80
    # ── tile over the upper triangle  ───────────────────────────────────
    total_pairs_min = nvec * (nvec - 1) // 2
    #pbar_min = tqdm(total=total_pairs_min, desc="angle scan [minimal]", unit="pair")
    for ib in range(0, N, block):
        iu_end   = min(ib + block, N)
        u_unit   = v_unit[ib:iu_end]       # (p,3)  p ≤ block
        u_orig   = vecs  [ib:iu_end]

        for jb in range(ib, N, block):
            jv_end   = min(jb + block, N)
            v_unit_blk = v_unit[jb:jv_end]     # (q,3)
            v_orig_blk = vecs  [jb:jv_end]

            same_block = (ib == jb)

            # call the NUMBA kernel 
            bi, bj, area = _scan_block(u_unit, v_unit_blk,
                                       u_orig, v_orig_blk,
                                       cos_t, tol_angle,
                                       ns, surface_tol,
                                       want_surface, same_block)

            if bi >= 0 and area < best_area:
                best_area      = area
                best_global_i  = ib + bi
                best_global_j  = jb + bj
                print(f"{best_area = }\n{best_global_i = }, {best_global_j = }\n{border}\n\n")
            #pbar_min.update(1)

    # return result in the same format as the reference implementation 
    if best_global_i < 0:
        #pbar_min.close()
        return []                              # no match

    actual_dot = np.dot(v_unit[best_global_i], v_unit[best_global_j])
    actual_angle = np.degrees(np.arccos(np.clip(actual_dot, -1.0, 1.0)))
    #pbar_min.close()
    return [(
        tuple(pairs[best_global_i]),
        tuple(pairs[best_global_j]),
        vecs[best_global_i],
        vecs[best_global_j],
        actual_angle,                                 
        best_area
    )]



def find_shortest_normal_vector(pairs, vecs,
                                plane_vec1, plane_vec2,
                                normal=(3, 2, 6),
                                col_tol=1e-6):
    """
    Find the shortest displacement vector collinear with a target normal.

    Parameters
    ----------
    pairs : ndarray, shape (M, 2)
        Atom index pairs returned by `compute_pairwise_vectors`.

    vecs : ndarray, shape (M, 3)
        Displacement vectors returned by `compute_pairwise_vectors`.

    plane_vec1, plane_vec2 : array, shape (3,)
        In-plane vectors defining the surface basis.

    normal : array, shape (3,), default=(3, 2, 6)
        Target normal direction.

    col_tol : float, default=1e-6
        Collinearity tolerance based on ``sin(theta)``.

    Returns
    -------
    result : tuple or None
        ``(atom_pair, vector)`` for the shortest matching vector, or ``None``
        if no matching vector is found.
    """
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)

    # collinearity test (sinθ ≈ |cross|/(|v||n|))
    sin_theta = np.linalg.norm(np.cross(vecs, n), axis=1) / np.linalg.norm(vecs, axis=1)
    mask = sin_theta < col_tol
    if not mask.any():
        return None

    cand_pairs = pairs[mask]
    cand_vecs  = vecs [mask]

    # orient so cross(a,b)·v > 0  (right-handed)
    area_vec = np.cross(plane_vec1, plane_vec2)
    sign = np.sign(cand_vecs @ area_vec)
    cand_vecs *= sign[:, None]

    lengths = np.linalg.norm(cand_vecs, axis=1)
    k = np.argmin(lengths)
    return tuple(cand_pairs[k]), cand_vecs[k]
