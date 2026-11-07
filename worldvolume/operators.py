"""Sparse finite-difference Laplacian with mixed boundary conditions.

This module implements spec section 4: a second-order central scheme on the
grid of :mod:`worldvolume.grid`, with ghost nodes at Neumann/Robin ends,
Dirichlet nodes eliminated, assembled in the **symmetric** form

    A_sym v = lambda W v,        W = diag(1/2, 1, ..., 1, 1/2)

(spec sections 4.3, 4.6-4.8).

The stencil
-----------
Interior node ``j`` (spec eq. 4.2)::

    (-Delta_h f)_j = (-f_{j-1} + 2 f_j - f_{j+1}) / h^2

Boundary node with a Neumann or Robin condition: the *same* three-point
formula, with the ghost value ``f_{-1}`` (or ``f_{N+2}``) eliminated using the
**central** difference of the derivative at the boundary node (spec eqs.
4.3-4.4).  With the outward normal in mind -- ``d/dn = -d/dx`` at ``x = 0`` and
``+d/dx`` at ``x = L`` -- the ghost relations are::

    x = 0:   f_{-1}  = f_1 - 2 h kappa_0 f_0          (kappa_0 = 0 for Neumann)
    x = L:   f_{N+2} = f_N - 2 h kappa_L f_{N+1}      (kappa_L = 0 for Neumann)

Note that the right-hand relation uses ``f_N``, not ``f_{N+1}``: a one-sided
difference there destroys both the second-order accuracy and the symmetry
(spec section 6 item 5).  Substituting gives the modified boundary rows
(spec eqs. 4.5)::

    (-Delta_h f)_0      = ((2 + 2 h kappa_0) f_0      - 2 f_1)     / h^2
    (-Delta_h f)_{N+1}  = (-2 f_N + (2 + 2 h kappa_L) f_{N+1})     / h^2

The resulting matrix ``A'`` is *not* symmetric -- the (0,1) entry is
``-2/h^2`` while (1,0) is ``-1/h^2``.  That is intrinsic to ghost elimination,
not an error.  ``A'`` is self-adjoint in the weighted inner product with the
trapezoidal weights ``w = (1/2, 1, ..., 1, 1/2)``, i.e. ``A_sym = W A'`` *is*
symmetric (spec eq. 4.8), and the eigenproblem is the symmetric generalised one
``A_sym v = lambda W v``.  Halving the boundary rows of ``A'`` while solving the
*ordinary* eigenproblem is a different problem with different eigenvalues
(spec section 4.3, trap #4).

Index convention in 2D
----------------------
``A = A1 (x) W2 + W1 (x) A2`` (spec eq. 4.11) is built with
``scipy.sparse.kron(A1, W2)``, in which the **first** factor varies slowest.
The flattened unknown index is therefore ``i * n_y + j`` for node
``(x_i, y_j)``: row-major with the ``y`` index fastest, so that
``v.reshape(n_x, n_y)[i, j]`` is the value at ``(x_i, y_j)``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .grid import Box2D, Grid1D

__all__ = [
    "laplacian_1d_unsymmetrised",
    "laplacian_1d",
    "mass_matrix_1d",
    "laplacian_2d",
    "assemble",
]


def _add(rows: list, cols: list, vals: list, i: int, j: int, v: float) -> None:
    rows.append(i)
    cols.append(j)
    vals.append(v)


def laplacian_1d_unsymmetrised(grid: Grid1D) -> sp.csr_matrix:
    """The ghost-eliminated ``A'`` of spec section 4.2, on the active nodes.

    This is the *unsymmetrised* operator ``-Delta_h``: rows at a Neumann/Robin
    boundary node carry a factor 2 in the off-diagonal entry, so ``A'`` is not
    symmetric.  It is exposed so that the symmetrisation ``A_sym = W A'`` can be
    checked directly (spec section 4.3).

    Dirichlet nodes are deleted (rows and columns), so for an all-Dirichlet
    problem this is the textbook ``(1/h^2) tridiag(-1, 2, -1)`` of size ``N``.
    """
    h = grid.h
    n = grid.n_nodes
    inv_h2 = 1.0 / (h * h)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # --- left boundary node x = 0 (spec eq. 4.5a) ---------------------------
    if grid.left.is_robin_family:
        kappa0 = grid.left.kappa
        _add(rows, cols, vals, 0, 0, (2.0 + 2.0 * h * kappa0) * inv_h2)
        _add(rows, cols, vals, 0, 1, -2.0 * inv_h2)

    # --- interior nodes (spec eq. 4.2) --------------------------------------
    for j in range(1, n - 1):
        _add(rows, cols, vals, j, j - 1, -inv_h2)
        _add(rows, cols, vals, j, j, 2.0 * inv_h2)
        _add(rows, cols, vals, j, j + 1, -inv_h2)

    # --- right boundary node x = L (spec eq. 4.5b) --------------------------
    if grid.right.is_robin_family:
        kappaL = grid.right.kappa
        _add(rows, cols, vals, n - 1, n - 2, -2.0 * inv_h2)
        _add(rows, cols, vals, n - 1, n - 1, (2.0 + 2.0 * h * kappaL) * inv_h2)

    a_full = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    active = grid.active
    return a_full[active][:, active].tocsr()


def mass_matrix_1d(grid: Grid1D) -> sp.dia_matrix:
    """The diagonal weight matrix ``W`` of the generalised problem (eq. 4.7).

    ``w_j = 1/2`` at a Neumann/Robin boundary node, ``1`` at an interior node,
    on the active nodes only.  ``W`` is the trapezoidal-rule mass matrix: the
    continuum inner product ``int f g`` is discretised as
    ``sum_j h w_j u_j v_j``, which is why eigenvectors of the generalised
    problem satisfy ``v_m^T W v_n = delta_mn`` and must be divided by
    ``sqrt(h)`` to represent function values (spec section 4.6).
    """
    return sp.dia_matrix(
        (grid.weights[np.newaxis, :], [0]), shape=(grid.n_active, grid.n_active)
    ).tocsr()


def laplacian_1d(grid: Grid1D) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Symmetric 1D pair ``(A_sym, W)`` for ``A_sym v = lambda W v``.

    ``A_sym = W @ A'`` where ``A'`` is :func:`laplacian_1d_unsymmetrised`; it is
    symmetric exactly (spec section 4.3), which is what makes the generalised
    problem solvable by ``scipy.linalg.eigh`` / ``eigsh`` with real eigenvalues.
    """
    a_prime = laplacian_1d_unsymmetrised(grid)
    w = mass_matrix_1d(grid)
    a_sym = (w @ a_prime).tocsr()
    return a_sym, w


def laplacian_2d(box: Box2D) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """2D Kronecker assembly (spec eq. 4.11).

    Returns ``(A, M)`` with::

        A = A1 (x) W2 + W1 (x) A2,      M = W1 (x) W2

    The eigenvalues of the pair are exactly the sums of the two 1D eigenvalue
    sets, the discrete counterpart of ``lambda_{n1 n2} = k_{n1}^2 + k_{n2}^2``
    (spec eq. 3.20).  The flattened index is ``i * n_y + j`` for node
    ``(x_i, y_j)`` (first Kronecker factor slowest); see the module docstring.
    """
    a1, w1 = laplacian_1d(box.x)
    a2, w2 = laplacian_1d(box.y)

    a = (sp.kron(a1, w2, format="csr") + sp.kron(w1, a2, format="csr")).tocsr()
    m = sp.kron(w1, w2, format="csr").tocsr()
    return a, m


def assemble(obj: Grid1D | Box2D) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Dispatch to :func:`laplacian_1d` or :func:`laplacian_2d`."""
    if isinstance(obj, Grid1D):
        return laplacian_1d(obj)
    if isinstance(obj, Box2D):
        return laplacian_2d(obj)
    raise TypeError(f"expected Grid1D or Box2D, got {type(obj).__name__}")
