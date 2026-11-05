"""Grids for the worldvolume Klein--Gordon solver (spec section 4.1).

Nodes are equally spaced **including both endpoints**::

    x_j = j * h,   h = L / (N + 1),   j = 0, 1, ..., N + 1

so there are ``N + 2`` nodes, ``N`` of them strictly interior, and ``x_0 = 0``
and ``x_{N+1} = L`` **are nodes**.  A boundary node is needed because Neumann
and Robin conditions constrain the field *at* the boundary and the discrete
operator there needs a ghost node outside the domain (spec section 4.2);
Dirichlet nodes are simply eliminated from the unknowns (spec section 4.4), so
including them costs nothing.

``h = L/(N+1)``, not ``L/N`` and not ``L/(N-1)``: with either of those the
scheme still converges, to the wrong values, at first order in ``h``
(spec section 6 item 8).

In 2D the same construction is applied per direction, with
``h_i = L_i/(N_i+1)``, giving an ``(N_1+2) x (N_2+2)`` grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from .bc import BoundaryCondition, _coerce_pair, as_bc

__all__ = ["Grid1D", "Box2D"]


@dataclass(frozen=True)
class Grid1D:
    """A 1D interval ``[0, L]`` with ``N`` interior nodes and a BC per side.

    Parameters
    ----------
    L:
        length of the interval, in inverse mass units.  Must be positive.
    N:
        number of **interior** nodes; the grid has ``N + 2`` nodes in total and
        ``h = L / (N + 1)``.
    left, right:
        boundary conditions at ``x = 0`` and ``x = L``, chosen independently.
        Either a :class:`~worldvolume.bc.BoundaryCondition`, a string ``"N"`` or
        ``"D"``, or a real number read as a Robin coupling ``kappa``.

    Examples
    --------
    >>> from worldvolume.bc import Dirichlet, Neumann
    >>> g = Grid1D(L=1.0, N=10, left=Dirichlet(), right=Neumann())
    >>> g.h
    0.09090909090909091
    >>> g.x[[0, -1]]
    array([0., 1.])
    >>> g.active.sum()
    11
    """

    L: float
    N: int
    left: BoundaryCondition
    right: BoundaryCondition

    def __post_init__(self) -> None:
        if not np.isfinite(self.L) or self.L <= 0.0:
            raise ValueError(f"L must be a positive finite length, got {self.L!r}")
        if int(self.N) != self.N or self.N < 1:
            raise ValueError(f"N must be a positive integer, got {self.N!r}")
        object.__setattr__(self, "left", as_bc(self.left))
        object.__setattr__(self, "right", as_bc(self.right))

    # -- construction helpers -------------------------------------------------
    @classmethod
    def uniform(
        cls, L: float, N: int, bc: Union[BoundaryCondition, str, float]
    ) -> "Grid1D":
        """Interval with the **same** condition on both ends."""
        left, right = _coerce_pair(bc)
        return cls(L=L, N=N, left=left, right=right)

    # -- geometry -------------------------------------------------------------
    @property
    def h(self) -> float:
        """Grid spacing ``L / (N + 1)``."""
        return self.L / (self.N + 1)

    @property
    def n_nodes(self) -> int:
        """Total number of nodes, including both endpoints: ``N + 2``."""
        return self.N + 2

    @property
    def x(self) -> np.ndarray:
        """All node coordinates, ``x_0 = 0`` through ``x_{N+1} = L``."""
        return np.linspace(0.0, self.L, self.N + 2)

    # -- unknowns -------------------------------------------------------------
    @property
    def active(self) -> np.ndarray:
        """Boolean mask over all nodes: ``True`` where the node is an unknown.

        Dirichlet nodes are ``False`` (spec section 4.4: eliminate, do not
        constrain).
        """
        mask = np.ones(self.n_nodes, dtype=bool)
        if self.left.is_dirichlet:
            mask[0] = False
        if self.right.is_dirichlet:
            mask[-1] = False
        return mask

    @property
    def x_active(self) -> np.ndarray:
        """Coordinates of the unknown (non-Dirichlet) nodes."""
        return self.x[self.active]

    @property
    def n_active(self) -> int:
        """Number of unknowns: ``N`` plus one per non-Dirichlet end."""
        return int(self.active.sum())

    @property
    def weights(self) -> np.ndarray:
        """Trapezoidal weights ``w_j`` on the **active** nodes (spec section 4.3).

        ``w_j = 1/2`` at a boundary node carrying a Neumann or Robin condition,
        ``w_j = 1`` at every interior node.  Dirichlet nodes are absent because
        they are not unknowns.  These are the diagonal entries of the mass
        matrix ``W`` of the symmetric generalised problem
        ``A_sym v = lambda W v``.
        """
        w = np.ones(self.n_nodes)
        if self.left.is_robin_family:
            w[0] = 0.5
        if self.right.is_robin_family:
            w[-1] = 0.5
        return w[self.active]

    def __repr__(self) -> str:
        return (
            f"Grid1D(L={self.L!r}, N={self.N!r}, left={self.left!r}, "
            f"right={self.right!r})"
        )


@dataclass(frozen=True)
class Box2D:
    """A 2D rectangular worldvolume ``[0, L_1] x [0, L_2]``.

    Built from two :class:`Grid1D` objects, one per direction, so each of the
    four boundary pieces carries its own condition::

        box = Box2D(
            x=Grid1D(L=1.0, N=32, left=Neumann(), right=Neumann()),
            y=Grid1D(L=1.5, N=32, left=Dirichlet(), right=Dirichlet()),
        )

    The ``x`` grid is the first direction (the slowest-varying index of the
    flattened unknown vector); see :func:`worldvolume.operators.laplacian_2d`
    for the index convention.
    """

    x: Grid1D
    y: Grid1D

    @classmethod
    def from_lengths(
        cls,
        L1: float,
        N1: int,
        bc_x: Union[BoundaryCondition, str, float, Sequence],
        L2: float,
        N2: int,
        bc_y: Union[BoundaryCondition, str, float, Sequence],
    ) -> "Box2D":
        """Convenience constructor from lengths and per-direction conditions.

        ``bc_x`` and ``bc_y`` are either a single condition applied to both
        ends of that direction, or a ``(left, right)`` pair.
        """
        lx, rx = _coerce_pair(bc_x)
        ly, ry = _coerce_pair(bc_y)
        return cls(
            x=Grid1D(L=L1, N=N1, left=lx, right=rx),
            y=Grid1D(L=L2, N=N2, left=ly, right=ry),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_x, n_y)``: the number of nodes per direction, including ends."""
        return (self.x.n_nodes, self.y.n_nodes)

    @property
    def active(self) -> np.ndarray:
        """2D boolean mask of unknowns, shape ``(n_x, n_y)``.

        This is the product of the two 1D active sets, so a Dirichlet face is
        removed in its entirety, corners included (spec section 4.4).
        """
        return np.outer(self.x.active, self.y.active)

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Node coordinates ``(X, Y)``, each of shape ``(n_x, n_y)``."""
        return np.meshgrid(self.x.x, self.y.x, indexing="ij")

    def __repr__(self) -> str:
        return f"Box2D(x={self.x!r}, y={self.y!r})"
