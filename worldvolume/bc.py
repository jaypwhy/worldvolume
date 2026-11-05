"""Boundary conditions for the worldvolume Klein--Gordon solver.

Conventions are those of ``docs/bc-conventions.md`` (spec sections 2.1 and 2.2).
Let ``dphi/dn`` be the derivative along the **outward** normal.  The three
conditions are

===============  ==========================================
Neumann          ``dphi/dn = 0``
Dirichlet        ``phi = 0``
Robin            ``dphi/dn + kappa * phi = 0``, ``kappa`` real
===============  ==========================================

On an interval ``[0, L]`` the outward normal is ``-x_hat`` at ``x = 0`` and
``+x_hat`` at ``x = L``, so a Robin condition with the *same* ``kappa`` at both
ends reads

    x = 0:   -phi'(0) + kappa * phi(0) = 0      i.e.  phi'(0) = +kappa * phi(0)
    x = L:   +phi'(L) + kappa * phi(L) = 0      i.e.  phi'(L) = -kappa * phi(L)

-- opposite signs for the same ``kappa``.  This is spec section 2.1 and the
single most common source of silently wrong spectra (spec section 6 item 1).

``kappa = 0`` is exactly Neumann and ``kappa -> infinity`` is exactly Dirichlet
(spec section 2.2), so the Robin family contains every case the solver
supports.  Negative ``kappa`` is allowed: it is self-adjoint and produces
negative eigenvalues (spec section 3.5.6), which the solver must return rather
than "fix".

A boundary condition is assigned to each side of each direction
independently::

    from worldvolume import Dirichlet, Neumann, Robin
    from worldvolume.grid import Grid1D

    Grid1D(L=1.0, N=40, left=Dirichlet(), right=Neumann())   # mixed N/D
    Grid1D(L=1.0, N=40, left=Robin(2.0), right=Robin(0.5))   # Robin, unequal
    Grid1D(L=1.0, N=40, left=Robin(0.0), right=Neumann())    # Robin(0) == Neumann
"""

from __future__ import annotations

import math
from typing import Union

__all__ = [
    "BoundaryCondition",
    "Neumann",
    "Dirichlet",
    "Robin",
    "as_bc",
]


class BoundaryCondition:
    """Base class for a boundary condition on one side of one direction.

    Instances are immutable, hashable and comparable by value.  Use the
    concrete classes :class:`Neumann`, :class:`Dirichlet` and :class:`Robin`.
    """

    #: short tag: ``"N"``, ``"D"`` or ``"R"``
    kind: str = "?"

    @property
    def kappa(self) -> float:
        """Robin coupling in the convention ``dphi/dn + kappa*phi = 0``.

        ``0.0`` for Neumann, ``math.inf`` for Dirichlet.
        """
        raise NotImplementedError

    @property
    def is_neumann(self) -> bool:
        return self.kind == "N"

    @property
    def is_dirichlet(self) -> bool:
        return self.kind == "D"

    @property
    def is_robin(self) -> bool:
        return self.kind == "R"

    @property
    def is_robin_family(self) -> bool:
        """True when the side needs a ghost node (Neumann or Robin)."""
        return self.kind in ("N", "R")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundaryCondition):
            return NotImplemented
        return self.kind == other.kind and self.kappa == other.kappa

    def __hash__(self) -> int:
        return hash((self.kind, self.kappa))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Neumann(BoundaryCondition):
    """Neumann: ``dphi/dn = 0``.  Contains the constant zero mode."""

    kind = "N"

    def __init__(self) -> None:
        pass

    @property
    def kappa(self) -> float:
        return 0.0


class Dirichlet(BoundaryCondition):
    """Dirichlet: ``phi = 0``.  The node is eliminated from the unknowns."""

    kind = "D"

    def __init__(self) -> None:
        pass

    @property
    def kappa(self) -> float:
        return math.inf


class Robin(BoundaryCondition):
    """Robin: ``dphi/dn + kappa*phi = 0`` with a finite real ``kappa``.

    ``kappa = 0`` is Neumann and large positive ``kappa`` approaches Dirichlet;
    both limits are handled exactly by the solver, so ``Robin(0.0)`` is a
    legitimate spelling of ``Neumann()``.

    ``kappa = +-inf`` is rejected: write :class:`Dirichlet` instead.  A silently
    accepted infinite coupling would put ``inf`` entries in the assembled
    matrix, and a silently *coerced* one would hide a modelling mistake.
    """

    kind = "R"

    __slots__ = ("_kappa",)

    def __init__(self, kappa: float) -> None:
        kappa = float(kappa)
        if not math.isfinite(kappa):
            raise ValueError(
                "Robin requires a finite kappa (got %r); use Dirichlet() for "
                "phi = 0, which is the kappa -> infinity limit." % (kappa,)
            )
        self._kappa = kappa

    @property
    def kappa(self) -> float:
        return self._kappa

    def __repr__(self) -> str:
        return f"Robin(kappa={self._kappa!r})"


BCLike = Union[BoundaryCondition, str, float, int]


def as_bc(obj: BCLike) -> BoundaryCondition:
    """Coerce ``obj`` to a :class:`BoundaryCondition`.

    Accepts a :class:`BoundaryCondition` (returned unchanged), the strings
    ``"N"`` or ``"D"`` (case-insensitive), or a real number, which is read as a
    Robin coupling ``kappa``.
    """
    if isinstance(obj, BoundaryCondition):
        return obj
    if isinstance(obj, str):
        key = obj.strip().upper()
        if key in ("N", "NEUMANN"):
            return Neumann()
        if key in ("D", "DIRICHLET"):
            return Dirichlet()
        raise ValueError(
            f"cannot interpret {obj!r} as a boundary condition; use "
            "'N', 'D', Neumann(), Dirichlet() or Robin(kappa)"
        )
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return Robin(float(obj))
    raise TypeError(f"cannot interpret {obj!r} as a boundary condition")


def _coerce_pair(
    bc: Union[BCLike, tuple, list],
) -> tuple[BoundaryCondition, BoundaryCondition]:
    """Turn ``bc`` into ``(left, right)``.

    A single condition is applied to both ends; a 2-sequence gives the two ends.
    """
    if isinstance(bc, (tuple, list)):
        if len(bc) != 2:
            raise ValueError(f"expected a pair of boundary conditions, got {bc!r}")
        return as_bc(bc[0]), as_bc(bc[1])
    single = as_bc(bc)
    return single, single
