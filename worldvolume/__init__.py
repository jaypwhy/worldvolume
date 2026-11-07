"""worldvolume -- Klein--Gordon modes on D-brane worldvolumes.

Solve ``-Delta f = lambda f`` on a rectangular worldvolume with a boundary
condition chosen independently on each side of each direction, and read off the
Klein--Gordon frequencies ``omega^2 = lambda + m^2``.

    >>> from worldvolume import Dirichlet, Neumann, Grid1D, solve
    >>> g = Grid1D(L=1.0, N=40, left=Dirichlet(), right=Neumann())
    >>> spec = solve(g, m=0.25, nmodes=3)
    >>> spec.omega2[:2]                       # doctest: +ELLIPSIS
    array([ 2.5299..., 22.2691...])

Neumann (``dphi/dn = 0``), Dirichlet (``phi = 0``) and Robin
(``dphi/dn + kappa phi = 0``, with the **outward** normal) are supported, with
the conventions fixed by ``docs/bc-conventions.md``.  The Neumann direction
along a brane and the Dirichlet direction transverse to it are just two
boundary conditions on two directions of the same box::

    >>> from worldvolume import Box2D
    >>> box = Box2D.from_lengths(
    ...     L1=1.0, N1=20, bc_x=Neumann(),
    ...     L2=1.5, N2=20, bc_y=Dirichlet(),
    ... )
    >>> box.n_active
    441
"""

from .analytic import (
    AnalyticSpectrum,
    dirichlet_mode,
    dirichlet_wavenumbers,
    eigenvalues,
    mixed_mode,
    mixed_nd_wavenumbers,
    neumann_mode,
    neumann_wavenumbers,
    robin_eigenvalues,
    robin_mode,
    robin_normalisation,
    robin_wavenumbers,
    spectrum_1d,
    spectrum_2d,
    wavenumbers,
)
from .bc import BoundaryCondition, Dirichlet, Neumann, Robin, as_bc
from .grid import Box2D, Grid1D
from .operators import (
    assemble,
    laplacian_1d,
    laplacian_1d_unsymmetrised,
    laplacian_2d,
    mass_matrix_1d,
)
from .solver import Spectrum, omega_from_lambda, solve, time_evolution

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # boundary conditions
    "BoundaryCondition",
    "Dirichlet",
    "Neumann",
    "Robin",
    "as_bc",
    # grids
    "Grid1D",
    "Box2D",
    # operators
    "assemble",
    "laplacian_1d",
    "laplacian_1d_unsymmetrised",
    "laplacian_2d",
    "mass_matrix_1d",
    # solver
    "Spectrum",
    "solve",
    "omega_from_lambda",
    "time_evolution",
    # analytic reference
    "AnalyticSpectrum",
    "dirichlet_mode",
    "dirichlet_wavenumbers",
    "eigenvalues",
    "mixed_mode",
    "mixed_nd_wavenumbers",
    "neumann_mode",
    "neumann_wavenumbers",
    "robin_eigenvalues",
    "robin_mode",
    "robin_normalisation",
    "robin_wavenumbers",
    "spectrum_1d",
    "spectrum_2d",
    "wavenumbers",
]
