"""Solve the Klein--Gordon spectral problem on a worldvolume.

The spectral problem is (spec eqs. 1.5-1.6)

    -Delta f = lambda f  on the worldvolume, with the assigned BC per boundary,
    omega^2 = lambda + m^2.

The mass enters **only** through the shift (1.6): it does not touch the
eigenfunctions and does not touch the spectrum of ``-Delta``.  So the solver
computes ``lambda`` from the boundary conditions alone and applies the shift as
a post-processing step.  ``omega`` is returned as a **complex** array, because
for ``m^2 < 0`` (a tachyon) or ``kappa < 0`` the combination ``lambda + m^2``
can be negative and the mode then grows as ``e^{+|omega| t}`` (spec 1.4 item
2).  It is never ``nan`` and never silently made positive.

Discretisation (spec sections 4.3, 4.8): with ``A_sym`` and ``W`` from
:mod:`worldvolume.operators`, solve the symmetric generalised problem

    A_sym v = lambda W v,        v^T W v = I

via ``scipy.linalg.eigh`` (dense) or ``scipy.sparse.linalg.eigsh`` with
shift-invert (sparse).  Both return real eigenvalues and ``W``-orthonormal
eigenvectors.  Function values are then ``u = v / sqrt(h)`` (in 2D
``v / sqrt(h_1 h_2)``) -- spec section 4.6, trap #6: the ``sqrt(h)`` factor is
what makes the discrete mode comparable with the continuum eigenfunction
``int |f|^2 = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import Box2D, Grid1D
from .operators import assemble

__all__ = ["Spectrum", "solve", "omega_from_lambda", "time_evolution"]


def omega_from_lambda(lam: Sequence[float], m: float = 0.0) -> np.ndarray:
    """``omega = sqrt(lambda + m^2)`` on the principal branch, as complex.

    ``lambda + m^2 >= 0`` gives a real frequency; ``lambda + m^2 < 0`` gives
    ``+i sqrt(-(lambda + m^2))``, the growing-mode frequency of spec 1.4 item 2.
    """
    lam = np.asarray(lam, dtype=float)
    return np.sqrt((lam + m * m).astype(complex))


@dataclass
class Spectrum:
    """A computed spectrum and the eigenvectors that go with it.

    Attributes
    ----------
    lam:
        eigenvalues of ``-Delta``, ascending.
    omega2, omega:
        ``lambda + m^2`` and its principal square root (complex).
    v:
        raw eigenvectors, shape ``(ndof, nmodes)``, ``W``-orthonormal:
        ``v^T W v = I``.  These are the vectors the eigensolver returns.
    u:
        the same vectors rescaled to function values, ``u = v / sqrt(h)``
        (``sqrt(h_1 h_2)`` in 2D), so that ``u`` approximates the continuum
        eigenfunction normalised by ``int |f|^2 = 1``.
    weights:
        the trapezoidal weights ``w`` on the active nodes.
    h:
        grid spacing (a 2-tuple in 2D).
    domain:
        the :class:`~worldvolume.grid.Grid1D` or :class:`~worldvolume.grid.Box2D`
        that was solved.
    m:
        mass used for the ``omega`` shift.
    method:
        ``"dense"`` or ``"sparse"``, which eigensolver ran.
    """

    lam: np.ndarray
    omega2: np.ndarray
    omega: np.ndarray
    v: np.ndarray
    u: np.ndarray
    weights: np.ndarray
    h: object
    domain: object
    m: float = 0.0
    method: str = "dense"

    # -- convenience ---------------------------------------------------------
    @property
    def ndim(self) -> int:
        return 1 if isinstance(self.domain, Grid1D) else 2

    @property
    def nmodes(self) -> int:
        return len(self.lam)

    def __len__(self) -> int:
        return self.nmodes

    def __repr__(self) -> str:
        first = self.lam[0] if self.nmodes else float("nan")
        return (
            f"Spectrum(nmodes={self.nmodes}, m={self.m!r}, "
            f"lam[0]={first!r}, method={self.method!r})"
        )

    def mode(self, index: int) -> np.ndarray:
        """Eigenfunction ``index`` as an array on the **full** grid.

        Dirichlet nodes are filled with zero, so the returned array has the
        shape of the grid (``(n_nodes,)`` in 1D, ``(n_x, n_y)`` in 2D) and its
        entries are the function values ``f(x_j)``.
        """
        vec = self.u[:, index]
        if self.ndim == 1:
            out = np.zeros(self.domain.n_nodes, dtype=vec.dtype)
            out[self.domain.active] = vec
            return out
        out = np.zeros(self.domain.shape, dtype=vec.dtype)
        out[self.domain.active] = vec
        return out

    def grid_values(self, index: int) -> tuple[np.ndarray, ...]:
        """Coordinates and values for plotting mode ``index``."""
        if self.ndim == 1:
            return (self.domain.x, self.mode(index))
        xx, yy = self.domain.mesh()
        return xx, yy, self.mode(index)

    def project(self, values: np.ndarray) -> np.ndarray:
        """Coefficients ``c_n = <f_n, values>`` in the discrete inner product.

        ``values`` is given on the full grid (Dirichlet entries are ignored).
        Uses ``sum_j h w_j u_nj values_j``, the trapezoidal discretisation of
        ``int f_n phi`` (spec 1.7).

        Since ``u = v / sqrt(h)`` and ``v^T W v = I``, this equals
        ``sqrt(h) * v^T W values``.  The ``h`` factor is **not** optional: it
        is what makes the projection of a normalised eigenfunction return
        ``1``.  Dropping it returns ``1/sqrt(h)`` instead (a factor of
        ``sqrt(31) ~ 5.57`` on an ``N = 30`` grid), which is the mode
        coefficient in the raw ``v`` basis, not the physical one.
        """
        flat = _flatten_active(self.domain, values)
        return _hfactor(self.domain) * (self.u.T @ (self.weights * flat))


def _flatten_active(domain, values: np.ndarray) -> np.ndarray:
    """Restrict ``values`` (full-grid shaped) to the active nodes, flattened."""
    values = np.asarray(values, dtype=float)
    if isinstance(domain, Grid1D):
        if values.shape != (domain.n_nodes,):
            raise ValueError(
                f"expected values on all {domain.n_nodes} nodes, got shape "
                f"{values.shape}"
            )
        return values[domain.active]
    if values.shape != domain.shape:
        raise ValueError(
            f"expected values on the full grid {domain.shape}, got shape "
            f"{values.shape}"
        )
    return values[domain.active]


def solve(
    domain: Grid1D | Box2D,
    m: float = 0.0,
    nmodes: int | None = None,
    *,
    method: str = "auto",
    dense_threshold: int = 1200,
) -> Spectrum:
    """Solve ``-Delta f = lambda f`` with the domain's boundary conditions.

    Parameters
    ----------
    domain:
        a :class:`~worldvolume.grid.Grid1D` or :class:`~worldvolume.grid.Box2D`,
        carrying one boundary condition per side of each direction.
    m:
        mass; used only for ``omega^2 = lambda + m^2`` (spec 1.6).
    nmodes:
        how many of the lowest eigenvalues to return.  ``None`` returns all of
        them (only possible on the dense path).
    method:
        ``"auto"`` (dense for small problems, sparse otherwise), ``"dense"``
        (``scipy.linalg.eigh`` on the generalised problem) or ``"sparse"``
        (``eigsh`` with shift-invert).

    Returns
    -------
    :class:`Spectrum`
    """
    a, mass = assemble(domain)
    ndof = a.shape[0]
    if nmodes is None:
        nmodes = ndof
    nmodes = int(min(nmodes, ndof))
    if nmodes < 1:
        raise ValueError("nmodes must be at least 1")

    if method == "auto":
        method = "dense" if ndof <= dense_threshold or nmodes >= ndof - 1 else "sparse"
    if method not in ("dense", "sparse"):
        raise ValueError(f"unknown method {method!r}")

    if method == "dense":
        lam, v = sla.eigh(a.toarray(), mass.toarray())
    else:
        # shift-invert around sigma; A can be singular for kappa < 0 (zero mode
        # at kappa = -2/L), so nudge the shift off zero when a coupling is
        # negative (spec 3.5.6).
        sigma = 0.0
        if _any_negative_coupling(domain):
            sigma = -1e-8 * float(abs(a).max())
        lam, v = spla.eigsh(
            a, k=nmodes, M=mass, sigma=sigma, which="LM", tol=0.0, maxiter=100000
        )
        order = np.argsort(lam)
        lam = lam[order]
        v = v[:, order]

    # physical function values: u = v / sqrt(h) (per direction in 2D, 4.6)
    h = _hfactor(domain)
    u = v / np.sqrt(h)

    weights = mass.diagonal()
    lam = np.asarray(lam, dtype=float)[:nmodes]
    v = np.asarray(v, dtype=float)[:, :nmodes]
    u = np.asarray(u, dtype=float)[:, :nmodes]

    # SPEC-DISCREPANCY (spec 3.3, 6 item 2): the spec calls omega_0^2 = m^2
    # "exactly".  What is exact is lambda_0 = 0 as an eigenvalue of the
    # assembled matrix (the constant vector is annihilated bitwise); `eigh`
    # returns it as O(1e-14) round-off, so omega_0^2 = m^2 + O(1e-14) and
    # omega_0 = |m| + O(1e-14/(2|m|)) -- and for m = 0, omega_0 = sqrt(lam_0)
    # = O(1e-7), not 0.  Behaviour left as is; the theorist is adjudicating.
    return Spectrum(
        lam=lam,
        omega2=lam + m * m,
        omega=omega_from_lambda(lam, m),
        v=v,
        u=u,
        weights=weights,
        h=h,
        domain=domain,
        m=m,
        method=method,
    )


def _hfactor(domain: Grid1D | Box2D) -> float:
    if isinstance(domain, Grid1D):
        return domain.h
    return domain.x.h * domain.y.h


def _any_negative_coupling(domain: Grid1D | Box2D) -> bool:
    grids = [domain] if isinstance(domain, Grid1D) else [domain.x, domain.y]
    return any(
        g.left.is_robin and g.left.kappa < 0.0 or g.right.is_robin and g.right.kappa < 0.0
        for g in grids
    )


def time_evolution(
    spectrum: Spectrum,
    phi0: np.ndarray,
    dphi0: np.ndarray,
    times: Sequence[float],
) -> np.ndarray:
    """Evolve initial data in time on the modes of ``spectrum`` (spec 1.6).

    Given ``phi(0, x)`` and ``dphi/dt(0, x)`` on the grid, project onto the
    eigenfunctions and resum (spec eq. 1.9 in real form)::

        phi(t, x) = sum_n [ a_n cos(omega_n t) + b_n sin(omega_n t)/omega_n ] f_n(x)

    with ``a_n = <f_n, phi0>`` and ``b_n = <f_n, dphi0>``.  The ``sin(x)/x``
    form is used rather than complex exponentials so that it is manifestly real
    and so that the zero mode (``omega = 0``) is handled by its limit ``t``.
    For imaginary ``omega`` the same expressions are ``cosh`` and ``sinh``, so
    exponentially growing modes come out correctly.

    Parameters
    ----------
    spectrum:
        a solved spectrum; the number of modes used is ``spectrum.nmodes``.
    phi0, dphi0:
        initial field and its time derivative, on the full grid (Dirichlet
        entries are ignored -- they are zero by construction).
    times:
        the times to evaluate at.

    Returns
    -------
    ndarray of shape ``(len(times),) + grid shape``.
    """
    # a_n = <f_n, phi0> and b_n = <f_n, dphi0>, the same physical
    # coefficients :meth:`Spectrum.project` returns.  The resummation then
    # uses the function values ``u`` -- ``a_n u_n = (v_n^T W phi0) v_n``,
    # so this is the same expansion in the raw ``v`` basis, with the
    # sqrt(h) factors explicitly on both sides.
    a = spectrum.project(phi0)
    b = spectrum.project(dphi0)

    times = np.asarray(times, dtype=float)
    out = np.empty((times.size, spectrum.v.shape[0]), dtype=float)
    for it, t in enumerate(times):
        acc = np.zeros(spectrum.v.shape[0], dtype=complex)
        for n in range(spectrum.nmodes):
            om = spectrum.omega[n]
            if abs(om) < 1e-12:
                cos_t, sinc_t = 1.0 + 0.0j, complex(t)
            else:
                cos_t = np.cos(om * t)
                sinc_t = np.sin(om * t) / om
            acc += (a[n] * cos_t + b[n] * sinc_t) * spectrum.u[:, n]
        if np.max(np.abs(acc.imag)) > 1e-8 * max(1.0, np.max(np.abs(acc.real))):
            raise RuntimeError(
                "time evolution produced a complex field from real initial "
                "data -- this indicates a non-real eigenvalue (a symmetry bug)"
            )
        out[it] = acc.real

    # expand back to the full grid, with zeros at the Dirichlet nodes, so that
    # the result has the same shape as the initial data
    if spectrum.ndim == 1:
        full = np.zeros((times.size, spectrum.domain.n_nodes))
        full[:, spectrum.domain.active] = out
        return full
    full = np.zeros((times.size,) + spectrum.domain.shape)
    full[:, spectrum.domain.active] = out
    return full
