"""Closed-form spectra of ``-Delta`` on an interval and on a rectangle.

Everything here is spec section 3 (``docs/bc-conventions.md``).  The 1D results
are the building blocks; the 2D box is assembled from them by (3.20),
``lambda_{n1 n2} = k_{n1}^{(1)2} + k_{n2}^{(2)2}``.

Sign and normalisation conventions (fixed by the spec -- do not improvise):

* Eigenfunctions carry **unit L2 norm** (1.8): no ``1/sqrt(2 omega)`` factor.
* Wavenumbers are ``n pi / L``, never ``2 pi n / L`` (box, not circle, 1.7).
* The Neumann zero mode is normalised with ``1/sqrt(L)``, the other Neumann
  modes with ``sqrt(2/L)`` (3.3b).
* Mixed Neumann/Dirichlet intervals have ``k_n = (n + 1/2) pi / L``, ``n >= 0``
  (3.6) -- half the Dirichlet fundamental, no zero mode.
* Robin uses ``dphi/dn + kappa phi = 0`` with the outward normal, so the
  spectral condition is (3.12)

      (kappa_0 kappa_L - k^2) sin(kL) + (kappa_0 + kappa_L) k cos(kL) = 0

  for ``lambda = k^2 > 0``, and (3.19)

      (kappa_0 kappa_L + mu^2) sinh(mu L) + (kappa_0 + kappa_L) mu cosh(mu L) = 0

  for ``lambda = -mu^2 < 0``.  **Both** are solved here: (3.12) alone is
  incomplete whenever a coupling is negative (spec section 3.5.6, item 7).
  ``k = 0`` is a *degeneracy* of (3.12), not a root -- it is an eigenvalue iff
  (3.18) holds, and is handled once, outside the scan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .bc import BoundaryCondition, Dirichlet, Neumann, Robin, as_bc
from .solver import omega_from_lambda

__all__ = [
    "AnalyticSpectrum",
    "dirichlet_wavenumbers",
    "neumann_wavenumbers",
    "mixed_nd_wavenumbers",
    "robin_wavenumbers",
    "robin_eigenvalues",
    "wavenumbers",
    "eigenvalues",
    "dirichlet_mode",
    "neumann_mode",
    "mixed_mode",
    "robin_mode",
    "robin_normalisation",
    "spectrum_1d",
    "spectrum_2d",
]

_TINY = 1.0e-300


# ---------------------------------------------------------------------------
# Closed forms for D, N and mixed N/D intervals
# ---------------------------------------------------------------------------
def dirichlet_wavenumbers(L: float, nmodes: int) -> np.ndarray:
    """``k_n = n pi / L``, ``n = 1 .. nmodes`` (spec eq. 3.2).

    ``n = 0`` is excluded because ``sin 0`` vanishes identically: there is no
    Dirichlet zero mode.
    """
    n = np.arange(1, nmodes + 1, dtype=float)
    return n * math.pi / L


def neumann_wavenumbers(L: float, nmodes: int) -> np.ndarray:
    """``k_n = n pi / L``, ``n = 0 .. nmodes - 1`` (spec eq. 3.3a).

    The ``n = 0`` entry is the constant zero mode ``f_0 = L^{-1/2}``.
    """
    n = np.arange(0, nmodes, dtype=float)
    return n * math.pi / L


def mixed_nd_wavenumbers(L: float, nmodes: int) -> np.ndarray:
    """``k_n = (n + 1/2) pi / L``, ``n = 0 .. nmodes - 1`` (spec eq. 3.6).

    The Neumann-at-``x=0``/Dirichlet-at-``x=L`` spectrum; the mirrored
    Dirichlet/Neumann interval is isospectral (spec eq. 3.8).  The lowest
    wavenumber is ``pi/(2L)``: a quarter wavelength fits in ``L``.
    """
    n = np.arange(0, nmodes, dtype=float)
    return (n + 0.5) * math.pi / L


# ---------------------------------------------------------------------------
# Robin: both spectral conditions
# ---------------------------------------------------------------------------
def _det_positive(k: float, kappa0: float, kappaL: float, L: float, scale: float) -> float:
    """Left-hand side of the positive-wavenumber condition (3.12), scaled.

    Handles the ``kappa -> infinity`` (Dirichlet) limits by dividing (3.12) by
    the divergent coupling, which is the correct limiting equation:

    * both ends Dirichlet:  ``sin(kL) = 0``
    * Dirichlet at ``x=0``, Robin at ``x=L``:  ``kappa_L sin(kL) + k cos(kL) = 0``
    * Robin at ``x=0``, Dirichlet at ``x=L``:  ``kappa_0 sin(kL) + k cos(kL) = 0``
    """
    x = k * L
    s = math.sin(x)
    c = math.cos(x)
    if math.isinf(kappa0) and math.isinf(kappaL):
        val = s
    elif math.isinf(kappa0):
        val = kappaL * s + k * c
    elif math.isinf(kappaL):
        val = kappa0 * s + k * c
    else:
        # SPEC-DISCREPANCY (spec 3.5.4 bullet 2): the spec states that
        # kappa_0 = 0 with finite kappa_L reduces (3.12) to cos(kL) = 0, "a
        # single Robin end already forces the half-integer set".  That is the
        # kappa_L -> infinity limit, not a finite coupling: with kappa_0 = 0
        # this determinant is -k^2 sin(kL) + kappa_L k cos(kL), i.e.
        # tan(kL) = kappa_L / k, whose roots sit between n*pi and (n+1/2)*pi.
        # This code implements tan(kL) = kappa_L/k.  Left as is pending the
        # theorist's adjudication.
        val = (kappa0 * kappaL - k * k) * s + (kappa0 + kappaL) * k * c
    return val / scale


def _det_negative(mu: float, kappa0: float, kappaL: float, L: float, scale: float) -> float:
    """Left-hand side of the imaginary-wavenumber condition (3.19), scaled.

    ``lambda = -mu^2``, ``mu > 0``.  Every term is positive for
    ``kappa_0, kappa_L >= 0``, so this equation has no root unless at least one
    coupling is negative -- exactly when (2.8) permits ``lambda < 0``.
    """
    y = mu * L
    s = math.sinh(y)
    c = math.cosh(y)
    if math.isinf(kappa0) and math.isinf(kappaL):
        val = s
    elif math.isinf(kappa0):
        val = kappaL * s + mu * c
    elif math.isinf(kappaL):
        val = kappa0 * s + mu * c
    else:
        val = (kappa0 * kappaL + mu * mu) * s + (kappa0 + kappaL) * mu * c
    return val / scale


def _has_zero_mode(kappa0: float, kappaL: float, L: float, tol: float = 1e-12) -> bool:
    """Criterion (3.18): ``k = 0`` is an eigenvalue iff this holds.

    ``kappa_0 + kappa_L + kappa_0 kappa_L L = 0``.  Note this is *not* solved by
    evaluating (3.12) at ``k = 0``: every term there vanishes identically, for
    every pair of couplings, so a root-finder that scans down to ``k = 0``
    reports a spurious zero mode whenever this criterion fails (spec section
    3.5.5 and item 2c of section 6).
    """
    if math.isinf(kappa0) or math.isinf(kappaL):
        # Dirichlet at either end: k=0 forces f = const * (a + b x), and the
        # Dirichlet end then forces f == 0.
        return False
    resid = kappa0 + kappaL + kappa0 * kappaL * L
    scale = abs(kappa0) + abs(kappaL) + abs(kappa0 * kappaL * L)
    return abs(resid) <= tol * max(scale, _TINY)


def _scan(
    f,
    lo: float,
    hi: float,
    step: float,
    *,
    detect_tangency: bool = False,
    tangency_tol: float = 1e-8,
) -> list[float]:
    """All sign changes of ``f`` on ``[lo, hi]``, refined by Brent's method.

    With ``detect_tangency`` the scan also looks for **near-tangent** pairs of
    roots: if an interval shows no sign change, ``|f|`` is minimised on it and,
    when the minimum is numerically zero, the interval is re-examined.  A sign
    change across the minimum means two roots (a narrow dip that the coarse grid
    stepped over); no sign change means a double root.

    This is needed for the negative-eigenvalue equation (3.19) with
    ``kappa < 0``: for ``|kappa| L >~ 3`` the two negative roots straddle
    ``mu = |kappa|`` and are separated by only ``~ |kappa| e^{-|kappa| L}`` in
    ``mu`` (e.g. ``kappa = -10``, ``L = 1``: roots at ``mu = 9.99909`` and
    ``10.00091``, a gap of ``1.8e-3``).  A plain sign-change scan with a step
    of ``pi/(8L)`` steps straight over the dip and reports **one** root instead
    of two -- see the note in ``docs/findings.md``.
    """
    n = max(int(math.ceil((hi - lo) / step)) + 1, 2)
    grid = np.linspace(lo, hi, n)
    vals = np.array([f(x) for x in grid])
    roots: list[float] = []
    for i in range(n - 1):
        a, b = float(grid[i]), float(grid[i + 1])
        fa, fb = float(vals[i]), float(vals[i + 1])
        if fa == 0.0:
            roots.append(a)
            continue
        if fa * fb < 0.0:
            roots.append(brentq(f, a, b, xtol=1e-14, rtol=8.9e-16, maxiter=200))
            continue
        if detect_tangency:
            res = minimize_scalar(
                lambda x: abs(f(x)),
                bounds=(a, b),
                method="bounded",
                options={"xatol": 1e-13 * (b - a)},
            )
            # only an *interior* minimum can be a genuine near-tangency: near
            # mu = 0 the determinant itself tends to zero (mu = 0 is always a
            # trivial root of 3.19), which would otherwise be reported as a
            # spurious double root at the left end of the first interval.
            width = b - a
            if not (a + 0.05 * width < res.x < b - 0.05 * width):
                continue
            fm = float(f(res.x))
            if abs(fm) < tangency_tol:
                if (fm < 0.0) != (fa < 0.0):
                    roots.append(
                        brentq(f, a, res.x, xtol=1e-14, rtol=8.9e-16, maxiter=200)
                    )
                    roots.append(
                        brentq(f, res.x, b, xtol=1e-14, rtol=8.9e-16, maxiter=200)
                    )
                else:
                    roots.append(float(res.x))  # double root
    if vals[-1] == 0.0:
        roots.append(float(grid[-1]))
    return roots


def robin_wavenumbers(
    kappa0: float,
    kappaL: float,
    L: float,
    nmodes: int,
    *,
    xmax: float | None = None,
) -> np.ndarray:
    """The ``nmodes`` smallest **positive** roots of (3.12), ascending.

    The scan step is ``pi/(8L)`` in ``k``, well below the ``pi/(4L)`` the spec
    requires (section 3.5.3): the determinant interlaces the ``tan`` poles, so
    one root per interval ``(n pi/L, (n+1) pi/L)`` is found by a sign change.
    The ``k = 0`` degeneracy is **not** reported here -- see
    :func:`robin_eigenvalues` and (3.18).

    Parameters
    ----------
    kappa0, kappaL:
        couplings at ``x = 0`` and ``x = L`` in ``dphi/dn + kappa phi = 0``.
        ``math.inf`` means Dirichlet at that end.
    """
    if nmodes <= 0:
        return np.zeros(0)
    step = math.pi / (8.0 * L)
    scale = 1.0
    if math.isfinite(kappa0) and math.isfinite(kappaL):
        scale = max(1.0, abs(kappa0 * kappaL))
    if math.isfinite(kappa0):
        scale += abs(kappa0) * (nmodes + 4) * math.pi / L
    if math.isfinite(kappaL):
        scale += abs(kappaL) * (nmodes + 4) * math.pi / L

    def f(k: float) -> float:
        return _det_positive(k, kappa0, kappaL, L, scale)

    # (3.12) vanishes *identically* at k = 0, so the scan cannot start there.
    # It also cannot start arbitrarily close: near k = 0 the determinant is a
    # difference of two terms of size ~ kappa_0 kappa_L k, so for couplings with
    # kappa_0 + kappa_L + kappa_0 kappa_L L = 0 (the zero-mode case, 3.18) the
    # leading term cancels and the value is O(k^3), below round-off for
    # k < sqrt(eps) ~ 1.5e-8.  Start at 1e-6/L and discard roots below 2e-6/L:
    # the smallest *genuine* positive root is ~ sqrt(2 kappa / L), which is
    # above 1e-5 for kappa > 1e-10.
    k_start = 1e-6 / L
    k_floor = 2.0 * k_start

    hi = xmax if xmax is not None else (nmodes + 2) * math.pi / L
    roots: list[float] = []
    while len(roots) < nmodes:
        found = _scan(f, k_start, hi, step)
        roots = [r for r in found if r > k_floor]
        if len(roots) >= nmodes or xmax is not None:
            break
        hi *= 2.0
    return np.array(sorted(roots)[:nmodes])


def robin_eigenvalues(
    kappa0: float, kappaL: float, L: float, nmodes: int
) -> np.ndarray:
    """The ``nmodes`` lowest eigenvalues of ``-d^2/dx^2`` with Robin conditions.

    Complete in the sense the spec demands (section 3.5.6): the positive roots
    of (3.12), the negative roots of (3.19) when a coupling is negative, and the
    zero mode of (3.18) are merged and sorted ascending.  ``lambda_min`` can be
    negative and must not be "fixed".

    SPEC-DISCREPANCY (spec 3.5.6 item 2): the spec says the number of negative
    eigenvalues is one "for the couplings I checked" (kappa = -0.5, -2, -3, -10
    at both ends).  That holds for kappa = -0.5 and -2 but not for -3 and -10,
    which have *two* each: the two roots straddle mu = |kappa| and are only
    ~|kappa| exp(-|kappa| L) apart, so a sign-change scan steps over the dip.
    This function returns both.  Left as is pending the theorist's
    adjudication.
    """
    if nmodes <= 0:
        return np.zeros(0)

    lambdas: list[float] = []

    # --- lambda = 0 (3.18), handled once, outside the scan ------------------
    if _has_zero_mode(kappa0, kappaL, L):
        lambdas.append(0.0)

    # --- lambda = k^2 > 0 (3.12) -------------------------------------------
    ks = robin_wavenumbers(kappa0, kappaL, L, nmodes + 2)
    lambdas.extend(float(k * k) for k in ks)

    # --- lambda = -mu^2 < 0 (3.19) -----------------------------------------
    if (math.isfinite(kappa0) and kappa0 < 0.0) or (
        math.isfinite(kappaL) and kappaL < 0.0
    ):
        scale = 1.0
        if math.isfinite(kappa0) and math.isfinite(kappaL):
            scale = max(1.0, abs(kappa0 * kappaL))
        if math.isfinite(kappa0):
            scale += abs(kappa0) * 20.0
        if math.isfinite(kappaL):
            scale += abs(kappaL) * 20.0
        # mu large enough: |kappa| is the asymptotic scale of the root
        mu_hi = 2.0 * max(abs(kappa0) if math.isfinite(kappa0) else 0.0,
                          abs(kappaL) if math.isfinite(kappaL) else 0.0,
                          1.0 / L) + 40.0 / L

        def f(mu: float) -> float:
            return _det_negative(mu, kappa0, kappaL, L, scale)

        for mu in _scan(
            f,
            1e-6 / L,
            mu_hi,
            math.pi / (8.0 * L),
            detect_tangency=True,
        ):
            if mu > 0.0:
                lambdas.append(-float(mu * mu))

    out = np.array(sorted(lambdas), dtype=float)
    return out[:nmodes]


def wavenumbers(
    left: BoundaryCondition | str | float,
    right: BoundaryCondition | str | float,
    L: float,
    nmodes: int,
) -> np.ndarray:
    """Wavenumbers ``k_n > 0`` for the interval with these two conditions.

    Dispatches to the closed forms of section 3; anything involving a finite
    Robin coupling goes through :func:`robin_wavenumbers`.
    """
    b0, b1 = as_bc(left), as_bc(right)
    if b0.is_dirichlet and b1.is_dirichlet:
        return dirichlet_wavenumbers(L, nmodes)
    if b0.is_neumann and b1.is_neumann:
        return neumann_wavenumbers(L, nmodes)
    if (b0.is_neumann and b1.is_dirichlet) or (b0.is_dirichlet and b1.is_neumann):
        return mixed_nd_wavenumbers(L, nmodes)
    return robin_wavenumbers(b0.kappa, b1.kappa, L, nmodes)


def eigenvalues(
    left: BoundaryCondition | str | float,
    right: BoundaryCondition | str | float,
    L: float,
    nmodes: int,
) -> np.ndarray:
    """The ``nmodes`` lowest eigenvalues of ``-d^2/dx^2`` for any BC pair.

    For D, N and mixed N/D this is the closed form; for Robin it is
    :func:`robin_eigenvalues`, which includes negative eigenvalues.
    """
    b0, b1 = as_bc(left), as_bc(right)
    if b0.is_robin or b1.is_robin:
        return robin_eigenvalues(b0.kappa, b1.kappa, L, nmodes)
    k = wavenumbers(b0, b1, L, nmodes)
    return np.sort(k * k)


# ---------------------------------------------------------------------------
# Eigenfunctions, unit L2 norm (1.8)
# ---------------------------------------------------------------------------
def dirichlet_mode(L: float, n: int, x: np.ndarray) -> np.ndarray:
    """``sqrt(2/L) sin(n pi x / L)`` (spec eq. 3.2), ``n >= 1``."""
    if n < 1:
        raise ValueError("Dirichlet mode numbers start at n = 1")
    return np.sqrt(2.0 / L) * np.sin(n * math.pi * np.asarray(x) / L)


def neumann_mode(L: float, n: int, x: np.ndarray) -> np.ndarray:
    """Neumann modes (spec eq. 3.3b).

    ``n = 0`` is the constant ``1/sqrt(L)`` -- note ``sqrt(L)``, not
    ``sqrt(2/L)``: the zero mode is the only one whose norm is ``L`` rather
    than ``L/2`` before normalisation.
    """
    x = np.asarray(x)
    if n < 0:
        raise ValueError("Neumann mode numbers start at n = 0")
    if n == 0:
        return np.full_like(x, 1.0 / math.sqrt(L), dtype=float)
    return np.sqrt(2.0 / L) * np.cos(n * math.pi * x / L)


def mixed_mode(L: float, n: int, x: np.ndarray, *, neumann_at_left: bool = True) -> np.ndarray:
    """Mixed N/D modes ``sqrt(2/L) cos((n+1/2) pi x / L)``, ``n >= 0`` (3.7).

    With ``neumann_at_left=False`` the mirrored (Dirichlet at ``x=0``) case is
    returned, ``sqrt(2/L) sin((n+1/2) pi x / L)`` (spec eq. 3.8).  The two are
    isospectral.  No special case is needed for ``n = 0``: ``cos(k_n L) = 0``
    makes the normalisation integral exactly ``L/2``.
    """
    if n < 0:
        raise ValueError("mixed N/D mode numbers start at n = 0")
    k = (n + 0.5) * math.pi / L
    x = np.asarray(x)
    if neumann_at_left:
        return np.sqrt(2.0 / L) * np.cos(k * x)
    return np.sqrt(2.0 / L) * np.sin(k * x)


def robin_normalisation(k: float, kappa0: float, L: float) -> float:
    """``I(k) = int_0^L (cos kx + c sin kx)^2 dx`` with ``c = kappa_0/k`` (3.17).

    The eigenfunction is ``f = I^{-1/2} (cos kx + c sin kx)``, which satisfies
    the condition at ``x = 0`` identically.  ``I`` is *not* ``L/2`` except in the
    Neumann/Dirichlet limits -- e.g. for ``kappa = 2``, ``L = 1``:
    ``I(k_1) = 1.851033886878379`` (spec section 3.5.2).

    The ``k -> 0`` limit is handled explicitly: the bracket tends to
    ``1 + kappa_0 x`` and ``I(0) = L + kappa_0 L^2 + kappa_0^2 L^3 / 3``.
    """
    if not math.isfinite(kappa0):
        # Dirichlet at x = 0: f = sqrt(2/L) sin(kx), I = L/2 for cos(kL) = 0
        return 0.5 * L
    if abs(k) * L < 1e-8:
        return L + kappa0 * L**2 + kappa0**2 * L**3 / 3.0
    c = kappa0 / k
    x = k * L
    return (
        0.5 * L * (1.0 + c * c)
        + (1.0 - c * c) / (4.0 * k) * math.sin(2.0 * x)
        + c / k * math.sin(x) ** 2
    )


def robin_mode(k: float, kappa0: float, L: float, x: np.ndarray) -> np.ndarray:
    """Normalised Robin eigenfunction ``I(k)^{-1/2} (cos kx + (kappa_0/k) sin kx)``.

    Satisfies ``-phi'(0) + kappa_0 phi(0) = 0`` and ``int_0^L f^2 = 1``.
    """
    x = np.asarray(x)
    c = 0.0 if (not math.isfinite(kappa0) or k == 0.0) else kappa0 / k
    if math.isfinite(kappa0):
        bracket = np.cos(k * x) + c * np.sin(k * x)
    else:
        bracket = np.sin(k * x)
    return bracket / math.sqrt(robin_normalisation(k, kappa0, L))


# ---------------------------------------------------------------------------
# Spectra as containers
# ---------------------------------------------------------------------------
@dataclass
class AnalyticSpectrum:
    """An exact spectrum from the closed forms of section 3.

    Attributes
    ----------
    lam:
        eigenvalues of ``-Delta``, ascending.
    omega2:
        ``lambda + m^2`` (1.6).
    omega:
        ``sqrt(lambda + m^2)`` as a **complex** array: for ``lambda + m^2 < 0``
        this is ``+i sqrt(-(lambda+m^2))``, an exponentially growing mode
        (1.4 item 2).  Never ``nan``, never ``abs``.
    labels:
        per level, the mode labels: ``(n,)`` in 1D, ``(n1, n2)`` in 2D.
    m:
        the mass used for ``omega``.
    """

    lam: np.ndarray
    omega2: np.ndarray
    omega: np.ndarray
    labels: list = field(default_factory=list)
    m: float = 0.0
    bc: tuple = ()
    L: tuple = ()

    def __len__(self) -> int:
        return len(self.lam)

    def __repr__(self) -> str:
        return (
            f"AnalyticSpectrum(nmodes={len(self.lam)}, m={self.m!r}, "
            f"lam[0]={self.lam[0] if len(self.lam) else None!r})"
        )


def _finish(lam: Sequence[float], labels: list, m: float, bc=(), L=()) -> AnalyticSpectrum:
    lam_arr = np.asarray(lam, dtype=float)
    order = np.argsort(lam_arr, kind="stable")
    lam_arr = lam_arr[order]
    if labels is not None and len(labels) == len(lam_arr):
        labels = [labels[i] for i in order]
    omega2 = lam_arr + m * m
    return AnalyticSpectrum(
        lam=lam_arr,
        omega2=omega2,
        omega=omega_from_lambda(lam_arr, m),
        labels=labels if labels is not None else [],
        m=m,
        bc=bc,
        L=L,
    )


def _per_direction_modes(
    left, right, L: float, nmodes: int
) -> list[tuple[tuple, float]]:
    """``[(label, k), ...]`` for one direction, ascending in ``k``."""
    b0, b1 = as_bc(left), as_bc(right)
    if b0.is_dirichlet and b1.is_dirichlet:
        return [((n,), k) for n, k in enumerate(dirichlet_wavenumbers(L, nmodes), start=1)]
    if b0.is_neumann and b1.is_neumann:
        return [((n,), k) for n, k in enumerate(neumann_wavenumbers(L, nmodes))]
    if (b0.is_neumann and b1.is_dirichlet) or (b0.is_dirichlet and b1.is_neumann):
        return [
            ((n,), k) for n, k in enumerate(mixed_nd_wavenumbers(L, nmodes))
        ]
    ks = robin_wavenumbers(b0.kappa, b1.kappa, L, nmodes)
    return [((n,), float(k)) for n, k in enumerate(ks, start=1)]


def spectrum_1d(
    left: BoundaryCondition | str | float,
    right: BoundaryCondition | str | float,
    L: float,
    nmodes: int,
    m: float = 0.0,
) -> AnalyticSpectrum:
    """Exact 1D spectrum for an arbitrary pair of boundary conditions."""
    b0, b1 = as_bc(left), as_bc(right)
    lam = eigenvalues(b0, b1, L, nmodes)
    labels = [(i + 1,) for i in range(len(lam))]
    return _finish(lam, labels, m, bc=(b0, b1), L=(L,))


def spectrum_2d(
    x_left: BoundaryCondition | str | float,
    x_right: BoundaryCondition | str | float,
    L1: float,
    y_left: BoundaryCondition | str | float,
    y_right: BoundaryCondition | str | float,
    L2: float,
    nmodes: int,
    m: float = 0.0,
) -> AnalyticSpectrum:
    """Exact 2D product spectrum (3.20) with per-direction conditions.

    Levels are ordered by eigenvalue, **not** lexicographically by
    ``(n1, n2)`` (spec section 3.6.1), and degenerate levels are kept with their
    full multiplicity (spec section 6 item 9).  Ties are broken by the label
    ``(n1, n2)`` in lexicographic order.
    """
    n_per = nmodes + 4
    modes1 = _per_direction_modes(x_left, x_right, L1, n_per)
    modes2 = _per_direction_modes(y_left, y_right, L2, n_per)

    levels = []
    for (lab1, k1) in modes1:
        for (lab2, k2) in modes2:
            levels.append((k1 * k1 + k2 * k2, (lab1[0], lab2[0])))
    levels.sort(key=lambda t: (t[0], t[1]))
    levels = levels[:nmodes]

    return _finish(
        [t[0] for t in levels],
        [t[1] for t in levels],
        m,
        bc=(as_bc(x_left), as_bc(x_right), as_bc(y_left), as_bc(y_right)),
        L=(L1, L2),
    )
