"""Second-order convergence, and the exact coefficient of the leading error.

Spec 4.7.2: doubling ``N`` (halving ``h``) must reduce the eigenvalue error by a
factor 4 asymptotically, for every boundary-condition combination including
Robin.  A measured rate of 1 means a one-sided boundary difference or a wrong
``h``; a rate of 4 or better means the comparison is against the discrete
spectrum rather than the continuum one.

Spec 4.7.1 (4.15): for **Dirichlet and Neumann** data the leading relative error
is ``-(k_n h)^2/12`` with no free constant, and the ratio of the measured error
to that prediction is 1 to within a few parts in a thousand.  For **Robin** the
coefficient is not 1/12 (it is ``kappa``- and mode-dependent and can have either
sign) -- the rate is still 2, which is what to test there.
"""

import math

import numpy as np
import pytest

from conftest import rate
from worldvolume import (
    Box2D,
    Dirichlet,
    Grid1D,
    Neumann,
    Robin,
    robin_eigenvalues,
    solve,
    spectrum_2d,
    wavenumbers,
)

GRIDS = [10, 20, 40, 80, 160]


def _eigenvalue_error(bc0, bc1, L, mode_index, N):
    """Absolute error of the ``mode_index``-th discrete eigenvalue (0-based)."""
    # for Neumann at both ends the lowest mode is the exact zero mode, so shift
    # to the first non-trivial one
    if bc0.is_neumann and bc1.is_neumann:
        mode_index += 1
    g = Grid1D(L=L, N=N, left=bc0, right=bc1)
    spec = solve(g, nmodes=mode_index + 1)
    if bc0.is_robin or bc1.is_robin:
        exact = robin_eigenvalues(bc0.kappa, bc1.kappa, L, mode_index + 1)[mode_index]
    else:
        exact = wavenumbers(bc0, bc1, L, mode_index + 1)[mode_index] ** 2
    return abs(spec.lam[mode_index] - exact)


@pytest.mark.parametrize(
    "bc0,bc1,name",
    [
        (Dirichlet(), Dirichlet(), "D-D"),
        (Neumann(), Neumann(), "N-N"),
        (Neumann(), Dirichlet(), "N-D"),
        (Robin(2.0), Robin(2.0), "R2-R2"),
        (Robin(0.5), Robin(3.0), "R0.5-R3"),
        (Robin(2.0), Neumann(), "R-N"),
    ],
)
@pytest.mark.parametrize("mode_index", [0, 1, 2])
def test_second_order_convergence(bc0, bc1, name, mode_index):
    errors = [_eigenvalue_error(bc0, bc1, 1.0, mode_index, N) for N in GRIDS]
    rates = rate(errors)
    # every successive halving must improve on the previous rate, approaching 2
    assert all(r > 1.8 for r in rates), f"{name} mode {mode_index}: rates {rates}"
    assert rates[-1] == pytest.approx(2.0, abs=0.03), f"{name}: rates {rates}"
    assert all(rates[i + 1] > rates[i] - 1e-9 for i in range(len(rates) - 1))


@pytest.mark.parametrize(
    "bc0,bc1,name",
    [
        (Dirichlet(), Dirichlet(), "D-D"),
        (Neumann(), Neumann(), "N-N"),
        (Neumann(), Dirichlet(), "N-D"),
    ],
)
@pytest.mark.parametrize("N,mode_index", [(40, 0), (80, 2)])
def test_leading_error_coefficient_is_minus_one_twelfth(bc0, bc1, name, N, mode_index):
    # spec (4.15): lambda_n^(h) = lambda_n [1 - (k_n h)^2/12 + O(h^4)]
    if bc0.is_neumann and bc1.is_neumann:
        mode_index += 1  # skip the exact zero mode
    g = Grid1D(L=1.0, N=N, left=bc0, right=bc1)
    spec = solve(g, nmodes=mode_index + 1)
    k = wavenumbers(bc0, bc1, 1.0, mode_index + 1)[mode_index]
    exact = k**2
    measured = (spec.lam[mode_index] - exact) / exact
    predicted = -(k * g.h) ** 2 / 12.0
    ratio = measured / predicted
    assert ratio == pytest.approx(1.0, abs=0.01), (
        f"{name} N={N} mode {mode_index}: rel err {measured:+.4e}, "
        f"-(kh)^2/12 = {predicted:+.4e}, ratio {ratio:.4f}"
    )


@pytest.mark.parametrize("mode_index", [0, 2])
def test_robin_error_coefficient_is_not_one_twelfth(mode_index):
    # spec 4.7.1: "Do not assert (4.14) or (4.15) for Robin.  The reason: the
    # ghost-node Robin condition contributes a boundary truncation error that
    # enters the eigenvalue at the same O(h^2) order with its own
    # kappa-dependent coefficient."
    g = Grid1D(L=1.0, N=40, left=Robin(2.0), right=Robin(2.0))
    spec = solve(g, nmodes=mode_index + 1)
    exact = robin_eigenvalues(2.0, 2.0, 1.0, mode_index + 1)[mode_index]
    k = math.sqrt(exact)
    measured = (spec.lam[mode_index] - exact) / exact
    predicted = -(k * g.h) ** 2 / 12.0
    ratio = measured / predicted
    assert abs(ratio - 1.0) > 0.2, (
        f"Robin mode {mode_index}: ratio {ratio:.4f} looks like the D/N "
        f"coefficient -- the Robin boundary row is probably wrong"
    )
    # the spec's measured values at N=40: mode 1 -> -0.46, mode 3 -> magnitude 0.71
    # SPEC-DISCREPANCY (spec 4.7.1, the Robin "ratio" column): the spec prints
    # the mode-1 relative error as +6.75e-5 and then calls it "-0.46 times" the
    # naive -(kh)^2/12, which is itself negative -- the two signs contradict
    # each other, and "mode 3 -> -0.71 times" has the opposite one again.  The
    # ratio is therefore asserted on its magnitude here.  Left as is pending
    # the theorist's adjudication.
    assert abs(ratio) == pytest.approx(0.46 if mode_index == 0 else 0.71, abs=0.02)


def test_2d_mixed_convergence_is_second_order():
    errors = []
    for N in (16, 32, 64):
        box = Box2D.from_lengths(1.0, N, Neumann(), 1.5, N, Dirichlet())
        spec = solve(box, nmodes=6)
        exact = spectrum_2d(
            Neumann(), Neumann(), 1.0, Dirichlet(), Dirichlet(), 1.5, 6
        ).lam
        errors.append(np.max(np.abs(spec.lam - exact)))
    rates = rate(errors)
    assert rates[-1] == pytest.approx(2.0, abs=0.08), f"2D rates {rates}"
    assert all(r > 1.8 for r in rates)


def test_robin_converges_to_the_dirichlet_spectrum_as_kappa_grows():
    # spec 3.5.4: with kappa = 1e6 at both ends the discrete spectrum reproduces
    # the Dirichlet convergence table exactly to the printed digits
    for N in (20, 40):
        g_robin = Grid1D(L=1.0, N=N, left=Robin(1e6), right=Robin(1e6))
        g_dir = Grid1D(L=1.0, N=N, left=Dirichlet(), right=Dirichlet())
        a = solve(g_robin, nmodes=4).lam
        b = solve(g_dir, nmodes=4).lam
        assert np.allclose(a, b, rtol=1e-4), f"N={N}: {a} vs {b}"
