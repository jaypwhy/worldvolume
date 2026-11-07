"""The Neumann zero mode and the absence of spurious ones (spec 3.3, 6 item 2).

The spec calls the ``n = 0`` row of Case B "the single most diagnostic row in the
document": the discrete eigenvalue must come out as ``0`` to machine precision,
the eigenvector must be the constant vector, and ``omega_0^2 = m^2`` exactly.
A wrong Neumann boundary row gives ``lambda_0 = O(h^2)`` instead of
``O(1e-14)``.

Note on "exactly": ``lambda_0 = 0`` is an exact eigenvalue of the assembled
matrix -- the constant vector is annihilated by it bitwise, which is asserted
below -- but ``scipy.linalg.eigh`` returns it as ``~1e-14`` because of round-off
in the tridiagonal reduction.  That is the sense in which the spec's own quoted
``lambda_0 = 5.8e-14`` is "exactly zero": pure round-off, four orders of
magnitude below the ``O(h^2)`` value a wrong Neumann row would give.

The converse trap is just as important: ``k = 0`` satisfies the Robin
determinant (3.12) *identically, for every pair of couplings*, so a root finder
that scans down to ``k = 0`` reports a **spurious** zero mode whenever the
criterion (3.18) fails.  Both directions are tested here.
"""

import math

import numpy as np
import pytest

from worldvolume import (
    Box2D,
    Dirichlet,
    Grid1D,
    Neumann,
    Robin,
    laplacian_1d,
    neumann_mode,
    robin_eigenvalues,
    solve,
    spectrum_1d,
)


@pytest.mark.parametrize("N", [4, 20, 64])
@pytest.mark.parametrize("L", [1.0, 2.0])
@pytest.mark.parametrize("m", [0.0, 1.0, 0.5])
def test_neumann_zero_mode_is_exactly_zero(N, L, m):
    g = Grid1D(L=L, N=N, left=Neumann(), right=Neumann())
    spec = solve(g, m=m, nmodes=3)
    # lambda_0 = 0 to machine precision, not O(h^2) (which would be ~1e-2 here)
    assert abs(spec.lam[0]) < 1e-12, f"lam_0 = {spec.lam[0]:.3e} at N={N}"
    # the assembled operator annihilates the constant vector *exactly*, so
    # lambda = 0 is an exact eigenvalue of the discrete problem
    a, _ = laplacian_1d(g)
    ones = np.ones(g.n_active)
    assert np.max(np.abs(a @ ones)) == 0.0
    assert float(ones @ (a @ ones)) == 0.0
    # omega_0^2 = m^2 to the same round-off, and omega_0 = |m|
    assert abs(spec.omega2[0] - m * m) < 1e-12 * max(1.0, m * m)
    # omega_0 = sqrt(m^2 + lambda_0), and lambda_0 is only known to round-off
    # (~1e-13 here).  The square root *amplifies* that: |d omega_0| is
    # |lambda_0|/(2|m|) for m != 0 and sqrt(|lambda_0|) for m = 0 (where the
    # value is ~1e-7, not 0).  So the tolerance has to be derived from the
    # actual lambda_0, not hard-coded below what the round-off can deliver.
    domega = (
        math.sqrt(abs(spec.lam[0])) if m == 0.0 else abs(spec.lam[0]) / (2.0 * abs(m))
    )
    assert spec.omega[0] == pytest.approx(complex(abs(m)), abs=domega + 1e-14)
    # the eigenvector is the constant vector, and u = 1/sqrt(L)
    v = spec.v[:, 0]
    assert np.max(np.abs(v - v[0])) < 1e-13
    u = spec.mode(0)
    if u[0] < 0:
        u = -u
    assert np.allclose(u, 1.0 / math.sqrt(L), rtol=1e-13, atol=1e-15)
    # W-orthonormal: sum_j w_j v_j^2 = 1 with v constant means |v_j| = 1/sqrt(sum w).
    # The *sign* of an eigenvector is arbitrary (eigh returns either), so the
    # assertion is on the magnitude -- asserting v[0] itself tests which sign
    # LAPACK happened to pick, which is not physics.
    assert abs(v[0]) == pytest.approx(1.0 / math.sqrt(np.sum(g.weights)), rel=1e-14)


def test_neumann_zero_mode_normalisation_is_not_sqrt2_over_L():
    # spec 3.3: the zero mode carries 1/sqrt(L), the non-zero modes sqrt(2/L).
    # Applying sqrt(2/L) uniformly gives a norm of 2, off by sqrt(2).
    L, N = 2.0, 32
    g = Grid1D(L=L, N=N, left=Neumann(), right=Neumann())
    spec = solve(g, nmodes=2)
    x = g.x
    zero = spec.mode(0)
    exact = neumann_mode(L, 0, x)
    if zero @ exact < 0:
        exact = -exact
    assert np.max(np.abs(zero - exact)) < 1e-13
    # L2 norm by the trapezoidal rule is 1 for the zero mode and for mode 1
    for i in range(2):
        u = spec.mode(i)
        norm2 = g.h * np.sum(g.weights * u[g.active] ** 2)
        assert norm2 == pytest.approx(1.0, rel=1e-12)


def test_neumann_zero_mode_in_2d():
    box = Box2D.from_lengths(1.0, 8, Neumann(), 1.5, 8, Neumann())
    spec = solve(box, m=0.3, nmodes=2)
    assert abs(spec.lam[0]) < 1e-12
    assert spec.omega2[0] == pytest.approx(0.09, rel=1e-12)
    grid = spec.mode(0)
    if grid[0, 0] < 0:
        grid = -grid
    assert np.max(np.abs(grid - grid[0, 0])) < 1e-13
    assert grid[0, 0] == pytest.approx(1.0 / math.sqrt(1.0 * 1.5), rel=1e-13)


def test_mixed_nd_has_no_zero_mode():
    # spec 2.5: for the mixed N_x D_y box the ground state is constant in x and
    # a half-wave in y, with lambda_01 = (pi/L_2)^2 in the continuum.  There is
    # no zero mode.  The *discrete* value is (4.14) for the y half-wave.
    box = Box2D.from_lengths(1.0, 8, Neumann(), 1.5, 8, Dirichlet())
    spec = solve(box, nmodes=2)
    h2 = box.y.h
    # a *full* half-wave in y: Dirichlet at both ends of [0, L_2] means
    # k_1 = pi/L_2.  (pi/(2L_2) would be the quarter-wave of a single
    # Dirichlet end, which is not this problem.)
    k2 = math.pi / 1.5
    discrete = 4.0 / h2**2 * math.sin(k2 * h2 / 2) ** 2
    assert spec.lam[0] == pytest.approx(discrete, rel=1e-12)
    # ... and it is within O(h^2) of the continuum (pi/L_2)^2
    assert spec.lam[0] == pytest.approx((math.pi / 1.5) ** 2, rel=2e-2)
    assert spec.lam[0] > 0.0
    grid = spec.mode(0)
    # constant along x, single half-wave along y
    assert np.allclose(grid, grid[:1, :], atol=1e-12)
    col = grid[0, :]
    assert np.all(col[1:-1] > 0) or np.all(col[1:-1] < 0)


@pytest.mark.parametrize(
    "kappa0,kappaL,has_zero",
    [
        # (3.18): k = 0 is an eigenvalue iff kappa_0 + kappa_L + kappa_0 kappa_L L = 0
        (0.0, 0.0, True),
        (-2.0, -2.0, True),
        (-3.0, -1.5, True),
        (0.0, 0.5, False),
        (0.5, 0.0, False),
        (2.0, 2.0, False),
        (-2.0, 0.0, False),
        (0.0, -2.0, False),
        (-1.0, -2.0, False),
        (-2.0, -1.0, False),
    ],
)
def test_zero_mode_criterion_3_18(kappa0, kappaL, has_zero):
    # L = 1, so the criterion is kappa_0 + kappa_L + kappa_0 kappa_L = 0
    lam = robin_eigenvalues(kappa0, kappaL, 1.0, 4)
    got_zero = bool(np.any(np.abs(lam) < 1e-12))
    assert got_zero is has_zero, f"({kappa0}, {kappaL}) -> {lam}"
    if not has_zero:
        # and no *spurious* zero mode sneaks in from evaluating (3.12) at k=0
        assert np.min(np.abs(lam)) > 1e-6


@pytest.mark.parametrize("kappa", [0.001, 0.01, 0.1, 1.0, 2.0, 10.0])
def test_positive_robin_coupling_has_strictly_positive_spectrum(kappa):
    # spec 6 item 1: for all kappa > 0, lambda_min > 0.  (A negative eigenvalue
    # here is the signature of a flipped normal-derivative sign.)
    lam = robin_eigenvalues(kappa, kappa, 1.0, 3)
    assert lam[0] > 0.0
    g = Grid1D(L=1.0, N=200, left=Robin(kappa), right=Robin(kappa))
    spec = solve(g, nmodes=1)
    assert spec.lam[0] > 0.0


def test_case_D_has_no_zero_mode():
    # spec 6 item 2: Case D (kappa = 2) must not return a zero eigenvalue
    spec = spectrum_1d(Robin(2.0), Robin(2.0), 1.0, 3)
    assert spec.lam[0] > 1.0
    g = Grid1D(L=1.0, N=64, left=Robin(2.0), right=Robin(2.0))
    assert solve(g, nmodes=1).lam[0] > 1.0
