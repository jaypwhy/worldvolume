"""Robin spectra: limits, negative couplings, small-kappa regime (spec 3.5).

The limits of 3.5.4 pin the sign convention: ``kappa = 0`` must reproduce the
Neumann spectrum exactly, ``kappa -> infinity`` the Dirichlet one, and a single
Robin end with the other Neumann must give the half-integer set (3.6).

Negative couplings are the case where the spectrum stops being positive, and
where **both** spectral conditions are needed: (3.12) for ``lambda = k^2 > 0``
and (3.19) for ``lambda = -mu^2 < 0`` (spec 3.5.6).  The reference values below
were computed independently with mpmath at 30-40 digit precision by a fine scan
of (3.19) and (3.12); they are not taken from the solver.

**Discrepancy with the spec, flagged rather than papered over** (see
``docs/findings.md``): spec 3.5.6 item 2 states that "the number of negative
eigenvalues is one for the couplings I checked (kappa = -0.5, -2, -3, -10 at
both ends, L=1)".  That is true for ``kappa = -0.5`` and ``kappa = -2`` but
**false for kappa = -3 and kappa = -10**, which have *two* negative eigenvalues
each.  For ``|kappa| L >~ 3`` the two roots straddle ``mu = |kappa|`` and are
separated by only ``~ |kappa| e^{-|kappa| L}``; a sign-change scan with a step
of ``pi/(8L)`` steps over the dip and reports one.  The spec's quoted
``lambda_min = -100.003`` for ``kappa = -10`` is the average of the two true
roots ``-100.0181`` and ``-99.9818`` (their N=400 discrete values, where the
near-degenerate pair is not resolved either).  The tests below assert the
two-root answer, and also assert that the finite-difference solver reproduces
it -- two independent methods agreeing is what makes this a finding rather than
a bug.
"""

import math

import numpy as np
import pytest

from conftest import assert_sig_figs
from worldvolume import (
    Dirichlet,
    Grid1D,
    Neumann,
    Robin,
    robin_eigenvalues,
    robin_mode,
    robin_normalisation,
    robin_wavenumbers,
    solve,
)


# ---------------------------------------------------------------------------
# limits (3.5.4)
# ---------------------------------------------------------------------------
def test_kappa_to_infinity_gives_dirichlet():
    # with kappa = 1e8 the roots sit below n pi by the (2.6) amount, so they
    # agree with n pi to 7-8 digits, not to machine precision
    k = robin_wavenumbers(1e8, 1e8, 1.0, 4)
    n = np.arange(1, 5)
    assert np.allclose(k, n * math.pi, rtol=1e-7)
    assert np.allclose(k, n * math.pi * (1 - 2e-8), rtol=1e-8)


def test_kappa_to_zero_gives_neumann():
    # at kappa = 0 exactly, (3.12) reduces to -k^2 sin(kL) = 0, whose *positive*
    # roots are n pi (robin_wavenumbers returns positive roots only; the k = 0
    # Neumann zero mode is added by robin_eigenvalues via (3.18))
    k = robin_wavenumbers(0.0, 0.0, 1.0, 4)
    assert np.allclose(k, np.arange(1, 5) * math.pi, rtol=1e-12)
    # and Robin(0.0) is Neumann, including the zero mode
    lam = robin_eigenvalues(0.0, 0.0, 1.0, 4)
    assert lam[0] == 0.0
    assert np.allclose(lam[1:], (np.arange(1, 4) * math.pi) ** 2, rtol=1e-12)


def test_neumann_end_with_a_finite_robin_end_gives_tan_kL_equals_kappa_over_k():
    # NOTE (discrepancy with spec 3.5.4, see docs/findings.md): the spec says
    # that kappa_0 = 0 with finite kappa_L reduces (3.12) to cos(kL) = 0, i.e.
    # to the half-integer set.  That is wrong algebra: with kappa_0 = 0 the
    # determinant is -k^2 sin(kL) + kappa_L k cos(kL), i.e.
    #     tan(kL) = kappa_L / k,
    # whose roots sit near n pi.  cos(kL) = 0 is the *kappa_L -> infinity* limit
    # (the spec's own next bullet).  The roots below are therefore near n pi,
    # not near (n + 1/2) pi, and they move to the half-integer set only as
    # kappa_L grows.
    for kappaL in (0.5, 3.0):
        k = robin_wavenumbers(0.0, kappaL, 1.0, 3)
        for kk in k:
            assert math.tan(kk) == pytest.approx(kappaL / kk, rel=1e-9)
        # Each root lies strictly between the Neumann value n pi (kappa_L = 0)
        # and the Dirichlet value (n + 1/2) pi (kappa_L -> infinity): that is
        # the precise sense in which they are "near n pi" and move towards the
        # half-integer set as kappa_L grows.  (An absolute atol=1.0 against
        # n pi is wrong for the n = 0 root, which sits at 1.19 for kappa_L = 3
        # -- already 1.19 away from 0.)
        n = np.arange(3)
        assert np.all(k > n * math.pi)
        assert np.all(k < (n + 0.5) * math.pi)
    # the limit kappa_L -> infinity does give the half-integer set
    k = robin_wavenumbers(0.0, 1e8, 1.0, 4)
    assert np.allclose(k, (np.arange(0, 4) + 0.5) * math.pi, rtol=1e-7)


def test_approach_to_dirichlet_is_from_below_at_rate_2_over_kappa_L():
    # spec eq. (2.6): k_1 = (pi/L)(1 - 2/(kappa L)) + O(kappa^-2)
    L = 1.0
    kappa = 1e6
    k1 = robin_wavenumbers(kappa, kappa, L, 1)[0]
    predicted = math.pi * (1 - 2.0 / (kappa * L))
    assert abs(k1 - predicted) < 1e-10


def test_dirichlet_robin_is_isospectral_to_robin_dirichlet():
    # the mirror x -> L - x swaps the two ends and cannot change the spectrum
    a = robin_eigenvalues(math.inf, 2.0, 1.0, 5)
    b = robin_eigenvalues(2.0, math.inf, 1.0, 5)
    assert np.allclose(a, b, rtol=1e-12)
    # D-R is NOT the same as D-N: check the values independently, from
    # tan(kL) = -k/kappa_L
    k = robin_wavenumbers(math.inf, 2.0, 1.0, 1)[0]
    assert math.tan(k) == pytest.approx(-k / 2.0, rel=1e-10)


# ---------------------------------------------------------------------------
# small kappa: the delicate regime (3.5.5)
# ---------------------------------------------------------------------------
SMALL_KAPPA_TABLE = [
    # kappa, k_1, lambda_1, 2 kappa / L
    # NOTE: the spec's table (3.5.5) prints lambda_1 = 0.0020000 for
    # kappa = 1e-3, which is 2 kappa/L (the asymptote, to 5 s.f.) and not the
    # eigenvalue: the spec's own k_1 column gives k_1^2 = 0.0447176^2 =
    # 0.0019997.  The other six rows are self-consistent (k_1^2 = lambda_1);
    # only that one is a rounding slip, so the entry below is the value
    # consistent with the spec's own k_1.
    (1e-3, "0.0447176", "0.0019997"),
    (1e-2, "0.1413036", "0.0199667"),
    (0.1, "0.4435208", "0.196711"),
    (1.0, "1.3065424", "1.707053"),
    (2.0, "1.7206672", "2.960696"),
    (10.0, "2.6276754", "6.904678"),
    (1e4, "3.1409645", "9.865658"),
]


@pytest.mark.parametrize("kappa,k1,lam1", SMALL_KAPPA_TABLE)
def test_small_kappa_table(kappa, k1, lam1):
    # spec 3.5.5: as kappa -> 0 the first Robin eigenvalue goes to 2 kappa/L,
    # NOT to (pi/2L)^2 -- a test asserting the Dirichlet-like value is wrong here
    k = robin_wavenumbers(kappa, kappa, 1.0, 1)[0]
    assert_sig_figs(k, k1, sig=6, label=f"kappa={kappa} k1")
    assert_sig_figs(k**2, lam1, sig=5, label=f"kappa={kappa} lambda1")


def test_small_kappa_eigenvalue_approaches_2_kappa_over_L():
    for kappa in (1e-3, 1e-4, 1e-5):
        lam1 = robin_eigenvalues(kappa, kappa, 1.0, 1)[0]
        assert lam1 == pytest.approx(2 * kappa, rel=1e-2)


def test_robin_eigenfunctions_are_normalised_and_obey_the_boundary_conditions():
    L, kappa0, kappaL = 1.0, 0.5, 3.0
    k = robin_wavenumbers(kappa0, kappaL, L, 3)
    x = np.linspace(0.0, L, 20001)
    dx = x[1] - x[0]
    for kk in k:
        f = robin_mode(kk, kappa0, L, x)
        norm = np.trapezoid(f**2, x)
        assert norm == pytest.approx(1.0, rel=1e-6)
        # Robin at x=0: -f'(0) + kappa_0 f(0) = 0, i.e. f'(0) = kappa_0 f(0).
        # The derivative of a sampled function has to be second order or the
        # test measures its own truncation error: a two-point one-sided
        # difference is off by (dx/2) k^2 f(0), which is 2.3e-3 relative for
        # the k = 6.77 mode at dx = 5e-5 -- larger than any sane tolerance.
        # The three-point one-sided stencil is O(dx^2).
        d0 = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * dx)
        assert d0 == pytest.approx(kappa0 * f[0], rel=1e-6)
        # Robin at x=L: f'(L) = -kappa_L f(L)
        dL = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * dx)
        assert dL == pytest.approx(-kappaL * f[-1], rel=1e-6)


def test_robin_normalisation_limits():
    # spec 3.5.2: c = 0 with kL = n pi gives I = L/2 (and I = L for k = 0)
    L = 1.0
    for n in (1, 2, 3):
        assert robin_normalisation(n * math.pi, 0.0, L) == pytest.approx(L / 2, rel=1e-12)
    assert robin_normalisation(0.0, 0.0, L) == pytest.approx(L, rel=1e-12)


# ---------------------------------------------------------------------------
# negative couplings (3.5.6)
# ---------------------------------------------------------------------------
#: independently computed with mpmath (30-40 digits) from (3.12) and (3.19)
NEGATIVE_KAPPA = [
    # kappa, sorted eigenvalues (negative ones first)
    (-0.5, [-1.0891570972, 7.76457194358, 37.4562698938, 86.8168596136]),
    (-2.0, [-5.75691535956, 0.0, 31.323857845, 80.7629142257]),
    (-3.0, [-10.5211832596, -6.63412184701, 27.4983764539, 76.8296260255]),
    (-10.0, [-100.01814515, -99.9818251689, 15.0715983647, 57.4631427076]),
]


@pytest.mark.parametrize("kappa,expected", NEGATIVE_KAPPA)
def test_negative_couplings_analytic(kappa, expected):
    lam = robin_eigenvalues(kappa, kappa, 1.0, len(expected))
    assert np.allclose(lam, expected, rtol=1e-9, atol=1e-9), f"kappa={kappa}: {lam}"


@pytest.mark.parametrize("kappa,expected", NEGATIVE_KAPPA)
def test_negative_couplings_finite_difference(kappa, expected):
    # the discrete solver must find the same eigenvalues -- a second,
    # independent method agreeing with the root finder
    g = Grid1D(L=1.0, N=400, left=Robin(kappa), right=Robin(kappa))
    lam = solve(g, nmodes=len(expected)).lam
    assert np.allclose(lam, expected, rtol=2e-3, atol=2e-3), f"kappa={kappa}: {lam}"


def test_two_negative_eigenvalues_for_kappa_minus_three_and_minus_ten():
    # see the module docstring: the spec claims one negative eigenvalue here
    for kappa in (-3.0, -10.0):
        lam = robin_eigenvalues(kappa, kappa, 1.0, 3)
        assert np.sum(lam < 0) == 2, f"kappa={kappa}: {lam}"


def test_variational_bound_on_the_lowest_eigenvalue():
    # spec 3.5.6 item 1: for symmetric kappa < 0, lambda_min <= 2 kappa / L
    for kappa in (-0.5, -1.0, -2.0, -3.0, -10.0):
        lam_min = robin_eigenvalues(kappa, kappa, 1.0, 1)[0]
        assert lam_min <= 2 * kappa / 1.0 + 1e-12


def test_zero_mode_at_kappa_minus_two_over_L_is_the_second_eigenvalue():
    # spec 3.5.6 item 3: at kappa = -2/L the eigenvalues are
    # -5.756915, 0, 31.32, ... -- the zero mode is NOT the lowest
    lam = robin_eigenvalues(-2.0, -2.0, 1.0, 3)
    assert_sig_figs(lam[0], "-5.756915", sig=7)
    assert abs(lam[1]) < 1e-12
    assert_sig_figs(lam[2], "31.32", sig=4)


def test_equal_and_opposite_couplings_give_dirichlet_plus_one_negative_mode():
    # spec 3.5.6 item 4: for kappa_0 = -kappa_L = a, (3.19) gives mu = |a| and
    # the positive spectrum is exactly the Dirichlet spectrum.  This is the
    # signature of the wrong-sign bug of section 6 item 1.
    for a in (0.5, 2.0, 3.0):
        lam = robin_eigenvalues(-a, a, 1.0, 5)
        assert lam[0] == pytest.approx(-a * a, rel=1e-12)
        assert np.allclose(lam[1:], (np.arange(1, 5) * math.pi) ** 2, rtol=1e-12)
    # and the finite-difference solver reproduces it at N = 200
    g = Grid1D(L=1.0, N=200, left=Robin(-2.0), right=Robin(2.0))
    lam = solve(g, nmodes=4).lam
    assert lam[0] == pytest.approx(-4.0, rel=1e-4)
    assert np.allclose(lam[1:], (np.arange(1, 4) * math.pi) ** 2, rtol=1e-3)


def test_negative_kappa_produces_a_growing_mode_not_a_nan():
    # spec 6 item 11: sqrt of a negative omega^2 must not be nan and must not be
    # silently made positive
    g = Grid1D(L=1.0, N=100, left=Robin(-2.0), right=Robin(-2.0))
    spec = solve(g, m=0.0, nmodes=2)
    assert spec.lam[0] < 0
    assert spec.omega2[0] < 0
    assert np.isfinite(spec.omega[0])
    assert spec.omega[0].real == 0.0
    assert spec.omega[0].imag > 0.0  # grows as e^{+|omega| t}
    assert spec.omega[0].imag == pytest.approx(math.sqrt(-spec.lam[0]), rel=1e-4)
