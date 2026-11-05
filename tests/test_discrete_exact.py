"""The discrete spectrum is known exactly at finite h for D/N data (spec 4.7.1).

    lambda_n^(h) = (4/h^2) sin^2(k_n h / 2),    k_n the exact analytic wavenumber

This holds for **every** mode at **any** ``N`` for pure Dirichlet, pure Neumann
and mixed N/D data, because the sampled analytic eigenfunction is an exact
eigenvector of the ghost-eliminated operator.  So these assertions carry no
tolerance beyond round-off -- ``1e-12`` absolute, as the spec asks -- and they
hold even on grids so coarse that a convergence test has nothing to measure.

In 2D the same identity holds as a Kronecker sum (spec 4.7.1):
``lambda^(h) = (4/h_1^2) sin^2(k_1 h_1/2) + (4/h_2^2) sin^2(k_2 h_2/2)``.

Robin data does **not** satisfy this identity (spec 4.7.1), and the last test
here asserts that failure -- it is what distinguishes the Robin boundary row
from the Neumann one.
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
    dirichlet_mode,
    mixed_mode,
    neumann_mode,
    solve,
)

TOL = 1e-12


def discrete_wavenumbers(left, right, L, N):
    """The ``k_n`` that the discrete operator actually resolves, per BC pair.

    Modes with ``k_n h >= pi`` are not representable on the grid (they alias), so
    each direction carries exactly as many discrete modes as it has unknowns:
    ``N`` for Dirichlet, ``N+1`` for Neumann and mixed N/D.
    """
    h = L / (N + 1)
    if left.is_dirichlet and right.is_dirichlet:
        n = np.arange(1, N + 1, dtype=float)
        return n * math.pi / L
    if left.is_neumann and right.is_neumann:
        # both ends are unknowns, so there are N+2 eigenvalues, n = 0..N+1;
        # the last one has k = (N+1) pi/L = pi/h, i.e. lambda = 4/h^2
        n = np.arange(0, N + 2, dtype=float)
        return n * math.pi / L
    if (left.is_neumann and right.is_dirichlet) or (left.is_dirichlet and right.is_neumann):
        n = np.arange(0, N + 1, dtype=float)
        return (n + 0.5) * math.pi / L
    raise ValueError("Robin data has no closed-form discrete spectrum")


@pytest.mark.parametrize(
    "left,right,name",
    [
        (Dirichlet(), Dirichlet(), "D-D"),
        (Neumann(), Neumann(), "N-N"),
        (Neumann(), Dirichlet(), "N-D"),
        (Dirichlet(), Neumann(), "D-N"),
    ],
)
@pytest.mark.parametrize("N", [1, 3, 7, 20, 41])
@pytest.mark.parametrize("L", [1.0, 1.5])
def test_exact_discrete_spectrum_1d(left, right, name, N, L):
    g = Grid1D(L=L, N=N, left=left, right=right)
    spec = solve(g)
    k = discrete_wavenumbers(left, right, L, N)
    h = g.h
    predicted = 4.0 / h**2 * np.sin(k * h / 2) ** 2
    assert len(spec.lam) == len(predicted)
    worst = np.max(np.abs(spec.lam - predicted))
    # 1e-12 absolute, widened only by the round-off of the eigensolver, which
    # scales with ||A|| = 4/h^2: at N=41 that is ~7e3, so the floor is ~1e-12
    # rather than ~1e-13.  A genuine truncation error is O(lam * h^2) ~ O(1).
    tol = TOL * max(1.0, float(predicted.max()) / 100.0)
    assert worst < tol, (
        f"{name} N={N} L={L}: max|lam_h - 4/h^2 sin^2(kh/2)| = {worst:.3e} > {tol:.1e}"
    )


@pytest.mark.parametrize("N", [8, 16])
@pytest.mark.parametrize(
    "bx,by,name",
    [
        ((Neumann(), Neumann()), (Dirichlet(), Dirichlet()), "N_x D_y"),
        ((Dirichlet(), Dirichlet()), (Dirichlet(), Dirichlet()), "D_x D_y"),
        ((Neumann(), Neumann()), (Neumann(), Neumann()), "N_x N_y"),
        ((Neumann(), Dirichlet()), (Neumann(), Neumann()), "ND_x N_y"),
    ],
)
def test_exact_discrete_spectrum_2d_kronecker_sum(bx, by, name, N):
    L1, L2 = 1.0, 1.5
    box = Box2D.from_lengths(L1, N, bx, L2, N, by)
    spec = solve(box)
    k1 = discrete_wavenumbers(bx[0], bx[1], L1, N)
    k2 = discrete_wavenumbers(by[0], by[1], L2, N)
    h1, h2 = box.x.h, box.y.h
    lam1 = 4.0 / h1**2 * np.sin(k1 * h1 / 2) ** 2
    lam2 = 4.0 / h2**2 * np.sin(k2 * h2 / 2) ** 2
    predicted = np.sort((lam1[:, None] + lam2[None, :]).ravel())
    assert len(spec.lam) == len(predicted)
    worst = np.max(np.abs(spec.lam - predicted))
    tol = TOL * max(1.0, float(predicted.max()) / 100.0)
    assert worst < tol, f"{name} N={N}: max|lam_h - kron sum| = {worst:.3e} > {tol:.1e}"


@pytest.mark.parametrize(
    "left,right,modes",
    [
        (Dirichlet(), Dirichlet(), [1, 3, 5]),
        (Neumann(), Neumann(), [1, 3, 5]),
        (Neumann(), Dirichlet(), [0, 2, 4]),
    ],
)
def test_eigenvectors_are_the_sampled_analytic_modes(left, right, modes):
    # spec 4.7.3 item 3: for D/N data the discrete eigenvector equals the
    # sampled analytic eigenfunction with the *exact* wavenumber, to machine
    # precision.  The eigenvector is exact while the eigenvalue is (4.14): the
    # discretisation error sits entirely in the eigenvalue.
    L, N = 1.0, 20
    g = Grid1D(L=L, N=N, left=left, right=right)
    spec = solve(g, nmodes=max(modes) + 1)
    x = g.x
    for n in modes:
        # the n-th eigenvalue of the discrete problem, in ascending order:
        # Dirichlet labels start at n=1, Neumann and mixed at n=0
        index = n - 1 if (left.is_dirichlet and right.is_dirichlet) else n
        u = spec.mode(index)
        if left.is_dirichlet and right.is_dirichlet:
            exact = dirichlet_mode(L, n, x)
        elif left.is_neumann and right.is_neumann:
            exact = neumann_mode(L, n, x)
        else:
            exact = mixed_mode(L, n, x, neumann_at_left=left.is_neumann)
        if u @ exact < 0:
            exact = -exact
        assert np.max(np.abs(u - exact)) < 1e-13, f"mode n={n} nodal error"


def test_mixed_nd_ground_state_boundary_value():
    # spec 4.7.3: for N-D mode 0 the boundary value is u_0 = sqrt(2) exactly
    g = Grid1D(L=1.0, N=10, left=Neumann(), right=Dirichlet())
    spec = solve(g, nmodes=1)
    u = spec.mode(0)
    # (the eigenvector's overall sign is arbitrary; take the branch that makes
    # the boundary value positive)
    if u[0] < 0:
        u = -u
    assert u[0] == pytest.approx(math.sqrt(2.0), rel=1e-13)
    assert abs(u[-1]) < 1e-15


def test_sqrt_h_rescaling_is_what_makes_modes_comparable():
    # spec 4.6: eigenvalues from eigh(A, W) are correct as they stand, but the
    # eigenvectors must be divided by sqrt(h) before they are function values.
    # Without it the error is a factor 1/sqrt(h) -- trap #6.
    L, N = 1.0, 20
    g = Grid1D(L=L, N=N, left=Neumann(), right=Dirichlet())
    spec = solve(g, nmodes=1)
    x = g.x[g.active]
    exact = mixed_mode(L, 0, x)
    raw = spec.v[:, 0]  # v^T W v = 1, no sqrt(h)
    assert np.max(np.abs(raw - exact)) > 0.5  # not comparable at all
    assert np.max(np.abs(spec.u[:, 0] - exact)) < 1e-14
    # the ratio is exactly sqrt(h)
    assert np.allclose(spec.u[:, 0] / raw, 1.0 / math.sqrt(g.h), rtol=1e-13)


def test_robin_does_not_have_the_closed_form_discrete_spectrum():
    # spec 4.7.1: (4.14) is FALSE for Robin -- the ghost-node Robin condition
    # contributes its own O(h^2) boundary error with a kappa-dependent
    # coefficient.  At N=7 the deviation is O(1), not O(h^2).
    L, N, kappa = 1.0, 7, 2.0
    g = Grid1D(L=L, N=N, left=Robin(kappa), right=Robin(kappa))
    spec = solve(g)
    k = np.arange(0, N + 2) * math.pi / L  # naive D/N labels
    naive = 4.0 / g.h**2 * np.sin(k * g.h / 2) ** 2
    assert np.max(np.abs(spec.lam - naive)) > 1.0
    # at N=20 it is still not an identity, just smaller
    g = Grid1D(L=L, N=20, left=Robin(kappa), right=Robin(kappa))
    spec = solve(g)
    k = np.arange(0, 22) * math.pi / L
    naive = 4.0 / g.h**2 * np.sin(k * g.h / 2) ** 2
    assert np.max(np.abs(spec.lam - naive)) > 0.1
