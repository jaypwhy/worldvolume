"""The eigenvectors are W-orthonormal, not orthonormal (spec 4.3, trap #3).

``eigh(A_sym, W)`` returns ``v`` with ``v^T W v = I`` where ``W`` has entries
1/2 and 1 -- no factor of ``h``.  The spec reports
``max|V^T W V - I| = 1.4e-15`` against ``max|V^T V - I| = 0.142`` for
``kappa = 2`` at both ends with ``N = 20``, and warns that a test asserting
``V^T V = I`` fails for any Neumann or Robin problem and passes only for
all-Dirichlet.
"""

import numpy as np
import pytest

from worldvolume import (
    Box2D,
    Dirichlet,
    Grid1D,
    Neumann,
    Robin,
    solve,
)


def _orthonormality(spec):
    v = spec.v
    w = spec.weights
    return v.T @ (w[:, None] * v)


@pytest.mark.parametrize(
    "left,right",
    [
        (Dirichlet(), Dirichlet()),
        (Neumann(), Neumann()),
        (Neumann(), Dirichlet()),
        (Robin(2.0), Robin(2.0)),
        (Robin(0.5), Robin(3.0)),
        (Robin(-1.5), Neumann()),
    ],
)
@pytest.mark.parametrize("N", [7, 20, 40])
def test_v_transpose_W_v_is_the_identity(left, right, N):
    g = Grid1D(L=1.0, N=N, left=left, right=right)
    spec = solve(g)
    gram = _orthonormality(spec)
    worst = np.max(np.abs(gram - np.eye(gram.shape[0])))
    assert worst < 1e-13, f"{left!r}/{right!r} N={N}: max|V^T W V - I| = {worst:.3e}"


def test_v_transpose_v_is_not_the_identity_for_robin():
    # the exact numbers from spec 4.3: kappa = 2 at both ends, N = 20
    g = Grid1D(L=1.0, N=20, left=Robin(2.0), right=Robin(2.0))
    spec = solve(g)
    v = spec.v
    dev_w = np.max(np.abs(v.T @ (spec.weights[:, None] * v) - np.eye(v.shape[1])))
    dev_plain = np.max(np.abs(v.T @ v - np.eye(v.shape[1])))
    assert dev_w < 1e-14, f"max|V^T W V - I| = {dev_w:.3e}"
    # the spec reports 0.142; assert the value, not merely "not small"
    assert dev_plain == pytest.approx(0.142, abs=0.001), (
        f"max|V^T V - I| = {dev_plain:.6f}, spec reports 0.142"
    )
    assert dev_plain > 100 * dev_w


def test_v_transpose_v_is_the_identity_for_all_dirichlet():
    # the one case where the plain inner product happens to be the right one
    g = Grid1D(L=1.0, N=20, left=Dirichlet(), right=Dirichlet())
    spec = solve(g)
    v = spec.v
    assert np.max(np.abs(v.T @ v - np.eye(v.shape[1]))) < 1e-13


def test_w_orthonormality_in_2d():
    box = Box2D.from_lengths(
        1.0, 6, (Neumann(), Robin(1.0)), 1.5, 5, (Dirichlet(), Neumann())
    )
    spec = solve(box, nmodes=12)
    gram = _orthonormality(spec)
    assert np.max(np.abs(gram - np.eye(gram.shape[0]))) < 1e-13
    v = spec.v
    dev_plain = np.max(np.abs(v.T @ v - np.eye(v.shape[1])))
    assert dev_plain > 0.01


def test_eigenvectors_satisfy_the_robin_boundary_rows():
    # an end-to-end check that the ghost rows impose the right conditions: the
    # eigenvector must satisfy the *discrete* boundary equation exactly,
    #   ((2 + 2 h kappa_0) u_0 - 2 u_1)/h^2 = lambda u_0
    #   (-2 u_N + (2 + 2 h kappa_L) u_{N+1})/h^2 = lambda u_{N+1}
    # which is (4.5) with the ghost value eliminated
    L, N, kappa0, kappaL = 1.0, 40, 0.5, 3.0
    g = Grid1D(L=L, N=N, left=Robin(kappa0), right=Robin(kappaL))
    spec = solve(g, nmodes=3)
    h = g.h
    for i in range(3):
        u = spec.mode(i)  # values on all nodes, including both ends
        lam = spec.lam[i]
        left = ((2 + 2 * h * kappa0) * u[0] - 2 * u[1]) / h**2
        right = (-2 * u[-2] + (2 + 2 * h * kappaL) * u[-1]) / h**2
        assert left == pytest.approx(lam * u[0], rel=1e-10, abs=1e-8)
        assert right == pytest.approx(lam * u[-1], rel=1e-10, abs=1e-8)
        # and the continuum Robin condition holds to the O(h) accuracy of a
        # one-sided difference at the boundary node: the error is (h/2) f'',
        # i.e. O(h k^2), not O(h kappa^2)
        assert abs((u[1] - u[0]) / h - kappa0 * u[0]) < 10 * h * (lam + 1) * abs(u[0]) + 1e-10
        assert abs((u[-1] - u[-2]) / h + kappaL * u[-1]) < 10 * h * (lam + 1) * abs(u[-1]) + 1e-10


def test_neumann_boundary_rows_are_satisfied_exactly():
    # for Neumann the ghost reflection f_{-1} = f_1 is exact, so the boundary
    # row reproduces the eigenvalue equation to machine precision
    L, N = 1.0, 30
    g = Grid1D(L=L, N=N, left=Neumann(), right=Neumann())
    spec = solve(g, nmodes=3)
    h = g.h
    for i in range(3):
        u = spec.mode(i)
        lam = spec.lam[i]
        assert (2 * u[0] - 2 * u[1]) / h**2 == pytest.approx(lam * u[0], rel=1e-12, abs=1e-10)
        assert (-2 * u[-2] + 2 * u[-1]) / h**2 == pytest.approx(lam * u[-1], rel=1e-12, abs=1e-10)
