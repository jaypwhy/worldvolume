"""Assembly of the finite-difference Laplacian (spec sections 4.2-4.5, 4.8).

The tests here check the *matrix*, not the spectrum: the ghost-node boundary
rows, the Dirichlet elimination, the symmetrisation ``A_sym = W A'``, and the
2D Kronecker structure with its index convention.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from worldvolume import (
    Box2D,
    Dirichlet,
    Grid1D,
    Neumann,
    Robin,
    laplacian_1d,
    laplacian_1d_unsymmetrised,
    laplacian_2d,
    mass_matrix_1d,
)


def test_explicit_matrix_matches_spec_4_6():
    # spec eq. 4.6: N = 3, Neumann at x=0, Robin kappa_L at x=L, L = 1 so h = 1/4
    L, N, kappaL = 1.0, 3, 0.7
    g = Grid1D(L=L, N=N, left=Neumann(), right=Robin(kappaL))
    h = g.h
    assert h == 0.25
    a = laplacian_1d_unsymmetrised(g).toarray()
    expected = np.array(
        [
            [2, -2, 0, 0, 0],
            [-1, 2, -1, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 0, -1, 2, -1],
            [0, 0, 0, -2, 2 + 2 * h * kappaL],
        ],
        dtype=float,
    ) / h**2
    assert np.array_equal(a, expected)
    # and it is genuinely not symmetric: the (0,1) entry is -2/h^2, (1,0) -1/h^2
    assert a[0, 1] == -2 / h**2 and a[1, 0] == -1 / h**2
    assert not np.array_equal(a, a.T)


@pytest.mark.parametrize("kappa", [0.0, 0.5, 2.0, -1.5, 1e6])
def test_boundary_rows_follow_4_5(kappa):
    # (-Delta_h f)_0      = ((2 + 2 h kappa_0) f_0 - 2 f_1)/h^2
    # (-Delta_h f)_{N+1}  = (-2 f_N + (2 + 2 h kappa_L) f_{N+1})/h^2
    L, N = 1.0, 5
    g = Grid1D(L=L, N=N, left=Robin(kappa), right=Robin(kappa))
    h = g.h
    a = laplacian_1d_unsymmetrised(g).toarray()
    assert a[0, 0] == pytest.approx((2 + 2 * h * kappa) / h**2, rel=1e-14)
    assert a[0, 1] == pytest.approx(-2 / h**2, rel=1e-14)
    assert a[-1, -1] == pytest.approx((2 + 2 * h * kappa) / h**2, rel=1e-14)
    # the right ghost row uses f_N, not f_{N+1} (spec section 6 item 5)
    assert a[-1, -2] == pytest.approx(-2 / h**2, rel=1e-14)
    # interior rows are the plain three-point stencil (4.2)
    for j in range(1, N + 1):
        assert a[j, j - 1] == pytest.approx(-1 / h**2, rel=1e-14)
        assert a[j, j] == pytest.approx(2 / h**2, rel=1e-14)
        assert a[j, j + 1] == pytest.approx(-1 / h**2, rel=1e-14)


def test_dirichlet_nodes_are_eliminated_not_constrained():
    # spec 4.4: for all-Dirichlet the matrix is the textbook (1/h^2)
    # tridiag(-1, 2, -1) of size N and W = I
    L, N = 1.0, 6
    g = Grid1D(L=L, N=N, left=Dirichlet(), right=Dirichlet())
    a, w = laplacian_1d(g)
    h = g.h
    dense = a.toarray()
    expected = (np.diag(np.full(N, 2.0)) + np.diag(np.full(N - 1, -1.0), 1)
                + np.diag(np.full(N - 1, -1.0), -1)) / h**2
    assert np.array_equal(dense, expected)
    assert np.array_equal(w.toarray(), np.eye(N))
    assert a.shape == (N, N)


def test_one_sided_dirichlet_leaves_no_ghost_row():
    g = Grid1D(L=1.0, N=4, left=Dirichlet(), right=Neumann())
    a = laplacian_1d_unsymmetrised(g).toarray()
    assert a.shape == (5, 5)
    # first row is the interior stencil (the x=0 node is gone)
    assert a[0, 0] == pytest.approx(2 / g.h**2) and a[0, 1] == pytest.approx(-1 / g.h**2)
    # last row is still the Neumann ghost row
    assert a[-1, -2] == pytest.approx(-2 / g.h**2)


@pytest.mark.parametrize(
    "left,right",
    [
        (Dirichlet(), Dirichlet()),
        (Neumann(), Neumann()),
        (Neumann(), Dirichlet()),
        (Dirichlet(), Neumann()),
        (Robin(2.0), Robin(2.0)),
        (Robin(0.5), Robin(3.0)),
        (Robin(0.0), Neumann()),
        (Robin(-2.0), Robin(2.0)),
    ],
)
@pytest.mark.parametrize("N", [1, 7, 20])
def test_a_sym_is_symmetric_to_zero_not_merely_small(left, right, N):
    # spec 4.3: A_sym = W A' is symmetric exactly; max|A_sym - A_sym^T| = 0
    g = Grid1D(L=1.0, N=N, left=left, right=right)
    a, w = laplacian_1d(g)
    diff = a - a.T
    assert diff.nnz == 0 or np.max(np.abs(diff.data)) == 0.0
    # and A_sym is literally W @ A'
    a_prime = laplacian_1d_unsymmetrised(g)
    assert np.array_equal(a.toarray(), (w @ a_prime).toarray())
    # W is diagonal and positive
    assert np.array_equal(w.toarray(), np.diag(g.weights))


def test_2d_assembly_is_symmetric_to_zero():
    box = Box2D.from_lengths(
        1.0, 7, (Neumann(), Robin(1.5)), 1.5, 5, (Dirichlet(), Neumann())
    )
    a, m = laplacian_2d(box)
    d = a - a.T
    assert d.nnz == 0 or np.max(np.abs(d.data)) == 0.0
    dm = m - m.T
    assert dm.nnz == 0 or np.max(np.abs(dm.data)) == 0.0
    assert np.all(m.diagonal() > 0)
    assert a.shape == (box.n_active, box.n_active)


def test_2d_kronecker_eigenvalues_are_the_1d_sums():
    # spec 4.5 / 4.11: eigenvalues of (A1 (x) W2 + W1 (x) A2, W1 (x) W2) are
    # exactly the sums of the 1D eigenvalues
    import scipy.linalg as sla

    box = Box2D.from_lengths(1.0, 6, Neumann(), 1.5, 5, Dirichlet())
    a1, w1 = laplacian_1d(box.x)
    a2, w2 = laplacian_1d(box.y)
    lam1 = sla.eigh(a1.toarray(), w1.toarray())[0]
    lam2 = sla.eigh(a2.toarray(), w2.toarray())[0]
    a, m = laplacian_2d(box)
    lam2d = sla.eigh(a.toarray(), m.toarray())[0]
    sums = np.sort((lam1[:, None] + lam2[None, :]).ravel())
    assert np.allclose(lam2d, sums, rtol=1e-12, atol=1e-10 * abs(sums).max())


def test_2d_index_convention_first_factor_slowest():
    # spec 4.5: with kron(A1, W2) the flattened index is i*n_y + j for (x_i, y_j),
    # i.e. row-major with the y index fastest.  Check it on the ground state of
    # a box that is Dirichlet in both directions: the mode is a single half-wave
    # in each direction, so the vector must equal the Kronecker product of the
    # two 1D ground states.
    import scipy.linalg as sla

    from worldvolume import solve

    box = Box2D.from_lengths(1.0, 5, Dirichlet(), 1.5, 4, Dirichlet())
    a1, w1 = laplacian_1d(box.x)
    a2, w2 = laplacian_1d(box.y)
    _, v1 = sla.eigh(a1.toarray(), w1.toarray())
    _, v2 = sla.eigh(a2.toarray(), w2.toarray())
    a, m = laplacian_2d(box)
    _, v = sla.eigh(a.toarray(), m.toarray())
    ground = v[:, 0]
    expected = np.outer(v1[:, 0], v2[:, 0]).ravel()
    if ground @ expected < 0:
        expected = -expected
    assert np.allclose(ground, expected, atol=1e-12)
    # reshaped with the y index fastest, it is a single half-wave in each
    # direction (no interior nodes)
    active_shape = (box.x.n_active, box.y.n_active)
    grid = ground.reshape(active_shape)
    assert np.all(grid > 0) or np.all(grid < 0)
    # and the solver's mode() puts it back on the full grid consistently: it
    # returns function values u = v/sqrt(h1 h2), i.e. the raw vector times
    # 1/sqrt(h1*h2) (spec 4.6)
    full = solve(box, nmodes=1).mode(0)
    scale = 1.0 / math.sqrt(box.x.h * box.y.h)
    assert np.allclose(full[box.active], ground * scale, atol=1e-12)
    assert np.all(full[:, 0] == 0.0)


def test_mass_matrix_is_the_trapezoidal_rule():
    g = Grid1D(L=2.0, N=9, left=Neumann(), right=Robin(1.0))
    w = mass_matrix_1d(g)
    assert sp.issparse(w)
    # N = 9 interior nodes, both ends Neumann/Robin: (1/2, 1, ..., 1, 1/2)
    assert np.array_equal(w.toarray(), np.diag([0.5] + [1.0] * 9 + [0.5]))
    # the trapezoidal rule reproduces int_0^L 1 dx exactly
    assert g.h * np.sum(g.weights) == pytest.approx(g.L, rel=0, abs=1e-15)
