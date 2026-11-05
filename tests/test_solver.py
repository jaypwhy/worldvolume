"""The solver: eigenvalues, omega, rescaled eigenvectors, time evolution.

Also the isospectrality checks of spec 3.4 / 3.6.3, which are exact symmetries
of the scheme and must hold to machine precision in the discrete solver too.
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
    omega_from_lambda,
    solve,
    spectrum_1d,
    spectrum_2d,
    time_evolution,
    wavenumbers,
)


# ---------------------------------------------------------------------------
# eigenvalues and omega
# ---------------------------------------------------------------------------
def test_solve_returns_lambda_omega2_omega_consistently():
    g = Grid1D(L=1.0, N=20, left=Neumann(), right=Dirichlet())
    m = 0.25
    spec = solve(g, m=m, nmodes=4)
    assert len(spec.lam) == len(spec.omega2) == len(spec.omega) == 4
    assert np.allclose(spec.omega2, spec.lam + m * m, rtol=1e-15)
    assert np.allclose(spec.omega.real, np.sqrt(spec.omega2), rtol=1e-14)
    assert np.all(spec.omega.imag == 0.0)
    assert np.all(np.diff(spec.lam) > 0)
    # lambda comes from the boundary conditions alone: m does not touch it
    spec0 = solve(g, m=0.0, nmodes=4)
    assert np.array_equal(spec.lam, spec0.lam)


def test_omega_of_a_tachyonic_mass_is_imaginary():
    # spec 1.4 item 2: m^2 < 0 can make lambda + m^2 < 0; omega is then
    # imaginary and the mode grows.  Must not be nan, must not be |...|.
    g = Grid1D(L=1.0, N=20, left=Neumann(), right=Neumann())
    spec = solve(g, m=1j, nmodes=2)  # m^2 = -1
    assert spec.lam[0] == pytest.approx(0.0, abs=1e-12)
    assert spec.omega2[0] == pytest.approx(-1.0, abs=1e-12)
    assert spec.omega[0] == pytest.approx(1j, abs=1e-10)
    assert np.isfinite(spec.omega).all()


def test_omega_from_lambda_branch():
    assert omega_from_lambda([4.0], 0.0)[0] == pytest.approx(2.0 + 0j)
    assert omega_from_lambda([-4.0], 0.0)[0] == pytest.approx(2j)
    assert omega_from_lambda([0.0], 1.5)[0] == pytest.approx(1.5 + 0j)


def test_dense_and_sparse_paths_agree():
    g = Grid1D(L=1.0, N=60, left=Robin(0.5), right=Neumann())
    dense = solve(g, nmodes=8, method="dense")
    sparse = solve(g, nmodes=8, method="sparse")
    assert dense.method == "dense" and sparse.method == "sparse"
    assert np.allclose(dense.lam, sparse.lam, rtol=1e-10, atol=1e-10)
    # eigenvectors agree up to sign
    for i in range(8):
        a, b = dense.u[:, i], sparse.u[:, i]
        if a @ b < 0:
            b = -b
        assert np.max(np.abs(a - b)) < 1e-8


def test_sparse_path_on_a_2d_box():
    box = Box2D.from_lengths(1.0, 20, Neumann(), 1.5, 18, Dirichlet())
    sparse = solve(box, nmodes=5, method="sparse")
    dense = solve(box, nmodes=5, method="dense")
    assert np.allclose(sparse.lam, dense.lam, rtol=1e-9, atol=1e-9)


def test_solve_nmodes_none_returns_everything():
    g = Grid1D(L=1.0, N=6, left=Dirichlet(), right=Dirichlet())
    spec = solve(g)
    assert len(spec.lam) == g.n_active == 6


def test_mode_has_the_shape_of_the_grid_and_zeros_at_dirichlet_nodes():
    g = Grid1D(L=1.0, N=10, left=Dirichlet(), right=Neumann())
    spec = solve(g, nmodes=2)
    u = spec.mode(0)
    assert u.shape == (g.n_nodes,)
    assert u[0] == 0.0  # Dirichlet node
    assert abs(u[-1]) > 0.1  # Neumann node is not constrained
    box = Box2D.from_lengths(1.0, 6, Neumann(), 1.5, 5, Dirichlet())
    spec2 = solve(box, nmodes=1)
    u2 = spec2.mode(0)
    assert u2.shape == box.shape
    assert np.all(u2[:, 0] == 0.0) and np.all(u2[:, -1] == 0.0)


def test_project_returns_the_mode_coefficients():
    g = Grid1D(L=1.0, N=30, left=Neumann(), right=Neumann())
    spec = solve(g, nmodes=5)
    # projecting an eigenvector returns the unit vector
    for i in range(5):
        c = spec.project(spec.mode(i))
        expected = np.zeros(5)
        expected[i] = 1.0
        assert np.allclose(c, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# isospectrality (spec 3.4, 3.6.3)
# ---------------------------------------------------------------------------
def test_nd_and_dn_are_isospectral_in_the_discrete_solver():
    for N in (5, 20):
        a = solve(Grid1D(1.0, N, Neumann(), Dirichlet()), nmodes=5).lam
        b = solve(Grid1D(1.0, N, Dirichlet(), Neumann()), nmodes=5).lam
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12)


def test_mirrored_2d_box_is_isospectral():
    # N_x D_y on (L1, L2) vs D_x N_y on (L2, L1)
    a = solve(Box2D.from_lengths(1.0, 8, Neumann(), 1.5, 6, Dirichlet()), nmodes=6).lam
    b = solve(Box2D.from_lengths(1.5, 6, Dirichlet(), 1.0, 8, Neumann()), nmodes=6).lam
    assert np.allclose(a, b, rtol=1e-12, atol=1e-12)


def test_robin_zero_coupling_is_neumann_in_the_discrete_solver():
    a = solve(Grid1D(1.0, 12, Robin(0.0), Robin(0.0)), nmodes=4).lam
    b = solve(Grid1D(1.0, 12, Neumann(), Neumann()), nmodes=4).lam
    assert np.allclose(a, b, rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------
# time evolution
# ---------------------------------------------------------------------------
def test_time_evolution_reproduces_initial_data():
    g = Grid1D(L=1.0, N=24, left=Neumann(), right=Dirichlet())
    spec = solve(g, m=0.3)
    x = g.x
    phi0 = np.sin(2 * np.pi * x) + 0.5 * np.cos(np.pi * x)
    dphi0 = np.zeros_like(x)
    out = time_evolution(spec, phi0, dphi0, [0.0])
    assert out.shape == (1, g.n_nodes)
    assert np.allclose(out[0], phi0, atol=1e-12)


def test_time_evolution_reproduces_the_initial_velocity():
    g = Grid1D(L=1.0, N=24, left=Neumann(), right=Neumann())
    spec = solve(g, m=0.5)
    x = g.x
    phi0 = np.zeros_like(x)
    dphi0 = np.cos(3 * np.pi * x)
    dt = 1e-6
    out = time_evolution(spec, phi0, dphi0, [-dt, dt])
    velocity = (out[1] - out[0]) / (2 * dt)
    assert np.allclose(velocity, dphi0, atol=1e-6)


def test_single_mode_evolves_as_a_cosine():
    g = Grid1D(L=1.0, N=24, left=Dirichlet(), right=Dirichlet())
    spec = solve(g, m=0.0, nmodes=3)
    phi0 = spec.mode(1)
    dphi0 = np.zeros_like(phi0)
    omega = spec.omega[1].real
    times = np.linspace(0.0, 2 * math.pi / omega, 7)
    out = time_evolution(spec, phi0, dphi0, times)
    for t, row in zip(times, out):
        assert np.allclose(row, math.cos(omega * t) * phi0, atol=1e-10)


def test_time_evolution_satisfies_the_klein_gordon_equation():
    # a genuine check of the helper: phi(t+dt) - 2 phi(t) + phi(t-dt) must equal
    # dt^2 (Delta_h phi - m^2 phi), with Delta_h the assembled operator
    from worldvolume import assemble

    m = 0.4
    g = Grid1D(L=1.0, N=20, left=Neumann(), right=Dirichlet())
    spec = solve(g, m=m)
    a, _ = assemble(g)
    x = g.x
    phi0 = np.exp(-((x - 0.4) ** 2) / 0.02)
    dphi0 = np.zeros_like(x)
    dt = 1e-4
    out = time_evolution(spec, phi0, dphi0, [-dt, 0.0, dt])
    # restrict to active nodes; the operator acts on the unknown vector
    active = g.active
    laplacian_phi = a @ out[1][active]
    rhs = (laplacian_phi - m * m * out[1][active]) * dt**2
    lhs = out[2][active] - 2 * out[1][active] + out[0][active]
    scale = max(np.max(np.abs(lhs)), 1e-30)
    assert np.max(np.abs(lhs - rhs)) < 1e-6 * scale


def test_static_zero_mode_does_not_move():
    # the m = 0 Neumann zero mode is a static solution (spec 3.3)
    g = Grid1D(L=2.0, N=16, left=Neumann(), right=Neumann())
    spec = solve(g, m=0.0, nmodes=1)
    phi0 = spec.mode(0)
    out = time_evolution(spec, phi0, np.zeros_like(phi0), [0.0, 1.0, 5.0])
    for row in out:
        assert np.allclose(row, phi0, atol=1e-12)


def test_growing_mode_grows_exponentially():
    # kappa < 0 gives an imaginary omega: the mode grows as cosh(|omega| t)
    g = Grid1D(L=1.0, N=100, left=Robin(-2.0), right=Robin(-2.0))
    spec = solve(g, m=0.0, nmodes=1)
    assert spec.lam[0] < 0
    nu = spec.omega[0].imag
    phi0 = spec.mode(0)
    times = np.array([0.0, 0.1, 0.2])
    out = time_evolution(spec, phi0, np.zeros_like(phi0), times)
    for t, row in zip(times, out):
        assert np.allclose(row, math.cosh(nu * t) * phi0, atol=1e-8)


def test_time_evolution_in_2d():
    box = Box2D.from_lengths(1.0, 10, Neumann(), 1.5, 8, Dirichlet())
    spec = solve(box, m=0.2, nmodes=6)
    xx, yy = box.mesh()
    phi0 = np.sin(math.pi * yy / 1.5) * np.cos(math.pi * xx)
    dphi0 = np.zeros_like(phi0)
    out = time_evolution(spec, phi0, dphi0, [0.0, 0.5])
    assert out.shape == (2,) + box.shape
    assert np.allclose(out[0], phi0, atol=1e-10)
