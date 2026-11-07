"""Boundary-condition objects and grid conventions (spec sections 2.1, 4.1)."""

import math

import numpy as np
import pytest

from worldvolume import Box2D, Dirichlet, Grid1D, Neumann, Robin, as_bc
from worldvolume.bc import BoundaryCondition


# ---------------------------------------------------------------------------
# boundary conditions
# ---------------------------------------------------------------------------
def test_kappa_encoding_of_the_three_conditions():
    # spec section 2.1: kappa = 0 is Neumann, kappa -> inf is Dirichlet
    assert Neumann().kappa == 0.0
    assert Dirichlet().kappa == math.inf
    assert Robin(2.5).kappa == 2.5
    assert Neumann().is_neumann and not Neumann().is_dirichlet
    assert Dirichlet().is_dirichlet and not Dirichlet().is_robin
    assert Robin(-1.0).is_robin and Robin(-1.0).is_robin_family
    assert Neumann().is_robin_family and not Dirichlet().is_robin_family


def test_robin_accepts_zero_and_negative_couplings():
    # kappa = 0 really is Neumann (spec 2.2), and negative kappa is legal and
    # self-adjoint (spec 2.3, 3.5.6)
    assert Robin(0.0).kappa == 0.0
    assert Robin(-3.0).kappa == -3.0


def test_robin_rejects_infinite_coupling():
    # an infinite coupling would put inf in the matrix; Dirichlet is the
    # explicit spelling of that limit
    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError):
            Robin(bad)


def test_equality_and_hash():
    assert Neumann() == Neumann()
    assert Neumann() != Dirichlet()
    assert Robin(2.0) == Robin(2.0)
    assert Robin(2.0) != Robin(2.000001)
    assert len({Neumann(), Neumann(), Dirichlet(), Robin(1.0)}) == 3
    assert isinstance(Neumann(), BoundaryCondition)
    assert repr(Robin(2.0)) == "Robin(kappa=2.0)"


def test_as_bc_coercion():
    assert as_bc("N") == Neumann()
    assert as_bc("d") == Dirichlet()
    assert as_bc(Neumann()) == Neumann()
    assert as_bc(2.0) == Robin(2.0)
    assert as_bc(0) == Robin(0.0)
    with pytest.raises(ValueError):
        as_bc("R")  # needs a kappa
    with pytest.raises(TypeError):
        as_bc(object())


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L,N", [(1.0, 7), (1.5, 20), (2.0, 1), (0.3, 64)])
def test_grid_spacing_and_endpoints(L, N):
    # spec eq. 4.1: h = L/(N+1), nodes j h for j = 0..N+1, both ends included
    g = Grid1D(L=L, N=N, left=Neumann(), right=Neumann())
    assert g.h == pytest.approx(L / (N + 1), rel=0, abs=0)
    assert g.n_nodes == N + 2
    x = g.x
    assert len(x) == N + 2
    assert x[0] == 0.0 and x[-1] == L
    assert np.allclose(np.diff(x), g.h, rtol=0, atol=1e-15 * L)


def test_grid_rejects_bad_parameters():
    with pytest.raises(ValueError):
        Grid1D(L=0.0, N=4, left=Neumann(), right=Neumann())
    with pytest.raises(ValueError):
        Grid1D(L=-1.0, N=4, left=Neumann(), right=Neumann())
    with pytest.raises(ValueError):
        Grid1D(L=1.0, N=0, left=Neumann(), right=Neumann())


@pytest.mark.parametrize(
    "left,right,expected_active,expected_weights",
    [
        # Dirichlet nodes are eliminated, so they are not unknowns (4.4)
        (Dirichlet(), Dirichlet(), 8, np.ones(8)),
        (Neumann(), Neumann(), 10, np.array([0.5] + [1.0] * 8 + [0.5])),
        (Neumann(), Dirichlet(), 9, np.array([0.5] + [1.0] * 8)),
        (Dirichlet(), Neumann(), 9, np.array([1.0] * 8 + [0.5])),
        (Robin(2.0), Robin(3.0), 10, np.array([0.5] + [1.0] * 8 + [0.5])),
        (Robin(0.0), Neumann(), 10, np.array([0.5] + [1.0] * 8 + [0.5])),
    ],
)
def test_active_set_and_trapezoidal_weights(left, right, expected_active, expected_weights):
    # spec eq. 4.7: w = 1/2 at a Neumann/Robin boundary node, 1 inside
    g = Grid1D(L=1.0, N=8, left=left, right=right)
    assert g.n_active == expected_active
    assert g.active.sum() == expected_active
    assert np.array_equal(g.weights, expected_weights)
    assert len(g.x_active) == expected_active


def test_boundary_conditions_are_independent_per_side():
    g = Grid1D(L=1.0, N=8, left=Dirichlet(), right=Robin(0.5))
    assert g.left.is_dirichlet and g.right.is_robin
    assert g.right.kappa == 0.5
    # swapping the ends gives a different (mirrored) problem
    h = Grid1D(L=1.0, N=8, left=Robin(0.5), right=Dirichlet())
    assert h.left.is_robin and h.right.is_dirichlet


def test_uniform_helper_and_pair_coercion():
    g = Grid1D.uniform(2.0, 5, "N")
    assert g.left == Neumann() and g.right == Neumann()
    b = Box2D.from_lengths(1.0, 4, (Neumann(), Dirichlet()), 1.5, 6, 2.0)
    assert b.x.left == Neumann() and b.x.right == Dirichlet()
    assert b.y.left == Robin(2.0) and b.y.right == Robin(2.0)


def test_box2d_active_mask_is_the_product_of_the_1d_sets():
    box = Box2D.from_lengths(1.0, 4, Neumann(), 1.5, 6, Dirichlet())
    assert box.shape == (6, 8)
    assert box.n_active == 6 * 6  # x: 6 unknowns, y: 6 interior nodes
    assert np.array_equal(box.active, np.outer(box.x.active, box.y.active))
    # the whole Dirichlet face is removed, corners included
    assert not box.active[:, 0].any() and not box.active[:, -1].any()
    xx, yy = box.mesh()
    assert xx.shape == yy.shape == box.shape
    assert xx[0, 0] == 0.0 and yy[0, 0] == 0.0
    assert xx[-1, -1] == pytest.approx(1.0) and yy[-1, -1] == pytest.approx(1.5)
