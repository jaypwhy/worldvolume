# worldvolume

An open-source Python toolkit for solving field equations on D-brane worldvolumes
with mixed Neumann / Dirichlet / Robin boundary conditions.

Status: in development. The first working version solves the Klein-Gordon equation
on a rectangular worldvolume with an independently chosen boundary condition per
direction, and validates the computed spectrum against closed-form analytic results.

See `docs/bc-conventions.md` for the derivation and conventions this implementation
is built against.

## Install

```bash
python3 -m venv /tmp/wv-venv
/tmp/wv-venv/bin/pip install -e ".[dev]"
/tmp/wv-venv/bin/python -m pytest        # 253 tests
```

## Worked examples

Three runnable scripts under `examples/`. Each one solves a problem, prints every
computed eigenvalue next to its closed-form value with the relative error, and
writes a figure into `figures/`.

| script | what it shows | figure |
|---|---|---|
| `examples/01_dirichlet.py` | pure Dirichlet, 1D and 2D: sine modes, the spectrum, degeneracies of the square box, second-order convergence | `figures/01-dirichlet.png` |
| `examples/02_neumann.py` | pure Neumann: the zero mode (`lambda = 0`, flat eigenfunction) and `omega_0 = m` | `figures/02-neumann.png` |
| `examples/03_mixed_intersecting.py` | the intersecting-brane case, Neumann along `x` and Dirichlet along `y`: mode structure and how the spectrum differs from either pure case | `figures/03-mixed-intersecting.png` |

```bash
/tmp/wv-venv/bin/python examples/01_dirichlet.py
/tmp/wv-venv/bin/python examples/02_neumann.py
/tmp/wv-venv/bin/python examples/03_mixed_intersecting.py
```

Measured on the runs committed with these examples: worst relative eigenvalue
error `1.5e-2` (2D Dirichlet, `N = 20`), `1.8e-2` (1D Dirichlet, `N = 40`, mode 6),
`6.8e-3` (mixed 2D, `N = 32`), with observed convergence orders `1.82, 1.92, 1.96,
1.98` (1D) and `1.82, 1.91, 1.95` (2D) against an expected 2.
