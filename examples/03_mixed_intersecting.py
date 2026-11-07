#!/usr/bin/env python3
"""Worked example 3 -- the intersecting-brane case: Neumann in x, Dirichlet in y.

This is the headline case.  A single worldvolume carries two different kinds of
boundary condition at once::

    x:  Neumann    dphi/dx = 0  at x = 0 and x = L1     (along the brane)
    y:  Dirichlet  phi = 0      at y = 0 and y = L2     (transverse to it)

The closed-form solution factorises (``docs/bc-conventions.md`` eq. 3.20)::

    f_{n1 n2}(x, y) = f_{n1}(x) f_{n2}(y)
    lambda_{n1 n2}  = k_{n1}^2 + k_{n2}^2
    k_{n1} = n1 pi / L1      n1 = 0, 1, 2, ...   (Neumann: n1 = 0 exists)
    k_{n2} = n2 pi / L2      n2 = 1, 2, 3, ...   (Dirichlet: no n2 = 0)

The mode labels do not range over the same set in the two directions, and that
is exactly what "mixed" means: the ground state of the box is *not* the
constant function.  Here it is ``f_{0,1}``, constant along the brane and a
half-wave transverse to it, with ``lambda_{01} = (pi/L2)^2``.  There is no zero
mode at all -- unlike the all-Neumann box, which has one -- and the spectrum
sits strictly below the all-Dirichlet box, whose ground state costs an extra
``(pi/L1)^2`` for varying in x.

Run from the repository root::

    python examples/03_mixed_intersecting.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from worldvolume import (  # noqa: E402
    Box2D,
    Dirichlet,
    Neumann,
    dirichlet_mode,
    neumann_mode,
    solve,
    spectrum_2d,
)

L1 = 1.0  # extent along the brane direction x
L2 = 1.5  # extent transverse to it, y
N1 = 32  # interior nodes in x
N2 = 32  # interior nodes in y
M = 0.3  # mass
NMODE = 8


def analytic_mode(label, X, Y):
    """The normalised product eigenfunction for label ``(n1, n2)``."""
    return np.outer(
        neumann_mode(L1, label[0], X[:, 0]),
        dirichlet_mode(L2, label[1], Y[0, :]),
    )


def main() -> None:
    C.setup()

    print(__doc__.strip().splitlines()[0])
    print(
        f"2D: L_x = {L1} (Neumann), L_y = {L2} (Dirichlet), "
        f"N = {N1}x{N2} interior nodes, m = {M}, lowest {NMODE} levels"
    )

    box = Box2D.from_lengths(
        L1=L1, N1=N1, bc_x=Neumann(),
        L2=L2, N2=N2, bc_y=Dirichlet(),
    )
    spec = solve(box, m=M, nmodes=NMODE)
    ana = spectrum_2d(
        Neumann(), Neumann(), L1,
        Dirichlet(), Dirichlet(), L2,
        NMODE, m=M,
    )

    X, Y = box.mesh()

    # --------------------------------------------------------- spectrum table
    C.heading("2D mixed spectrum: solver vs closed form")
    rows = []
    for i in range(NMODE):
        label = ana.labels[i]
        rows.append(
            [
                i + 1,
                f"({label[0]},{label[1]})",
                f"{spec.lam[i]:.10f}",
                f"{ana.lam[i]:.10f}",
                f"{C.rel_error(spec.lam[i], ana.lam[i]):.3e}",
                f"{spec.omega[i].real:.8f}",
                f"{ana.omega[i].real:.8f}",
            ]
        )
    C.table(
        ["level", "(n1,n2)", "lambda_num", "lambda_analytic", "rel. error",
         "omega_num", "omega_analytic"],
        rows,
        align=">^<<<>>",
    )
    worst = float(np.max(C.rel_error(spec.lam, ana.lam)))
    print(f"\nworst relative eigenvalue error: {worst:.3e}")
    print("note the ordering is by lambda, not lexicographic in (n1,n2):")
    print("level 5 is (0,3) and level 6 is (2,1).")

    # ------------------------------------------------- eigenfunction residuals
    C.heading("Eigenfunction residual  max|f_solver - f_analytic|")
    rows = []
    for i in range(NMODE):
        label = ana.labels[i]
        exact = analytic_mode(label, X, Y)
        num = C.align(spec.mode(i), exact)
        rows.append(
            [i + 1, f"({label[0]},{label[1]})", f"{float(np.max(np.abs(num - exact))):.3e}"]
        )
    C.table(["level", "(n1,n2)", "max abs. deviation"], rows, align=">^>")
    print("\n(levels are non-degenerate here, so each eigenvector is unique up")
    print(" to an overall sign, which is fixed before the comparison)")

    # --------------------------------------------- comparison with pure cases
    C.heading("The same box under the other three boundary-condition choices")
    cases = [
        ("x N, y D  (this example)", Neumann(), Neumann(), Dirichlet(), Dirichlet()),
        ("x D, y D  (all Dirichlet)", Dirichlet(), Dirichlet(), Dirichlet(), Dirichlet()),
        ("x N, y N  (all Neumann)", Neumann(), Neumann(), Neumann(), Neumann()),
        ("x D, y N", Dirichlet(), Dirichlet(), Neumann(), Neumann()),
    ]
    spectra = {}
    rows = []
    for name, x0, x1, y0, y1 in cases:
        s = spectrum_2d(x0, x1, L1, y0, y1, L2, 6, m=M)
        spectra[name] = s
        rows.append([name] + [f"{v:.6f}" for v in s.lam])
    C.table(["boundary conditions", "lambda_1", "lambda_2", "lambda_3",
             "lambda_4", "lambda_5", "lambda_6"], rows)
    print("\nall-Neumann starts at lambda = 0 (the constant zero mode);")
    print("all-Dirichlet starts at (pi/L1)^2 + (pi/L2)^2 = "
          f"{np.pi**2 / L1**2 + np.pi**2 / L2**2:.6f}, because it must vary in both")
    print("directions; the mixed box starts at (pi/L2)^2 = "
          f"{np.pi**2 / L2**2:.6f} -- it varies only where the brane does not extend.")

    # ------------------------------------------------------------ convergence
    C.heading("Convergence of the mixed 2D spectrum (lowest 4 levels)")
    ns = [8, 16, 32, 64]
    hs, errs = [], []
    for n in ns:
        b = Box2D.from_lengths(
            L1=L1, N1=n, bc_x=Neumann(), L2=L2, N2=n, bc_y=Dirichlet()
        )
        s = solve(b, m=M, nmodes=NMODE)
        a = spectrum_2d(
            Neumann(), Neumann(), L1, Dirichlet(), Dirichlet(), L2, NMODE, m=M
        )
        hs.append(b.x.h)
        errs.append(float(np.max(C.rel_error(s.lam[:4], a.lam[:4]))))
    rates = C.rates(errs)
    rows = []
    for i, n in enumerate(ns):
        rate = f"{rates[i - 1]:.2f}" if i else "-"
        rows.append([n, f"{hs[i]:.6f}", f"{errs[i]:.3e}", rate])
    C.table(["N (per direction)", "h = L_x/(N+1)", "max rel. error", "rate"],
            rows, align=">><>")
    print(f"\nobserved orders: {', '.join(f'{r:.2f}' for r in rates)}  (expected 2)")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))

    # (a) the ground state: constant along x, half-wave in y
    for ax, level, note in (
        (axes[0, 0], 0, "constant along x, one half-wave in y"),
        (axes[0, 1], 1, "one node along x, one half-wave in y"),
    ):
        label = ana.labels[level]
        exact = analytic_mode(label, X, Y)
        num = C.align(spec.mode(level), exact)
        lim = float(np.max(np.abs(num)))
        mesh = ax.pcolormesh(X, Y, num, shading="auto", cmap="RdBu_r",
                             vmin=-lim, vmax=lim)
        ax.contour(X, Y, exact, levels=[0.0], colors="k", linewidths=0.8,
                   linestyles="--")
        fig.colorbar(mesh, ax=ax, label=r"$f(x,y)$")
        ax.set_xlabel("x   (Neumann: along the brane)")
        ax.set_ylabel("y   (Dirichlet: transverse)")
        ax.set_title(
            f"({'ab'[level]}) level {level + 1}, $(n_1,n_2)=({label[0]},{label[1]})$, "
            f"$\\lambda={spec.lam[level]:.3f}$\n{note}",
            fontsize=10.5,
        )
        ax.grid(False)

    # (c) the spectrum of all four boundary-condition choices
    ax = axes[1, 0]
    markers = {"x N, y D  (this example)": "o", "x D, y D  (all Dirichlet)": "s",
               "x N, y N  (all Neumann)": "^", "x D, y N": "D"}
    colours = {"x N, y D  (this example)": C.NUM,
               "x D, y D  (all Dirichlet)": C.ANA,
               "x N, y N  (all Neumann)": "#0f766e",
               "x D, y N": "#a16207"}
    for name, s in spectra.items():
        ax.plot(np.arange(1, 7), s.lam, marker=markers[name], ms=6,
                color=colours[name], lw=1.3, label=name)
    ax.set_xlabel("level")
    ax.set_ylabel(r"eigenvalue  $\lambda$")
    ax.set_title("(c) the mixed spectrum is neither pure case")
    ax.legend(loc="upper left", fontsize=8.5)

    # (d) convergence
    ax = axes[1, 1]
    ax.loglog(hs, errs, "o-", color=C.NUM, ms=5, label="max rel. error, levels 1-4")
    C.slope_guide(ax, hs, errs, 2.0, "slope 2 reference")
    C.log_xticks(ax, hs, "{:.3f}")
    ax.set_xlabel("grid spacing  h = L_x/(N+1)")
    ax.set_ylabel("max relative eigenvalue error")
    ax.set_title("(d) second-order convergence in 2D")
    ax.margins(y=0.18)
    ax.legend(loc="lower right")

    fig.suptitle(
        "Example 3 - intersecting brane: Neumann along x, Dirichlet along y",
        fontsize=12.5,
    )
    fig.tight_layout()
    C.save(fig, "03-mixed-intersecting.png")


if __name__ == "__main__":
    main()
