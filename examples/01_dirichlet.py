#!/usr/bin/env python3
"""Worked example 1 -- pure Dirichlet boundary conditions, 1D and 2D.

Solves the worldvolume spectral problem

    -Delta f = lambda f,        f = 0 on every boundary,

whose closed-form solution is (``docs/bc-conventions.md`` eqs. 3.2, 3.20)

    1D:  lambda_n = (n pi / L)^2,  f_n(x) = sqrt(2/L) sin(n pi x / L),  n >= 1
    2D:  lambda_{n1 n2} = k_{n1}^2 + k_{n2}^2,  f = f_{n1}(x) f_{n2}(y)

Every computed eigenvalue is printed next to its analytic value with the
relative error, then the error is tracked over a sequence of grids to show
second-order convergence.  The 2D part also exhibits the degeneracies of the
square box: the solver must return level 2 *twice*.

Run from the repository root::

    python examples/01_dirichlet.py
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
    Grid1D,
    dirichlet_mode,
    solve,
    spectrum_1d,
    spectrum_2d,
)

L = 1.0  # length of the interval
N = 40  # interior nodes
M = 0.5  # mass (enters only through omega^2 = lambda + m^2)
NMODE = 6


def main() -> None:
    C.setup()

    print(__doc__.strip().splitlines()[0])
    print(f"1D: L = {L}, N = {N} interior nodes, m = {M}, lowest {NMODE} modes")

    # ------------------------------------------------------------------ 1D
    grid = Grid1D.uniform(L=L, N=N, bc=Dirichlet())
    spec = solve(grid, m=M, nmodes=NMODE)
    ana = spectrum_1d(Dirichlet(), Dirichlet(), L, NMODE, m=M)

    C.heading("1D spectrum: solver vs closed form")
    rows = []
    for i in range(NMODE):
        rows.append(
            [
                i + 1,
                f"{spec.lam[i]:.10f}",
                f"{ana.lam[i]:.10f}",
                f"{C.rel_error(spec.lam[i], ana.lam[i]):.3e}",
                f"{spec.omega[i].real:.10f}",
                f"{ana.omega[i].real:.10f}",
            ]
        )
    C.table(
        ["n", "lambda_num", "lambda_analytic", "rel. error", "omega_num", "omega_analytic"],
        rows,
        align=">><<><",
    )
    worst = float(np.max(C.rel_error(spec.lam, ana.lam)))
    print(f"\nworst relative eigenvalue error: {worst:.3e}")

    # --- the leading-order prediction, which has no free constant -----------
    # lambda_n^(h) = lambda_n [1 - (k_n h)^2 / 12 + O(h^4)]  (spec 4.15)
    C.heading("Leading-order error law  rel.err ~= -(k_n h)^2 / 12  (spec 4.15)")
    rows = []
    for n_grid in (40, 80):
        g = Grid1D.uniform(L=L, N=n_grid, bc=Dirichlet())
        s = solve(g, m=M, nmodes=NMODE)
        a = spectrum_1d(Dirichlet(), Dirichlet(), L, NMODE, m=M)
        for n in (1, 2, 3):
            measured = float(C.rel_error(s.lam[n - 1], a.lam[n - 1]))
            k = n * np.pi / L
            predicted = -((k * g.h) ** 2) / 12.0
            rows.append(
                [n, n_grid, f"{g.h:.6f}", f"{measured:.4e}", f"{predicted:.4e}",
                 f"{measured / predicted:.4f}"]
            )
    C.table(
        ["n", "N", "h", "measured rel.err", "-(k h)^2/12", "ratio"],
        rows,
        align=">>><<>",
    )

    # --- convergence: the error must fall like h^2 --------------------------
    C.heading("Convergence of the 1D spectrum (modes 1-4)")
    ns = [10, 20, 40, 80, 160]
    hs, errs = [], []
    for n_grid in ns:
        g = Grid1D.uniform(L=L, N=n_grid, bc=Dirichlet())
        s = solve(g, m=M, nmodes=NMODE)
        a = spectrum_1d(Dirichlet(), Dirichlet(), L, NMODE, m=M)
        hs.append(g.h)
        errs.append(float(np.max(C.rel_error(s.lam[:4], a.lam[:4]))))
    rates = C.rates(errs)
    rows = []
    for i, n_grid in enumerate(ns):
        rate = f"{rates[i - 1]:.2f}" if i else "-"
        rows.append([n_grid, f"{hs[i]:.6f}", f"{errs[i]:.3e}", rate])
    C.table(["N", "h", "max rel. error", "rate"], rows, align=">><>")
    print(f"\nobserved orders: {', '.join(f'{r:.2f}' for r in rates)}  (expected 2)")

    # ------------------------------------------------------------------ 2D
    L2, N2, NMODE2 = 1.0, 20, 8
    side = Grid1D.uniform(L=L2, N=N2, bc=Dirichlet())
    box = Box2D(x=side, y=side)
    spec2 = solve(box, m=0.0, nmodes=NMODE2)
    ana2 = spectrum_2d(
        Dirichlet(), Dirichlet(), L2, Dirichlet(), Dirichlet(), L2, NMODE2, m=0.0
    )

    C.heading(f"2D spectrum: L1 = L2 = {L2}, N = {N2} per direction, Dirichlet everywhere")
    rows = []
    for i in range(NMODE2):
        label = ana2.labels[i]
        # multiplicity: how many of the 8 lowest levels share this eigenvalue
        mult = int(np.sum(np.abs(ana2.lam - ana2.lam[i]) < 1e-9))
        rows.append(
            [
                i + 1,
                f"({label[0]},{label[1]})",
                f"{spec2.lam[i]:.8f}",
                f"{ana2.lam[i]:.8f}",
                f"{C.rel_error(spec2.lam[i], ana2.lam[i]):.3e}",
                mult,
            ]
        )
    C.table(
        ["level", "(n1,n2)", "lambda_num", "lambda_analytic", "rel. error", "mult."],
        rows,
        align=">^<<<>",
    )
    print("\nlevels 2 and 3 are the degenerate pair (1,2)/(2,1): the solver")
    print("returns the eigenvalue twice rather than deduplicating it.")
    print(f"worst relative eigenvalue error: {float(np.max(C.rel_error(spec2.lam, ana2.lam))):.3e}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))

    # (a) the sine modes themselves
    ax = axes[0, 0]
    xx = grid.x
    for n in (1, 2, 3):
        exact = dirichlet_mode(L, n, xx)
        num = C.align(spec.mode(n - 1), exact)
        ax.plot(xx, exact, color=C.ANA, lw=1.6, label="analytic" if n == 1 else None)
        ax.plot(
            xx[::3], num[::3], "o", ms=5.0, mfc="white", mec=C.NUM, mew=1.3,
            label="solver (every 3rd node)" if n == 1 else None,
        )
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("x   (L = 1)")
    ax.set_ylabel(r"$f_n(x)$")
    ax.set_title("(a) Dirichlet sine modes, $n=1,2,3$")
    ax.legend(loc="lower left")

    # (b) the 1D spectrum
    ax = axes[0, 1]
    n_idx = np.arange(1, NMODE + 1)
    ax.semilogy(n_idx, ana.lam, "-", color=C.ANA, label="analytic $(n\\pi/L)^2$")
    ax.semilogy(n_idx, spec.lam, "o", ms=5, mfc="none", mec=C.NUM, mew=1.2,
                label="solver")
    ax.set_xlabel("mode number n")
    ax.set_ylabel(r"eigenvalue  $\lambda_n$")
    ax.set_title("(b) 1D spectrum, N = 40")
    ax.legend(loc="upper left")

    # (c) convergence
    ax = axes[0, 2]
    ax.loglog(hs, errs, "o-", color=C.NUM, ms=5, label="max rel. error, modes 1-4")
    C.slope_guide(ax, hs, errs, 2.0, "slope 2 reference")
    ax.set_xlabel("grid spacing  h = L/(N+1)")
    ax.set_ylabel("max relative eigenvalue error")
    ax.set_title("(c) second-order convergence")
    ax.margins(y=0.15)
    ax.legend(loc="lower right")

    # (d) 2D spectrum with degeneracies
    ax = axes[1, 0]
    lvl = np.arange(1, NMODE2 + 1)
    ax.plot(lvl, ana2.lam, "s", ms=7, mfc="none", mec=C.ANA, mew=1.4,
            label="analytic")
    ax.plot(lvl, spec2.lam, "o", ms=4, color=C.NUM, label="solver")
    for i in range(NMODE2):
        ax.annotate(
            f"({ana2.labels[i][0]},{ana2.labels[i][1]})",
            (lvl[i], ana2.lam[i]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7.5,
            color="0.35",
        )
    ax.set_xlabel("level")
    ax.set_ylabel(r"eigenvalue  $\lambda$")
    ax.set_title("(d) 2D spectrum, $L_1=L_2=1$, N = 20")
    ax.set_ylim(0, ana2.lam.max() * 1.15)
    ax.legend(loc="upper left")

    # (e) a 2D eigenfunction: level (2,2), non-degenerate
    X, Y = box.mesh()
    i22 = int(np.argmin(np.abs(spec2.lam - 8.0 * np.pi**2 / L2**2)))
    label = ana2.labels[i22]
    exact2 = np.outer(
        dirichlet_mode(L2, label[0], box.x.x), dirichlet_mode(L2, label[1], box.y.x)
    )
    num2 = C.align(spec2.mode(i22), exact2)
    err2 = float(np.max(np.abs(num2 - exact2)))

    ax = axes[1, 1]
    lim = float(np.max(np.abs(num2)))
    mesh = ax.pcolormesh(X, Y, num2, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.contour(X, Y, exact2, levels=[0.0], colors="k", linewidths=0.8,
               linestyles="--")
    fig.colorbar(mesh, ax=ax, label=r"$f(x,y)$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"(e) mode ({label[0]},{label[1]}): solver, dashed = analytic nodes")
    ax.grid(False)

    # (f) a slice through that mode
    ax = axes[1, 2]
    j = int(np.argmin(np.abs(box.y.x - 0.25 * L2)))
    ax.plot(X[:, j], exact2[:, j], "-", color=C.ANA, lw=1.8, label="analytic")
    ax.plot(X[:, j], num2[:, j], "o", ms=4.5, mfc="none", mec=C.NUM, mew=1.2,
            label="solver")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$f(x, y_0)$")
    ax.set_title(f"(f) slice at y = {box.y.x[j]:.3f}")
    ax.legend(loc="upper right")
    ax.text(
        0.03, 0.04,
        f"max$|f_{{num}}-f_{{ana}}|$ = {err2:.1e}",
        transform=ax.transAxes, va="bottom", fontsize=9,
    )

    fig.suptitle(
        "Example 1 - pure Dirichlet: sine modes, the spectrum, and its second-order convergence",
        fontsize=12.5,
    )
    fig.tight_layout()
    C.save(fig, "01-dirichlet.png")


if __name__ == "__main__":
    main()
