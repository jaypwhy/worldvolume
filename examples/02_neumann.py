#!/usr/bin/env python3
"""Worked example 2 -- pure Neumann boundary conditions and the zero mode.

Solves the worldvolume spectral problem

    -Delta f = lambda f,        df/dn = 0 on every boundary,

whose closed-form solution is (``docs/bc-conventions.md`` eq. 3.3)

    lambda_n = (n pi / L)^2,   n = 0, 1, 2, ...
    f_0(x) = 1/sqrt(L)                     (the constant zero mode)
    f_n(x) = sqrt(2/L) cos(n pi x / L)     (n >= 1)

The ``n = 0`` entry is the point of this example.  ``lambda_0 = 0`` exactly, so
it has no relative error to report -- its *absolute* error is quoted instead --
and its eigenfunction is a constant, which is what makes it a zero mode rather
than a mode that happens to be small.

The mass enters only through ``omega^2 = lambda + m^2`` (spec eq. 1.6), so the
zero mode keeps a non-zero frequency ``omega_0 = m``: it is the mode that does
not oscillate in space but still oscillates in time.

Run from the repository root::

    python examples/02_neumann.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from worldvolume import (  # noqa: E402
    Grid1D,
    Neumann,
    neumann_mode,
    solve,
    spectrum_1d,
)

L = 2.0  # length of the interval
N = 40  # interior nodes
M = 1.0  # mass
NMODE = 6  # n = 0 .. 5


def main() -> None:
    C.setup()

    print(__doc__.strip().splitlines()[0])
    print(f"1D: L = {L}, N = {N} interior nodes, m = {M}, modes n = 0 .. {NMODE - 1}")

    grid = Grid1D.uniform(L=L, N=N, bc=Neumann())
    spec = solve(grid, m=M, nmodes=NMODE)
    ana = spectrum_1d(Neumann(), Neumann(), L, NMODE, m=M)

    # ------------------------------------------------------------------ table
    C.heading("1D spectrum: solver vs closed form")
    rows = []
    for i in range(NMODE):
        lam_num, lam_ana = spec.lam[i], ana.lam[i]
        rel = "  --  " if lam_ana == 0.0 else f"{C.rel_error(lam_num, lam_ana):.3e}"
        rows.append(
            [
                i,
                f"{lam_num:.10e}",
                f"{lam_ana:.10e}",
                f"{abs(lam_num - lam_ana):.3e}",
                rel,
                f"{spec.omega[i].real:.10f}",
                f"{ana.omega[i].real:.10f}",
            ]
        )
    C.table(
        ["n", "lambda_num", "lambda_analytic", "|error|", "rel. error",
         "omega_num", "omega_analytic"],
        rows,
        align="> > > > > > >".replace(" ", ""),
    )
    print("\nThe zero mode has lambda_analytic = 0 exactly, so its relative error")
    print("is undefined; the absolute error above is pure round-off.")

    # ------------------------------------------------------- zero-mode checks
    C.heading("Zero mode diagnostics")
    exact0 = neumann_mode(L, 0, grid.x)  # the constant 1/sqrt(L)
    u0 = C.align(spec.mode(0), exact0)  # function values on the full grid
    flat_dev = float(np.max(np.abs(u0 - exact0)))
    print(f"lambda_0 (computed)                    : {spec.lam[0]:.3e}")
    print(f"lambda_0 (exact)                       : 0")
    print(f"max_x |u_0(x) - 1/sqrt(L)|             : {flat_dev:.3e}")
    print(f"1/sqrt(L)                              : {1.0 / np.sqrt(L):.15f}")
    print(f"omega_0 = sqrt(lambda_0 + m^2)         : {spec.omega[0].real:.15f}")
    print(f"m                                      : {M:.15f}")
    print(f"eigenvector spread  max(u_0)-min(u_0)  : {float(u0.max() - u0.min()):.3e}")
    print("\nThe lowest eigenvector is constant to machine precision and its")
    print("eigenvalue is zero to machine precision: a genuine zero mode, not a")
    print("small eigenvalue.  Everything else is spaced by (pi/L)^2 = "
          f"{(np.pi / L) ** 2:.6f}.")

    # ---------------------------------------------- zero mode across the grids
    C.heading("lambda_0 does not shrink with h -- it is round-off, not truncation")
    rows = []
    for n_grid in (10, 20, 40, 80):
        g = Grid1D.uniform(L=L, N=n_grid, bc=Neumann())
        s = solve(g, m=M, nmodes=NMODE)
        u0 = C.align(s.mode(0), np.full(g.n_nodes, 1.0 / np.sqrt(L)))
        dev = float(np.max(np.abs(u0 - 1.0 / np.sqrt(L))))
        rows.append([n_grid, f"{g.h:.6f}", f"{s.lam[0]:.3e}", f"{dev:.3e}"])
    C.table(["N", "h", "lambda_0", "max|u_0 - 1/sqrt(L)|"], rows, align=">><>")

    # ------------------------------------------- convergence of the other modes
    C.heading("Convergence of the non-zero modes 1-4")
    ns = [10, 20, 40, 80, 160]
    hs, errs = [], []
    for n_grid in ns:
        g = Grid1D.uniform(L=L, N=n_grid, bc=Neumann())
        s = solve(g, m=M, nmodes=NMODE)
        a = spectrum_1d(Neumann(), Neumann(), L, NMODE, m=M)
        hs.append(g.h)
        errs.append(float(np.max(C.rel_error(s.lam[1:5], a.lam[1:5]))))
    rates = C.rates(errs)
    rows = []
    for i, n_grid in enumerate(ns):
        rate = f"{rates[i - 1]:.2f}" if i else "-"
        rows.append([n_grid, f"{hs[i]:.6f}", f"{errs[i]:.3e}", rate])
    C.table(["N", "h", "max rel. error", "rate"], rows, align=">><>")
    print(f"\nobserved orders: {', '.join(f'{r:.2f}' for r in rates)}  (expected 2)")
    print("(the zero mode is excluded: it is exact at every N, so it has no")
    print(" error to converge)")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    # (a) the modes, with the flat zero mode on top
    ax = axes[0, 0]
    xx = grid.x
    colours = {0: "#0f766e", 1: C.ANA, 2: "#7c3aed"}
    for n in (0, 1, 2):
        exact = neumann_mode(L, n, xx)
        num = C.align(spec.mode(n), exact)
        ax.plot(
            xx, exact, color=colours[n], lw=3.2 if n == 0 else 1.6,
            label=f"analytic $f_{n}$" if n < 2 else "analytic $f_2$",
        )
        ax.plot(xx[::3], num[::3], "o", ms=5.0, mfc="white", mec=colours[n],
                mew=1.3, label="solver (every 3rd node)" if n == 0 else None)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.annotate(
        "zero mode: constant $1/\\sqrt{L}$, $\\lambda_0 = 0$",
        xy=(0.6 * L, 1.0 / np.sqrt(L)),
        xytext=(0.08 * L, 1.45),
        arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.0),
        color="#0f766e",
        fontsize=9,
    )
    ax.set_xlabel("x   (L = 2)")
    ax.set_ylabel(r"$f_n(x)$")
    ax.set_title("(a) Neumann modes: $n=0$ is flat, $n\\geq1$ are cosines")
    ax.set_ylim(-1.5, 2.25)
    ax.legend(loc="upper center", ncol=3, columnspacing=1.2)

    # (b) the spectrum, with lambda_0 sitting at zero
    ax = axes[0, 1]
    n_idx = np.arange(NMODE)
    ax.plot(n_idx, ana.lam, "s", ms=8, mfc="none", mec=C.ANA, mew=1.5,
            label="analytic $(n\\pi/L)^2$")
    ax.plot(n_idx, spec.lam, "o", ms=5, color=C.NUM, label="solver")
    ax.annotate(
        f"$\\lambda_0$ = {spec.lam[0]:.1e}\n(exact 0)",
        xy=(0, spec.lam[0]),
        xytext=(1.1, 18.0),
        arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9),
        fontsize=9,
        color="0.25",
    )
    ax.set_xlabel("mode number n")
    ax.set_ylabel(r"eigenvalue  $\lambda_n$")
    ax.set_title("(b) spectrum starts at zero, no gap")
    ax.set_ylim(-2, ana.lam.max() * 1.12)
    ax.legend(loc="upper left")

    # (c) the mass shift: omega_0 = m
    ax = axes[1, 0]
    ax.plot(n_idx, ana.omega.real, "s", ms=8, mfc="none", mec=C.ANA, mew=1.5,
            label=r"analytic $\sqrt{\lambda_n + m^2}$")
    ax.plot(n_idx, spec.omega.real, "o", ms=5, color=C.NUM, label="solver")
    ax.axhline(M, color="#0f766e", lw=1.0, ls=":")
    ax.annotate(
        f"$\\omega_0 = m = {spec.omega[0].real:.6f}$",
        xy=(0, spec.omega[0].real),
        xytext=(1.1, 2.6),
        arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.0),
        fontsize=9,
        color="#0f766e",
    )
    ax.set_xlabel("mode number n")
    ax.set_ylabel(r"frequency  $\omega_n$")
    ax.set_title("(c) the zero mode still oscillates in time: $\\omega_0 = m$")
    ax.set_ylim(0, ana.omega.real.max() * 1.1)
    ax.legend(loc="upper left")

    # (d) convergence
    ax = axes[1, 1]
    ax.loglog(hs, errs, "o-", color=C.NUM, ms=5, label="max rel. error, modes 1-4")
    C.slope_guide(ax, hs, errs, 2.0, "slope 2 reference")
    C.log_xticks(ax, hs, "{:.3f}")
    ax.set_xlabel("grid spacing  h = L/(N+1)")
    ax.set_ylabel("max relative eigenvalue error")
    ax.set_title("(d) second-order convergence (zero mode excluded)")
    ax.margins(y=0.15)
    ax.legend(loc="lower right")

    fig.suptitle(
        "Example 2 - pure Neumann: the zero mode is flat, exact, and keeps $\\omega_0 = m$",
        fontsize=12.5,
    )
    fig.tight_layout()
    C.save(fig, "02-neumann.png")


if __name__ == "__main__":
    main()
