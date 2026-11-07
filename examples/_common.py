"""Formatting helpers shared by the worked examples.

Nothing in here does any physics: the numbers all come from
:mod:`worldvolume`.  This module only prints aligned tables, computes
convergence rates and writes figures into ``figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGDIR = REPO_ROOT / "figures"

#: colour for the solver's numbers
NUM = "#1d4ed8"
#: colour for the closed-form result
ANA = "#b91c1c"
#: colour for reference guide lines (slope-2 etc.)
REF = "#64748b"


def setup() -> None:
    """Matplotlib defaults shared by every example figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def heading(text: str) -> None:
    """A section header in the console output."""
    print()
    print(text)
    print("=" * len(text))


def table(headers, rows, align: str | None = None) -> None:
    """Print a simple aligned text table."""
    cells = [[str(c) for c in row] for row in rows]
    widths = []
    for i, h in enumerate(headers):
        widths.append(max([len(str(h))] + [len(r[i]) for r in cells]))
    if align is None:
        align = "<" * len(headers)

    def render(values) -> str:
        return "  ".join(f"{v:{a}{w}}" for v, a, w in zip(values, align, widths))

    print(render([str(h) for h in headers]))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print(render(row))


def rel_error(num, ana):
    """``|num - ana| / |ana|``, elementwise; ``nan`` where ``ana == 0``.

    The zero eigenvalue has no relative error.  Examples that can produce one
    report its absolute error instead and say so.
    """
    num = np.asarray(num, dtype=float)
    ana = np.asarray(ana, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(num - ana) / np.abs(ana)
    return out


def rates(errors) -> list[float]:
    """Observed order of convergence between successive grid refinements.

    ``errors`` must be ordered from coarse to fine with ``h`` halving each
    step, so the rate is ``log2(e_k / e_{k+1})``.
    """
    e = np.asarray(errors, dtype=float)
    return [float(np.log2(e[i] / e[i + 1])) for i in range(len(e) - 1)]


def align(u_num: np.ndarray, u_ana: np.ndarray) -> np.ndarray:
    """Flip the sign of a computed eigenvector to match the analytic one.

    Eigenvectors are defined up to an overall sign, which the eigensolver
    picks arbitrarily; every mode comparison has to fix it first.
    """
    return u_num if float(np.sum(u_num * u_ana)) >= 0.0 else -u_num


def save(fig, name: str) -> Path:
    """Write ``fig`` to ``figures/<name>`` and report the path."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024.0
    print(f"\nfigure written: figures/{name}  ({size_kb:.0f} kB)")
    return path


def log_xticks(ax, values, fmt: str = "{:.3f}") -> None:
    """Label a log x-axis at exactly ``values``, with no minor labels.

    Matplotlib's automatic log locator puts far too many labelled ticks on the
    narrow ranges a convergence plot spans; this pins one tick per grid size.
    """
    from matplotlib.ticker import (
        FixedFormatter,
        FixedLocator,
        NullFormatter,
        NullLocator,
    )

    vals = np.asarray(values, dtype=float)
    ax.xaxis.set_major_locator(FixedLocator(vals))
    ax.xaxis.set_major_formatter(FixedFormatter([fmt.format(v) for v in vals]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def slope_guide(ax, x, y, exponent: float, label: str) -> None:
    """Draw a ``y ~ x**exponent`` guide line anchored at ``(x[0], y[0])``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax.plot(
        x,
        y[0] * (x / x[0]) ** exponent,
        "--",
        color=REF,
        linewidth=1.1,
        label=label,
    )
