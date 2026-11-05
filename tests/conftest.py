"""Shared helpers for the worldvolume test suite.

The reference values are the printed numbers of spec section 5
(``docs/bc-conventions.md``), which are quoted to **12 significant figures**.
:func:`assert_sig_figs` is the strict reading of that claim: the computed value
must agree with the printed one to within half a unit in the 12th significant
digit.  It is deliberately not a blanket relative tolerance -- a relative
tolerance of ``1e-12`` is *looser* than 12 significant figures for a value whose
leading digit is 1, and *tighter* for one whose leading digit is 9.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["assert_sig_figs", "assert_rel", "rel_err", "rate"]


def assert_sig_figs(value, printed, sig: int = 12, label: str = "") -> None:
    """Assert ``value`` agrees with the printed reference to ``sig`` sig figs.

    ``printed`` may be a string (as it appears in the spec table) or a float.
    """
    value = float(value)
    expected = float(printed)
    tag = f" [{label}]" if label else ""
    if expected == 0.0:
        assert value == 0.0, f"expected exactly 0, got {value!r}{tag}"
        return
    exponent = math.floor(math.log10(abs(expected)))
    tol = 0.5 * 10.0 ** (exponent - sig + 1)
    # a hair of slack so that a value sitting exactly on a rounding boundary is
    # not rejected for a last-bit reason
    tol *= 1.0 + 1e-9
    assert abs(value - expected) <= tol, (
        f"value {value!r} differs from the {sig}-significant-figure reference "
        f"{printed!r} by {abs(value - expected):.3e} > {tol:.3e}{tag}"
    )


def rel_err(computed, exact) -> float:
    """Relative error ``(computed - exact) / exact``."""
    return float((computed - exact) / exact)


def assert_rel(computed, exact, rtol: float = 1e-12, label: str = "") -> None:
    """Assert ``|computed - exact| <= rtol * |exact|``."""
    computed = np.asarray(computed, dtype=float)
    exact = np.asarray(exact, dtype=float)
    dev = np.abs(computed - exact)
    scale = np.maximum(np.abs(exact), 1e-300)
    worst = float(np.max(dev / scale)) if dev.size else 0.0
    tag = f" [{label}]" if label else ""
    assert worst <= rtol, f"relative deviation {worst:.3e} > {rtol:.3e}{tag}"


def rate(errors) -> list[float]:
    """Observed convergence rates ``log2(e_i / e_{i+1})`` for a halving grid."""
    errors = np.asarray(errors, dtype=float)
    return [float(r) for r in np.log2(errors[:-1] / errors[1:])]
