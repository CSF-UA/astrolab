"""Self-checks for apps/approximation/logic.py. Run: uv run python tests/test_approximation.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.approximation import logic  # noqa: E402
from apps.approximation.logic import Interval  # noqa: E402

rng = np.random.default_rng(0)
T = np.arange(0, 3, 2 / 1440)


def eclipse(t, t0=1.5, depth=300.0, d=0.03):
    """Algol-like minimum in magnitudes: flat baseline, eclipse around t0."""
    return depth * np.exp(1 - np.cosh((t - t0) / d)) + rng.normal(0, 2, t.size)


def test_auto_order_by_bic():
    x = np.linspace(-0.05, 0.05, 80)
    y = 2e5 * x**3 - 3e3 * x**2 + 5 * x + rng.normal(0, 0.5, x.size)  # cubic + noise
    _, order, _ = logic._pick_order(x, y, "auto", 10)
    assert order <= 5, order  # min-SSE always took 10


def test_interval_includes_last_point():
    mags = eclipse(T)
    s = int(np.searchsorted(T, 1.5))
    res = logic.approximate_all(T, mags, [Interval(s - 1, s + 2)], "auto")  # 4 points, end inclusive
    assert len(res) == 1 and res[0].order == 3, res


def test_wings_extend_the_fitted_segment():
    s, e = int(np.searchsorted(T, 1.45)), int(np.searchsorted(T, 1.55))
    seg = logic.segment(T, Interval(s, e), 0.5)
    assert abs(T[seg.start] - 1.40) < 0.002 and abs(T[seg.stop - 1] - 1.60) < 0.002, (T[seg.start], T[seg.stop - 1])
    assert logic.segment(T, Interval(s, e), 0.0) == slice(s, e + 1)


def test_brat_with_wings_finds_the_minimum():
    mags = eclipse(T)
    s, e = int(np.searchsorted(T, 1.44)), int(np.searchsorted(T, 1.56))
    res = logic.approximate_all(T, mags, [Interval(s, e)], {"method": "brat", "params": None}, wings=0.5)
    assert abs(res[0].t0 - 1.5) < 2 / 1440 and res[0].wings == 0.5, res
    assert abs(res[0].y_at_t0 - 300) < 10, res[0].y_at_t0  # marker at the eclipse bottom, not at the baseline


def test_failed_interval_is_skipped():
    mags = eclipse(T)
    s = int(np.searchsorted(T, 1.44))
    good = Interval(s, s + 100)
    res = logic.approximate_all(T, mags, [good, Interval(10, 11), good], "auto")  # middle one has 2 points
    assert [r.index for r in res] == [0, 2], [r.index for r in res]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
