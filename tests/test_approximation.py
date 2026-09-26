"""Self-checks for apps/approximation/logic.py. Run: uv run python tests/test_approximation.py"""

import sys
import tempfile
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


def test_load_intervals_reads_the_kind_column():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "iv.txt"
        tmp.write_text("0 5 min\n10 20 max\n30 40\n")  # Splitter v5 writes 'start end kind'; older files have none
        assert [iv.kind for iv in logic.load_intervals(tmp)] == ["min", "max", None]


def test_auto_times_an_eclipse_on_a_slope():
    mags = eclipse(T) + 400 * (T - 1.5)  # spots or a trend tilt the baseline
    s, e = int(np.searchsorted(T, 1.44)), int(np.searchsorted(T, 1.56))
    auto = logic.approximate_all(T, mags, [Interval(s, e, "min")], {"method": "auto"}, wings=0.5)[0]
    plain = logic.approximate_all(T, mags, [Interval(s, e, "min")], {"method": "brat", "params": None}, wings=0.5)[0]
    assert auto.method == "brat" and abs(auto.t0 - 1.5) < 1 / 1440, auto
    assert abs(auto.t0 - 1.5) < 0.5 * abs(plain.t0 - 1.5), (auto.t0, plain.t0)


def test_auto_uses_a_polynomial_for_maxima_and_when_the_eclipse_fit_fails():
    mags = 200 * np.cos(2 * np.pi * T / 0.8)  # magnitude minima = brightness maxima at 0.4, 1.2, ...
    s, e = int(np.searchsorted(T, 1.1)), int(np.searchsorted(T, 1.3))
    res = logic.approximate_all(T, mags, [Interval(s, e, "max"), Interval(100, 104, "min")], {"method": "auto"})
    assert res[0].method == "poly" and abs(res[0].t0 - 1.2) < 1 / 1440, res[0]
    assert res[1].method == "poly", res[1]  # 5 points, no wings: too few for the 6 parameters of the eclipse profile


def test_auto_tells_a_minimum_from_the_data_without_a_kind():
    mags = eclipse(T)
    s, e = int(np.searchsorted(T, 1.44)), int(np.searchsorted(T, 1.56))
    assert logic.approximate_all(T, mags, [Interval(s, e)], {"method": "auto"}, wings=0.5)[0].method == "brat"


def test_auto_minimum_of_a_contact_binary_with_wide_wings():
    P = 0.35  # EW: brightness minima (magnitude maxima) at phases 0 and 0.5, maxima in the wings
    mags = 300 * np.cos(4 * np.pi * T / P) + rng.normal(0, 3, T.size)
    t_min = 4 * P  # 1.4
    s, e = int(np.searchsorted(T, t_min - 0.03)), int(np.searchsorted(T, t_min + 0.03))
    for wings in (0.5, 1.0, 2.0):
        r = logic.approximate_all(T, mags, [Interval(s, e, "min")], {"method": "auto"}, wings=wings)[0]
        assert abs(r.t0 - t_min) < 2 / 1440 and r.kind == "min", (wings, r.method, r.t0, r.kind)


def test_kind_is_brightness_for_every_method():
    mags = eclipse(T)  # an eclipse: brightness minimum, astrolab kind "min" like the polynomial gives
    s, e = int(np.searchsorted(T, 1.44)), int(np.searchsorted(T, 1.56))
    for choice in ("auto", {"method": "brat", "params": None}, {"method": "exponential", "params": None}, {"method": "auto"}):
        r = logic.approximate_all(T, mags, [Interval(s, e, "min")], choice, wings=0.5 if choice != "auto" else 0.0)[0]
        assert r.kind == "min", (choice, r.method, r.kind)


def test_bic_keeps_residual_freedom_on_short_intervals():
    x = np.linspace(-0.01, 0.01, 8)
    y = 3e4 * x**2 + rng.normal(0, 0.5, x.size)
    assert logic._pick_order(x, y, "auto", 7)[1] <= 5  # order N-1 interpolates: SSE 0, BIC -inf


def test_auto_respects_zero_wings():
    mags = eclipse(T)
    s, e = int(np.searchsorted(T, 1.44)), int(np.searchsorted(T, 1.56))
    assert logic.approximate_all(T, mags, [Interval(s, e, "min")], {"method": "auto"}, wings=0.0)[0].wings == 0.0


def test_auto_rejects_an_eclipse_fit_on_a_spike():
    mags = rng.normal(0, 3, T.size)  # no eclipse, one 6-min spike: the eclipse profile would lock onto it
    s = int(np.searchsorted(T, 1.47))
    mags[s + 20 : s + 23] += 80
    res = logic.approximate_all(T, mags, [Interval(s, s + 43, "min")], {"method": "auto"}, wings=0.5)
    assert all(r.method != "brat" for r in res), [(r.method, r.coefficients) for r in res]


def test_a_fit_of_the_other_kind_is_a_failure():
    P = 0.8
    mags = 200 * np.cos(2 * np.pi * T / P) + rng.normal(0, 2, T.size)  # magnitude minima = brightness maxima at 0.4, 1.2
    s, e = int(np.searchsorted(T, 1.1)), int(np.searchsorted(T, 1.3))
    errors = {}
    res = logic.approximate_all(T, mags, [Interval(s, e, "max")], {"method": "exponential", "params": None}, errors=errors)
    assert res == [] and 0 in errors, (res, errors)  # a bump model cannot time a brightness maximum: no result
    res = logic.approximate_all(T, mags, [Interval(s, e, "min")], {"method": "polynomial", "order": "auto"}, errors=errors)
    assert res == [] and "brightness max" in errors[0], (res, errors)  # the file says minimum, the data a maximum


def test_sigma_t0_matches_the_scatter_of_t0():
    """The fit's error of t0 agrees with the scatter of t0 over noise realisations (white noise)."""
    x = T[(T > 1.3) & (T < 1.7)]
    hump = lambda t: -40 * np.exp(-(((t - 1.5) / 0.15) ** 2))  # a brightness maximum, in magnitudes
    cases = {
        "brat": (lambda: eclipse(x), {"method": "brat", "slope": True, "dip": True}),
        "poly": (lambda: hump(x) + rng.normal(0, 2, x.size), "auto"),
    }
    wide = T[(T > 1.35) & (T < 1.65)]
    cases |= {
        "sym": (lambda: hump(x) + rng.normal(0, 2, x.size), "sym"),
        "wsl": (lambda: logic.wsl_model(x, 40, -900, 1.5, 0.02) + rng.normal(0, 2, x.size), "wsl"),
        "apar": (lambda: logic.apar_model(wide, 0, 30000, 1.5, 0.03, 0.08) + rng.normal(0, 2, wide.size), "apar"),
    }
    for name, (data, choice) in cases.items():
        xs = wide if name == "apar" else x
        fits = [logic.fit_extremum(xs, data(), choice) for _ in range(60)]
        t0 = np.array([f.t0 for f in fits])
        sigma = np.array([f.sigma_t0 for f in fits])
        assert np.isfinite(sigma).all() and (sigma > 0).all(), name
        ratio = np.median(sigma) / np.std(t0)
        assert 0.6 < ratio < 1.6, (name, ratio, np.median(sigma), np.std(t0))


def test_near_extremum_functions():
    """MAVKA's functions: the moment and kind of a flat maximum between eclipses (with a spot bump that pulls
    a polynomial off), a total eclipse and an asymmetric maximum; drawn by evaluate()."""
    x = np.linspace(0, 1, 400)
    dips = lambda t: 40 * (np.exp(-((t / 0.06) ** 2)) + np.exp(-(((t - 1) / 0.06) ** 2)))
    y = dips(x) - 3 * np.exp(-(((x - 0.3) / 0.1) ** 2)) + rng.normal(0, 0.3, x.size)  # magnitudes
    sym, poly = logic.fit_extremum(x, y, "sym"), logic.fit_extremum(x, y, "auto")
    assert sym.kind == "max" and abs(sym.t0 - 0.5) < 0.01 and abs(poly.t0 - 0.5) > 0.1, (sym.t0, poly.t0)
    assert abs(np.sum((y - logic.evaluate(sym, x)) ** 2) - sym.sse) < 1e-6 * sym.sse  # drawn as fitted
    x = T[(T > 1.4) & (T < 1.6)]
    wsl = logic.fit_extremum(x, logic.wsl_model(x, 40, -900, 1.5, 0.02) + rng.normal(0, 2, x.size), "wsl")
    assert wsl.kind == "min" and abs(wsl.t0 - 1.5) < 3 * wsl.sigma_t0 < 6 / 1440, wsl
    assert abs(wsl.coefficients[3] - 0.02) < 4e-3 and wsl.x_range[0] > 1.41 and wsl.x_range[1] < 1.59, wsl  # the core
    x = T[(T > 1.35) & (T < 1.65)]
    ap = logic.fit_extremum(x, logic.apar_model(x, 0, 30000, 1.5, 0.03, 0.08) + rng.normal(0, 2, x.size), "apar")
    assert ap.kind == "max" and abs(ap.t0 - 1.5) < 3 * ap.sigma_t0 < 3 / 1440, ap
    s, e = int(np.searchsorted(T, 1.4)), int(np.searchsorted(T, 1.6))
    res = logic.approximate_all(T, logic.wsl_model(T, 40, -900, 1.5, 0.02), [Interval(s, e, "min")], "wsl")
    assert res[0].method == "wsl" and abs(res[0].t0 - 1.5) < 1e-5, res


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
