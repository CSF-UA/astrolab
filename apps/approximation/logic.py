from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
from scipy.optimize import curve_fit, minimize_scalar

import matplotlib.pyplot as plt
import numpy as np


AUTO_ORDERS = list(range(3, 11))


@dataclass
class Interval:
    start: int
    end: int
    kind: Optional[str] = None  # Splitter's third column: "min" / "max" brightness; None if the file has none


@dataclass
class FitResult:
    index: int
    interval: Interval
    order: int
    t0: float
    kind: str
    coefficients: np.ndarray
    x_mean: float
    sse: float
    y_at_t0: float
    method: str = "poly"
    wings: float = 0.0
    sigma_t0: float = float("nan")  # 1-sigma error of t0 from the fit (inf when t0 is not a turning point)
    x_range: Optional[Tuple[float, float]] = None  # the times fitted, when only a part of the interval was


@dataclass
class BratParams:
    c0: Optional[float] = None
    c1: Optional[float] = None
    t0: Optional[float] = None
    d: Optional[float] = None
    gamma: Optional[float] = None


@dataclass
class ExpParams:
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    d: Optional[float] = None
    f: Optional[float] = None


class DataLoadError(Exception):
    pass


def load_light_curve(path: Union[str, Path], mag_min: float = -1000, mag_max: float = 1000) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"Light curve file not found: {p}")
    try:
        data = np.loadtxt(p, comments="#", dtype=float)
    except Exception:
        data_rows = []
        with p.open("r") as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    data_rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if not data_rows:
            raise DataLoadError(f"No usable data in {p}")
        data = np.array(data_rows, dtype=float)

    if data.ndim == 1:
        if data.size < 2:
            raise DataLoadError(f"Expected at least 2 columns in {p}")
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise DataLoadError(f"Expected at least 2 columns in {p}")

    times = np.asarray(data[:, 0], dtype=float)
    mags = np.asarray(data[:, 1], dtype=float)
    mask = np.isfinite(times) & np.isfinite(mags) & (mags >= mag_min) & (mags <= mag_max)
    times = times[mask]
    mags = mags[mask]
    if times.size == 0:
        raise DataLoadError(f"No usable data in {p} after filtering to [{mag_min}, {mag_max}]")
    return times, mags


def load_intervals(path: Union[str, Path]) -> List[Interval]:
    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"Interval file not found: {p}")
    intervals: List[Interval] = []
    with p.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                continue
            if start < 0 or end < 0 or end <= start:
                continue
            kind = parts[2] if len(parts) > 2 and parts[2] in ("min", "max") else None
            intervals.append(Interval(start=start, end=end, kind=kind))
    if not intervals:
        raise DataLoadError(f"No usable intervals in {p}")
    return intervals


def _fit_for_order(x: np.ndarray, y: np.ndarray, order: int) -> Tuple[np.ndarray, float]:
    coeffs = np.polyfit(x, y, deg=order)
    residuals = y - np.polyval(coeffs, x)
    sse = float(np.sum(residuals**2))
    return coeffs, sse


def _pick_order(x: np.ndarray, y: np.ndarray, order_choice: Union[int, str], max_order: int) -> Tuple[np.ndarray, int, float]:
    if isinstance(order_choice, int):
        chosen_order = min(order_choice, max_order)
        coeffs, sse = _fit_for_order(x, y, chosen_order)
        return coeffs, chosen_order, sse
    candidates = [n for n in AUTO_ORDERS if n <= min(max_order, max(x.size - 3, 3))]  # 2+ residual degrees of freedom
    best, best_bic = None, float("inf")
    for n in candidates:
        try:
            coeffs, sse = _fit_for_order(x, y, n)
        except np.linalg.LinAlgError:
            continue
        # BIC: the least SSE always picks the highest order, which then wiggles around the extremum
        bic = x.size * np.log(max(sse, 1e-300) / x.size) + (n + 1) * np.log(x.size)
        if bic < best_bic:
            best, best_bic = (coeffs, n, sse), bic
    if best is None:
        raise ValueError("Could not fit polynomial for any order in auto mode.")
    return best


def exponential_model(x: np.ndarray, a: float, b: float, c: float, d: float, f: float) -> np.ndarray:
    return a * np.exp(-b * np.abs(x - c) ** d) + f


def brat_model(x, c0, c1, t0, d, gamma, c2=0.0):
    # c2: optional linear baseline (spots or a trend under the eclipse)
    arg = (x - t0) / d
    # Use cosh but clip large values to avoid overflow in exp
    ch = np.cosh(arg)
    # The formula is 1 - exp(1 - cosh(...)**gamma)
    # We clip to avoid numerical instability
    val = 1 - np.exp(1 - np.power(ch, gamma))
    return c0 + c1 * val + c2 * (x - t0)


def _fit_brat(
    x: np.ndarray, y: np.ndarray, params: BratParams | None = None, slope: bool = False, dip: bool = False
) -> Tuple[np.ndarray, float, np.ndarray]:
    """dip: a brightness minimum (magnitude peak) is wanted, so start there and keep c1 <= 0.
    Returns the parameters, the SSE and their covariance."""
    # Initial guesses from the data: the model equals c0 at t0 and c0 + c1 far from it, so start from the
    # point farthest from the median (the eclipse bottom, for magnitudes or flux alike) with the matching sign of c1
    base = float(np.median(y))
    k = min(5, y.size)
    ys = np.convolve(y, np.ones(k) / k, mode="valid")
    peak = int(np.argmax(ys) if dip else np.argmax(np.abs(ys - base)))
    c0_init = params.c0 if params and params.c0 is not None else float(ys[peak])
    c1_init = params.c1 if params and params.c1 is not None else base - float(ys[peak])
    t0_init = params.t0 if params and params.t0 is not None else float(x[peak + k // 2])
    d_init = params.d if params and params.d is not None else float((np.max(x) - np.min(x)) / 4.0)
    gamma_init = params.gamma if params and params.gamma is not None else 1.0

    p0 = [c0_init, c1_init, t0_init, d_init, gamma_init]
    # Bounds: c0, c1, t0, d > 0 (well, c1 should be positive for dimming), gamma > 0
    # Actually c0 can be anything. c1 should be positive for a dip if magnitude is used.
    # But let's allow flexibility.
    bounds = (
        (-np.inf, -np.inf, -np.inf, 1e-9, 1e-9),
        (np.inf, 0.0 if dip else np.inf, np.inf, np.inf, 10.0),  # Limit gamma to avoid extreme values
    )
    
    popt, pcov = curve_fit(brat_model, x, y, p0=p0, bounds=bounds)
    if slope:  # then free the linear baseline, starting from the flat one
        bounds = (bounds[0] + (-np.inf,), bounds[1] + (np.inf,))
        popt, pcov = curve_fit(brat_model, x, y, p0=[*popt, 0.0], bounds=bounds, maxfev=5000)
    y_fit = brat_model(x, *popt)
    sse = float(np.sum((y - y_fit) ** 2))
    return popt, sse, pcov


def _fit_exponential(
    x: np.ndarray, y: np.ndarray, params: ExpParams | None = None
) -> Tuple[np.ndarray, float, np.ndarray]:
    a_init = params.a if params and params.a is not None else float(np.max(y) - np.min(y))
    b_init = params.b if params and params.b is not None else 2.0
    c_init = params.c if params and params.c is not None else float(np.average(x))
    d_init = params.d if params and params.d is not None else 0.5
    f_init = params.f if params and params.f is not None else float(np.average(y))

    p0 = [a_init, b_init, c_init, d_init, f_init]
    bounds = (
        (1e-9, 1e-9, -np.inf, 1e-9, -np.inf),  # Slightly above zero for A, B, D
        (np.inf, np.inf, np.inf, np.inf, np.inf),
    )
    popt, pcov = curve_fit(exponential_model, x, y, p0=p0, bounds=bounds)
    y_fit = exponential_model(x, *popt)
    sse = float(np.sum((y - y_fit) ** 2))
    return popt, sse, pcov


def _sigma(pcov: np.ndarray, i: int) -> float:
    """1-sigma error of parameter i from curve_fit's covariance (scaled by the residuals)."""
    v = float(pcov[i, i]) if pcov is not None else float("nan")
    return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("inf")


def _poly_sigma_t0(x: np.ndarray, y: np.ndarray, coeffs: np.ndarray, t: float) -> float:
    """Error of the turning point t of the polynomial fit: p'(t) = 0, so dt/dc = -(dp'/dc) / p''(t),
    propagated through the covariance of the coefficients (scaled by the residuals)."""
    n = coeffs.size - 1
    try:
        _, cov = np.polyfit(x, y, n, cov=True)
    except (ValueError, np.linalg.LinAlgError):  # too few points for the residual variance
        return float("inf")
    curvature = np.polyval(np.polyder(coeffs, 2), t) if n >= 2 else 0.0
    if curvature == 0:
        return float("inf")
    grad = -np.array([(n - j) * t ** (n - j - 1) if j < n else 0.0 for j in range(n + 1)]) / curvature
    v = float(grad @ cov @ grad)
    return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("inf")


# Near-extremum functions of MAVKA (Andrych & Andronov 2019, arXiv:1812.06949)

def sym_model(x, t0, *c):
    """Symmetric polynomial: sum of c_k (x - t0)^(2k), highest power first."""
    return np.polyval(c, (x - t0) ** 2)


def wsl_model(x, a, b, t0, h):
    """Wall-supported line: flat a within |x - t0| < h (a total eclipse), walls b (|x - t0| - h)^1.5 outside."""
    return a + b * np.clip(np.abs(x - t0) - h, 0, None) ** 1.5


def apar_model(x, a, b, t0, dl, dr):
    """Asymptotic parabola (Marsakova & Andronov 1996): a + b (x - t0)^2 on [t0 - dl, t0 + dr], its tangent
    lines outside, so the two branches may differ (asymmetric maxima of pulsating stars)."""
    u = x - t0
    return a + b * np.where(u > dr, 2 * dr * u - dr**2, np.where(u < -dl, -2 * dl * u - dl**2, u**2))


MODELS = {"exp": exponential_model, "brat": brat_model, "sym": sym_model, "wsl": wsl_model, "apar": apar_model}


def evaluate(res: FitResult, x: np.ndarray) -> np.ndarray:
    """The fitted curve of a result at times x."""
    if res.method in MODELS:
        return MODELS[res.method](x, *res.coefficients)
    return np.polyval(res.coefficients, x - res.x_mean)


def _sym_sse(x, y, t, k):
    a = np.vander((x - t) ** 2, k + 1)
    c = np.linalg.lstsq(a, y, rcond=None)[0]
    r = y - a @ c
    return float(r @ r), c


def _fit_sym(x: np.ndarray, y: np.ndarray):
    """Symmetric polynomial of degree 2k (k = 1..4 by BIC). t0: least SSE, from a grid over the middle 70 %
    of the interval, refined; its error from the curvature of SSE(t0), the coefficients following t0."""
    lo, hi = float(x.min()), float(x.max())
    w = hi - lo
    grid = np.linspace(lo + 0.15 * w, hi - 0.15 * w, 41)
    step = grid[1] - grid[0]
    g = grid[int(np.argmin([_sym_sse(x, y, t, 2)[0] for t in grid]))]
    best = None
    for k in range(1, 5):
        if x.size < k + 4:  # 2+ residual degrees of freedom
            break
        t0 = float(minimize_scalar(lambda t: _sym_sse(x, y, t, k)[0], bounds=(g - 2 * step, g + 2 * step),
                                   method="bounded", options={"xatol": w * 1e-8}).x)
        sse, c = _sym_sse(x, y, t0, k)
        bic = x.size * np.log(max(sse, 1e-300) / x.size) + (k + 2) * np.log(x.size)
        if best is None or bic < best[0]:
            best = (bic, k, t0, sse, c)
    if best is None:
        raise ValueError("Interval too short to fit.")
    _, k, t0, sse, c = best
    h = w * 1e-3
    curvature = (_sym_sse(x, y, t0 + h, k)[0] - 2 * sse + _sym_sse(x, y, t0 - h, k)[0]) / h**2
    s2 = sse / (x.size - k - 2)
    sigma = float(np.sqrt(2 * s2 / curvature)) if curvature > 0 else float("inf")
    return t0, k, sse, c, sigma


def _core(x: np.ndarray, y: np.ndarray) -> slice:
    """The part of the interval beyond half the height of its extremum over the level at the interval ends:
    where the near-extremum functions hold (a Splitter window reaches the neighbouring extrema)."""
    q = np.polyfit(x - x.mean(), y, 2)
    n = max(y.size // 10, 1)
    edge = float(np.median(np.r_[y[:n], y[-n:]]))
    peak = float(np.percentile(y, 98 if q[0] < 0 else 2))  # magnitudes: a peak is a brightness minimum
    idx = np.flatnonzero((y - (edge + peak) / 2) * np.sign(peak - edge) > 0)
    return slice(idx[0], idx[-1] + 1) if idx.size >= 8 else slice(None)


def _fit_walls(model, x: np.ndarray, y: np.ndarray):
    """Wall-supported line or asymptotic parabola on the core of the interval, started from the vertex of a
    parabola through it."""
    core = _core(x, y)
    x, y = x[core], y[core]
    lo, hi = float(x.min()), float(x.max())
    w = hi - lo
    xm = float(np.mean(x))
    q = np.polyfit(x - xm, y, 2)
    t = float(np.clip(xm - q[1] / (2 * q[0]), lo + 0.15 * w, hi - 0.15 * w)) if q[0] else xm
    a = float(np.median(y[np.abs(x - t) < 0.1 * w])) if np.any(np.abs(x - t) < 0.1 * w) else float(np.polyval(q, t - xm))
    if model is wsl_model:
        b = np.sign(q[0] or 1.0) * np.ptp(y) / (0.5 * w) ** 1.5
        p0, bounds = [a, b, t, 0.1 * w], ([-np.inf, -np.inf, lo, 0.0], [np.inf, np.inf, hi, w])
    else:
        p0 = [a, q[0] or 1e-9, t, 0.25 * w, 0.25 * w]
        bounds = ([-np.inf, -np.inf, lo, 1e-3 * w, 1e-3 * w], [np.inf, np.inf, hi, 2 * w, 2 * w])
    popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=5000)
    return popt, float(np.sum((y - model(x, *popt)) ** 2)), pcov, (lo, hi)


def fit_extremum(
    times: np.ndarray, mags: np.ndarray, method_choice: Union[int, str, dict]
) -> FitResult:
    if len(times) < 3:
        raise ValueError("Interval too short to fit.")

    name = method_choice.get("method") if isinstance(method_choice, dict) else method_choice
    if name == "sym":
        t0, k, sse, c, sigma = _fit_sym(times, mags)
        return FitResult(index=-1, interval=Interval(start=0, end=0), order=2 * k, t0=t0,
                         kind="max" if c[-2] > 0 else "min",  # magnitudes: up from t0 is a brightness maximum
                         coefficients=np.r_[t0, c], x_mean=float(np.mean(times)), sse=sse, y_at_t0=float(c[-1]),
                         method="sym", sigma_t0=sigma)
    if name in ("wsl", "apar"):
        popt, sse, pcov, span = _fit_walls(MODELS[name], times, mags)
        return FitResult(index=-1, interval=Interval(start=0, end=0), order=-1, t0=float(popt[2]),
                         kind="max" if popt[1] > 0 else "min", coefficients=popt, x_mean=float(np.mean(times)),
                         sse=sse, y_at_t0=float(popt[0]), method=name, sigma_t0=_sigma(pcov, 2), x_range=span)

    x_mean = float(np.mean(times))
    centered_x = times - x_mean

    is_exp = False
    is_brat = False
    exp_params = None
    brat_params = None

    if isinstance(method_choice, str):
        if method_choice == "exponential":
            is_exp = True
        elif method_choice == "brat":
            is_brat = True
    elif isinstance(method_choice, dict):
        m = method_choice.get("method")
        if m == "exponential":
            is_exp = True
            exp_params = method_choice.get("params")
        elif m == "brat":
            is_brat = True
            brat_params = method_choice.get("params")

    if is_exp:
        popt, sse, pcov = _fit_exponential(times, mags, params=exp_params)
        t0 = float(popt[2])
        y_at_t0 = float(popt[0] + popt[4])  # A + F
        kind = "min"  # A > 0: a magnitude peak, i.e. a brightness minimum
        return FitResult(
            index=-1,
            interval=Interval(start=0, end=0),
            order=-1,
            t0=t0,
            kind=kind,
            coefficients=popt,
            x_mean=x_mean,
            sse=sse,
            y_at_t0=y_at_t0,
            method="exp",
            sigma_t0=_sigma(pcov, 2),
        )
    
    if is_brat:
        opts = method_choice if isinstance(method_choice, dict) else {}
        popt, sse, pcov = _fit_brat(
            times, mags, params=brat_params, slope=opts.get("slope", False), dip=opts.get("dip", False)
        )
        t0 = float(popt[2])
        y_at_t0 = float(popt[0])  # psi(t0) = 0, so the model at t0 is c0
        kind = "min" if popt[1] < 0 else "max"  # c1 < 0: a magnitude peak at t0, i.e. a brightness minimum
        return FitResult(
            index=-1,
            interval=Interval(start=0, end=0),
            order=-1,
            t0=t0,
            kind=kind,
            coefficients=popt,
            x_mean=x_mean,
            sse=sse,
            y_at_t0=y_at_t0,
            method="brat",
            sigma_t0=_sigma(pcov, 2),
        )

    # Polynomial path
    order_val = method_choice
    if isinstance(method_choice, dict):
        order_val = method_choice.get("order", "auto")

    max_order = max(1, min(len(times) - 1, max(AUTO_ORDERS)))
    # Ensure order_val is Union[int, str]
    if not isinstance(order_val, (int, str)):
        order_val = "auto"
    coeffs, chosen_order, sse = _pick_order(centered_x, mags, order_val, max_order)
    dense_x = np.linspace(centered_x.min(), centered_x.max(), 5000)
    dense_y = np.polyval(coeffs, dense_x)
    idx_min = int(np.argmin(dense_y))
    idx_max = int(np.argmax(dense_y))
    dist_min = abs(dense_x[idx_min])
    dist_max = abs(dense_x[idx_max])
    if dist_min <= dist_max:
        t0_centered = dense_x[idx_min]
        kind = "max"
        y_at_t0 = float(dense_y[idx_min])
    else:
        t0_centered = dense_x[idx_max]
        kind = "min"
        y_at_t0 = float(dense_y[idx_max])
    t0 = float(t0_centered + x_mean)
    turning = 0 < (idx_min if kind == "max" else idx_max) < dense_x.size - 1
    return FitResult(
        index=-1,
        interval=Interval(start=0, end=0),
        order=chosen_order,
        t0=t0,
        kind=kind,
        coefficients=coeffs,
        x_mean=x_mean,
        sse=sse,
        y_at_t0=y_at_t0,
        method="poly",
        sigma_t0=_poly_sigma_t0(centered_x, mags, coeffs, t0_centered) if turning else float("inf"),
    )


def segment(times: np.ndarray, interval: Interval, wings: float = 0.0) -> slice:
    """Points of the interval (end inclusive), widened by `wings` of its width on each side."""
    end = min(interval.end, times.size - 1)
    if interval.start > end:
        return slice(interval.start, interval.start)  # beyond the data: nothing to fit
    if wings <= 0:
        return slice(interval.start, end + 1)
    pad = wings * (times[end] - times[interval.start])
    return slice(
        int(np.searchsorted(times, times[interval.start] - pad)),
        int(np.searchsorted(times, times[end] + pad, side="right")),
    )


def _fit_interval(times: np.ndarray, mags: np.ndarray, idx: int, interval: Interval, order_choice, wings: float) -> FitResult:
    if isinstance(order_choice, dict) and order_choice.get("method") == "auto":
        return _fit_auto(times, mags, idx, interval, wings)
    seg = segment(times, interval, wings)
    fit = fit_extremum(times[seg], mags[seg], order_choice)
    if not times[seg][0] <= fit.t0 <= times[seg][-1]:
        raise ValueError(f"Fit diverged: extremum at {fit.t0:.5f} is outside the interval.")
    if interval.kind and fit.kind != interval.kind:
        raise ValueError(f"Found a brightness {fit.kind}, the interval is a {interval.kind}.")
    fit.index, fit.interval, fit.wings = idx, interval, wings
    return fit


def _fit_auto(times: np.ndarray, mags: np.ndarray, idx: int, interval: Interval, wings: float) -> FitResult:
    """Brightness minimum: the eclipse profile with a linear baseline on the interval with wings, kept if
    its t0 is inside the interval and its width is that of an eclipse (on the TESS sample: a smaller O-C
    scatter than a polynomial for 14 of 16 stars). Maximum, or no such fit: polynomial of BIC order on
    the interval itself."""
    kind = interval.kind
    if kind is None:  # magnitudes: a brightness minimum is higher in the middle than at the ends
        m = mags[segment(times, interval)]
        third = max(len(m) // 3, 1)
        kind = "min" if np.median(m[third:-third] if len(m) > 2 else m) > np.median(np.r_[m[:third], m[-third:]]) else "max"
    if kind == "min":
        try:
            fit = _fit_interval(times, mags, idx, interval, {"method": "brat", "params": None, "slope": True, "dip": True}, wings)
            a, b = times[interval.start], times[min(interval.end, times.size - 1)]
            d, gamma = fit.coefficients[3], fit.coefficients[4]
            fwhm = 2 * d * np.arccosh((1 + np.log(2)) ** (1 / gamma))  # psi = 1/2
            # eclipses: FWHM 0.3-0.6 of the interval; else a neighbour, a spike or no dip at all
            if a <= fit.t0 <= b and 0.15 * (b - a) < fwhm < 2 * (b - a):
                return fit
        except Exception:
            pass
    return _fit_interval(times, mags, idx, interval, "auto", 0.0)


def approximate_all(
    times: np.ndarray, mags: np.ndarray, intervals: Iterable[Interval], order_choice: Union[int, str, dict],
    wings: float = 0.0, errors: Optional[dict] = None,
) -> List[FitResult]:
    """Fit every interval; an interval whose fit fails is skipped (its index is missing from the results,
    the reason goes to `errors[index]` if a dict is given)."""
    results: List[FitResult] = []
    for idx, interval in enumerate(intervals):
        try:
            results.append(_fit_interval(times, mags, idx, interval, order_choice, wings))
        except Exception as exc:
            if errors is not None:
                errors[idx] = str(exc)
    return results


def recompute_result(
    times: np.ndarray, mags: np.ndarray, result: FitResult, order_choice: Union[int, str, dict], wings: float = 0.0
) -> FitResult:
    return _fit_interval(times, mags, result.index, result.interval, order_choice, wings)


def results_to_table_rows(results: List[FitResult]) -> List[List[str]]:
    rows: List[List[str]] = []
    for res in results:
        order_str = str(res.order) if res.method in ("poly", "sym") else "N/A"
        rows.append(
            [
                str(res.index + 1),
                f"{res.interval.start}",
                f"{res.interval.end}",
                order_str,
                f"{res.t0:.6f}",
                res.kind,
                f"{res.sse:.4f}",
                res.method,
            ]
        )
    return rows


def save_results_and_figures(
    results: List[FitResult],
    times: np.ndarray,
    mags: np.ndarray,
    save_dir: Union[str, Path],
) -> Path:
    target = Path(save_dir)
    target.mkdir(parents=True, exist_ok=True)
    results_path = target / "RESULTS.txt"
    with results_path.open("w") as f:
        # Header makes downstream parsing easier and keeps values in fixed columns
        f.write("t0 kind order start end sse\n")
        for res in results:
            f.write(
                f"{res.t0:.10f} {res.kind} {res.order} {res.interval.start} {res.interval.end} {res.sse:.6f}\n"
            )
    for res in results:
        seg = segment(times, res.interval, res.wings)
        seg_x, seg_y = times[seg], mags[seg]
        dense_x = np.linspace(*(res.x_range or (seg_x.min(), seg_x.max())), 1000)
        dense_y = evaluate(res, dense_x)
        label = f"{res.method} n={res.order}" if res.method in ("poly", "sym") else f"{res.method} fit"
        fig, ax = plt.subplots(figsize=(8, 5))
        disp_seg_y = -seg_y
        disp_dense_y = -dense_y
        ax.scatter(seg_x, disp_seg_y, s=10, color="black", label="data")
        ax.plot(dense_x, disp_dense_y, color="tab:blue", label=label)
        ax.axvline(res.t0, color="red", linestyle="--", label=f"{res.kind} at {res.t0:.6f}")
        ax.set_xlabel("time")
        ax.set_ylabel("mmag")
        ax.legend()
        fig.tight_layout()
        fig.savefig(target / f"interval_{res.index + 1}.png", dpi=150)
        plt.close(fig)
    return results_path
