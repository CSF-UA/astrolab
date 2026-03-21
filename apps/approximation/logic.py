from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import numpy as np


AUTO_ORDERS = list(range(3, 11))


@dataclass
class Interval:
    start: int
    end: int


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
            intervals.append(Interval(start=start, end=end))
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
    candidates = [n for n in AUTO_ORDERS if n <= max_order]
    best_coeffs, best_order, best_sse = None, None, float("inf")
    for n in candidates:
        try:
            coeffs, sse = _fit_for_order(x, y, n)
        except np.linalg.LinAlgError:
            continue
        if sse < best_sse:
            best_coeffs, best_order, best_sse = coeffs, n, sse
    if best_coeffs is None:
        raise ValueError("Could not fit polynomial for any order in auto mode.")
    return best_coeffs, best_order, best_sse


def exponential_model(x: np.ndarray, a: float, b: float, c: float, d: float, f: float) -> np.ndarray:
    return a * np.exp(-b * np.abs(x - c) ** d) + f


def brat_model(x, c0, c1, t0, d, gamma):
    # Ensure float arguments to avoid issues with numpy
    arg = (x - t0) / d
    # Use cosh but clip large values to avoid overflow in exp
    ch = np.cosh(arg)
    # The formula is 1 - exp(1 - cosh(...)**gamma)
    # We clip to avoid numerical instability
    val = 1 - np.exp(1 - np.power(ch, gamma))
    return c0 + c1 * val


def _fit_brat(x: np.ndarray, y: np.ndarray, params: BratParams | None = None) -> Tuple[np.ndarray, float]:
    # Initial guesses
    c0_init = params.c0 if params and params.c0 is not None else float(np.average(y))
    c1_init = params.c1 if params and params.c1 is not None else float(np.max(y) - np.min(y))
    t0_init = params.t0 if params and params.t0 is not None else float(np.average(x))
    d_init = params.d if params and params.d is not None else float((np.max(x) - np.min(x)) / 4.0)
    gamma_init = params.gamma if params and params.gamma is not None else 1.0

    p0 = [c0_init, c1_init, t0_init, d_init, gamma_init]
    # Bounds: c0, c1, t0, d > 0 (well, c1 should be positive for dimming), gamma > 0
    # Actually c0 can be anything. c1 should be positive for a dip if magnitude is used.
    # But let's allow flexibility.
    bounds = (
        (-np.inf, -np.inf, -np.inf, 1e-9, 1e-9),
        (np.inf, np.inf, np.inf, np.inf, 10.0),  # Limit gamma to avoid extreme values
    )
    
    popt, _ = curve_fit(brat_model, x, y, p0=p0, bounds=bounds)
    y_fit = brat_model(x, *popt)
    sse = float(np.sum((y - y_fit) ** 2))
    return popt, sse


def _fit_exponential(
    x: np.ndarray, y: np.ndarray, params: ExpParams | None = None
) -> Tuple[np.ndarray, float]:
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
    popt, _ = curve_fit(exponential_model, x, y, p0=p0, bounds=bounds)
    y_fit = exponential_model(x, *popt)
    sse = float(np.sum((y - y_fit) ** 2))
    return popt, sse


def fit_extremum(
    times: np.ndarray, mags: np.ndarray, method_choice: Union[int, str, dict]
) -> FitResult:
    if len(times) < 3:
        raise ValueError("Interval too short to fit.")

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
        popt, sse = _fit_exponential(times, mags, params=exp_params)
        t0 = float(popt[2])
        y_at_t0 = float(popt[0] + popt[4])  # A + F
        kind = "max"
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
        )
    
    if is_brat:
        popt, sse = _fit_brat(times, mags, params=brat_params)
        t0 = float(popt[2])
        # For Brat+, y_at_t0 is c0 + c1 * (1 - exp(1 - cosh(0)**gamma)) = c0 + c1 * (1 - exp(1-1)) = c0
        # Wait, if c1 is the depth, then at t0, cosh(0)=1, val = 1 - exp(0) = 0. So y = c0.
        # But looking at the formula: psi = 1 - exp(1 - cosh(...)^gamma). 
        # If arg=0, cosh(0)=1, psi = 1 - exp(1 - 1) = 0. So f = c0.
        # So c0 is the baseline, and c1 is the depth?
        # The user says: "c1 strictly defines the maximum depth".
        # If psi=0 at t0, then c1 doesn't seem to be the depth unless psi ranges from 0 to 1.
        # Let's check: cosh(x) >= 1. cosh(x)^gamma >= 1 for positive gamma.
        # exp(1 - cosh^gamma) <= exp(0) = 1.
        # So psi = 1 - exp(...) ranges from 0 to 1.
        # So f ranges from c0 + c1*0 = c0 to c0 + c1*1.
        # So c0 is out-of-eclipse, and c1 is the depth.
        y_at_t0 = float(popt[0] + popt[1])
        kind = "max"
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
    )


def approximate_all(times: np.ndarray, mags: np.ndarray, intervals: Iterable[Interval], order_choice: Union[int, str]) -> List[FitResult]:
    results: List[FitResult] = []
    for idx, interval in enumerate(intervals):
        seg_x = times[interval.start : interval.end]
        seg_y = mags[interval.start : interval.end]
        fit = fit_extremum(seg_x, seg_y, order_choice)
        fit.index = idx
        fit.interval = interval
        results.append(fit)
    return results


def recompute_result(times: np.ndarray, mags: np.ndarray, result: FitResult, order_choice: Union[int, str]) -> FitResult:
    seg_x = times[result.interval.start : result.interval.end]
    seg_y = mags[result.interval.start : result.interval.end]
    updated = fit_extremum(seg_x, seg_y, order_choice)
    updated.index = result.index
    updated.interval = result.interval
    return updated


def results_to_table_rows(results: List[FitResult]) -> List[List[str]]:
    rows: List[List[str]] = []
    for i, res in enumerate(results, start=1):
        order_str = str(res.order) if res.method == "poly" else "N/A"
        rows.append(
            [
                str(i),
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
        seg_x = times[res.interval.start : res.interval.end]
        seg_y = mags[res.interval.start : res.interval.end]
        dense_x = np.linspace(seg_x.min(), seg_x.max(), 1000)
        if res.method == "exp":
            dense_y = exponential_model(dense_x, *res.coefficients)
            label = "exp fit"
        elif res.method == "brat":
            dense_y = brat_model(dense_x, *res.coefficients)
            label = "brat+ fit"
        else:
            dense_x_centered = dense_x - res.x_mean
            dense_y = np.polyval(res.coefficients, dense_x_centered)
            label = f"poly n={res.order}"
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
