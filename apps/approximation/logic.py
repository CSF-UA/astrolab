from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Union

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


def fit_extremum(times: np.ndarray, mags: np.ndarray, order_choice: Union[int, str]) -> FitResult:
    if len(times) < 3:
        raise ValueError("Interval too short to fit.")
    x_mean = float(np.mean(times))
    centered_x = times - x_mean
    max_order = max(1, min(len(times) - 1, max(AUTO_ORDERS)))
    coeffs, chosen_order, sse = _pick_order(centered_x, mags, order_choice, max_order)
    dense_x = np.linspace(centered_x.min(), centered_x.max(), 5000)
    dense_y = np.polyval(coeffs, dense_x)
    idx_min = int(np.argmin(dense_y))
    idx_max = int(np.argmax(dense_y))
    dist_min = abs(dense_x[idx_min])
    dist_max = abs(dense_x[idx_max])
    if dist_min <= dist_max:
        t0_centered = dense_x[idx_min]
        kind = "min"
        y_at_t0 = float(dense_y[idx_min])
    else:
        t0_centered = dense_x[idx_max]
        kind = "max"
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
        rows.append(
            [
                str(i),
                f"{res.interval.start}",
                f"{res.interval.end}",
                f"{res.order}",
                f"{res.t0:.6f}",
                res.kind,
                f"{res.sse:.4f}",
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
        dense_x_centered = dense_x - res.x_mean
        dense_y = np.polyval(res.coefficients, dense_x_centered)
        fig, ax = plt.subplots(figsize=(8, 5))
        disp_seg_y = -seg_y
        disp_dense_y = -dense_y
        ax.scatter(seg_x, disp_seg_y, s=10, color="black", label="data")
        ax.plot(dense_x, disp_dense_y, color="tab:blue", label=f"poly n={res.order}")
        ax.axvline(res.t0, color="red", linestyle="--", label=f"{res.kind} at {res.t0:.6f}")
        ax.set_xlabel("time")
        ax.set_ylabel("mmag")
        ax.legend()
        fig.tight_layout()
        fig.savefig(target / f"interval_{res.index + 1}.png", dpi=150)
        plt.close(fig)
    return results_path
