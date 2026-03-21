"""Core logic for computing re-radiation and O'Connell effect parameters.

Re-radiation parameter:
    Δm_refl = m_start − m_end
    where m_start / m_end are smoothed magnitudes at the boundaries of a maximum.

O'Connell effect:
    Δm = m_max2 − m_max1
    Difference between fitted magnitudes at two consecutive maxima.

Smoothing and splitting methods are ported from the Splitter project (core.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Data loading (shared with approximation)
# ---------------------------------------------------------------------------

class DataLoadError(Exception):
    pass


def load_light_curve(
    path: Union[str, Path], mag_min: float = -1000, mag_max: float = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a two-column (JD  mag) light-curve file (.tess or .txt)."""
    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"Light curve file not found: {p}")
    try:
        data = np.loadtxt(p, comments="#", dtype=float)
    except Exception:
        data_rows: list[tuple[float, float]] = []
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
    mask = (
        np.isfinite(times)
        & np.isfinite(mags)
        & (mags >= mag_min)
        & (mags <= mag_max)
    )
    times, mags = times[mask], mags[mask]
    if times.size == 0:
        raise DataLoadError(f"No usable data in {p} after filtering")
    return times, mags


# ---------------------------------------------------------------------------
# Smoothing (ported from Splitter core.py)
# ---------------------------------------------------------------------------


def _vectorized_sm(y: np.ndarray, N: int) -> np.ndarray:
    """Moving average.  First/last N samples are left unchanged."""
    if N <= 0:
        return y.copy()
    window_size = 2 * N + 1
    if len(y) < window_size:
        return y.copy()
    kernel = np.ones(window_size) / window_size
    convolved = np.convolve(y, kernel, mode="valid")
    Y = y.copy()
    Y[N : len(y) - N] = convolved
    return Y


def _NN(T0: float, alpha: float) -> list[int]:
    """Window sizes for the smoothing cascade."""
    N: list[int] = []
    Nmax = alpha * T0 * 584
    n = 2
    nn = 0
    while nn < Nmax:
        nn = int(0.7734 * np.exp(0.4484 * n))
        N.append(nn)
        n += 1
    return N


def smooth(T0: float, alpha: float, y: np.ndarray) -> Tuple[np.ndarray, int]:
    """Cascading forward-backward smooth."""
    N_vals = _NN(T0, alpha)
    Y = y.copy()
    for n_val in N_vals:
        Y = _vectorized_sm(Y, n_val)
    for n_val in reversed(N_vals):
        Y = _vectorized_sm(Y, n_val)
    return Y, (max(N_vals) if N_vals else 0)


# ---------------------------------------------------------------------------
# Splitting by second-derivative zero-crossings (ported from Splitter core.py)
# ---------------------------------------------------------------------------


def splitting_normal(
    x: np.ndarray,
    y: np.ndarray,
    T0: float,
    alpha: float = 0.12,
) -> Tuple[list[int], list[int]]:
    """Segment a light curve into intervals at inflection points and gaps."""
    yy, Nmax = smooth(T0, alpha, y)

    if len(x) < 3:
        return [], []

    dx = x[1:] - x[:-1]
    dy = yy[1:] - yy[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = dy / dx
        d[~np.isfinite(d)] = 0.0
    for N in [3, 5, 9, 13, 9, 5, 3]:
        d = _vectorized_sm(d, N)

    d_diff = d[1:] - d[:-1]
    dx_shifted = dx[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = d_diff / dx_shifted
        dd[~np.isfinite(dd)] = 0.0
    for N in [3, 5, 9, 13, 9, 5, 3]:
        dd = _vectorized_sm(dd, N)

    start_idx = Nmax + 1
    end_idx = len(x) - Nmax - 1
    if start_idx >= end_idx:
        return [], []

    scan_indices = np.arange(start_idx, end_idx - 1)
    valid_mask = (
        (scan_indices < len(dd))
        & ((scan_indices - 1) < len(dd))
        & (scan_indices < len(dx))
    )
    scan_indices = scan_indices[valid_mask]
    if len(scan_indices) == 0:
        return [], []

    gaps = dx[scan_indices] > (T0 * 0.5)
    crossings = (dd[scan_indices] * dd[scan_indices - 1]) < 0
    breakpoints = gaps | crossings
    split_indices = scan_indices[breakpoints]

    intervals_start: list[int] = [int(start_idx)]
    intervals_finish: list[int] = []
    if len(split_indices) > 0:
        intervals_finish.extend(int(s) for s in split_indices)
        intervals_start.extend(int(s) + 1 for s in split_indices)
    intervals_finish.append(int(end_idx))

    return intervals_start, intervals_finish


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MaximumInfo:
    """Information about a single maximum."""
    index: int
    start_idx: int
    end_idx: int
    t0: float               # time of extremum (from polynomial fit)
    y_at_t0: float           # magnitude at extremum
    m_start: float           # smoothed magnitude at start of maximum
    m_end: float             # smoothed magnitude at end of maximum
    delta_m_refl: float      # re-radiation parameter = m_start - m_end


@dataclass
class OConnellResult:
    """O'Connell effect between a pair of consecutive maxima."""
    max1_index: int
    max2_index: int
    t0_max1: float
    t0_max2: float
    y_max1: float
    y_max2: float
    delta_m: float           # O'Connell = m_max2 - m_max1


@dataclass
class AnalysisResult:
    """Full analysis output."""
    maxima: List[MaximumInfo] = field(default_factory=list)
    oconnell: List[OConnellResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Polynomial extremum fitting (simplified from approximation/logic.py)
# ---------------------------------------------------------------------------


def _fit_poly_extremum(
    times: np.ndarray, mags: np.ndarray
) -> Tuple[float, float, str]:
    """Fit a polynomial and find the extremum closest to center.

    Returns (t0, y_at_t0, kind).
    """
    if len(times) < 3:
        raise ValueError("Interval too short to fit")

    x_mean = float(np.mean(times))
    cx = times - x_mean

    max_order = min(len(times) - 1, 10)
    best_coeffs, best_sse = None, float("inf")
    for order in range(3, max_order + 1):
        try:
            coeffs = np.polyfit(cx, mags, deg=order)
            residuals = mags - np.polyval(coeffs, cx)
            sse = float(np.sum(residuals**2))
        except np.linalg.LinAlgError:
            continue
        if sse < best_sse:
            best_coeffs, best_sse = coeffs, sse
    if best_coeffs is None:
        raise ValueError("Could not fit polynomial")

    dense_x = np.linspace(cx.min(), cx.max(), 5000)
    dense_y = np.polyval(best_coeffs, dense_x)
    idx_min = int(np.argmin(dense_y))
    idx_max = int(np.argmax(dense_y))
    dist_min = abs(dense_x[idx_min])
    dist_max = abs(dense_x[idx_max])

    if dist_min <= dist_max:
        t0 = float(dense_x[idx_min]) + x_mean
        y_at_t0 = float(dense_y[idx_min])
        kind = "max"     # minimum in magnitude = brightness maximum
    else:
        t0 = float(dense_x[idx_max]) + x_mean
        y_at_t0 = float(dense_y[idx_max])
        kind = "min"
    return t0, y_at_t0, kind


# ---------------------------------------------------------------------------
# Main analysis functions
# ---------------------------------------------------------------------------


def analyze_maxima(
    times: np.ndarray,
    mags: np.ndarray,
    T0: float,
    alpha: float = 0.12,
    interval_starts: list[int] | None = None,
    interval_ends: list[int] | None = None,
) -> AnalysisResult:
    """Run the full analysis pipeline.

    Parameters
    ----------
    times, mags : light curve arrays
    T0 : orbital period (days) — used by the smoother/splitter
    alpha : smoothing strength (default 0.12)
    interval_starts, interval_ends :
        Pre-computed interval boundaries. If *None* the splitter runs
        automatically on the light curve.

    Returns
    -------
    AnalysisResult with per-maximum info and O'Connell pairs.
    """
    # 1. Smooth the light curve (needed for boundary magnitudes)
    y_smooth, _ = smooth(T0, alpha, mags)

    # 2. Obtain intervals (split or use provided)
    if interval_starts is not None and interval_ends is not None:
        starts, ends = list(interval_starts), list(interval_ends)
    else:
        starts, ends = splitting_normal(times, mags, T0, alpha)

    if not starts:
        return AnalysisResult()

    # 3. For each interval, fit the extremum and compute re-radiation
    maxima: List[MaximumInfo] = []
    max_idx = 0
    for s, e in zip(starts, ends):
        seg_t = times[s : e + 1]
        seg_m = mags[s : e + 1]
        if len(seg_t) < 5:
            continue

        try:
            t0, y_at_t0, kind = _fit_poly_extremum(seg_t, seg_m)
        except ValueError:
            continue

        if kind != "max":
            # We only need maxima for these calculations
            continue

        m_start = float(y_smooth[s])
        m_end = float(y_smooth[e])
        delta_m_refl = m_start - m_end

        maxima.append(
            MaximumInfo(
                index=max_idx,
                start_idx=s,
                end_idx=e,
                t0=t0,
                y_at_t0=y_at_t0,
                m_start=m_start,
                m_end=m_end,
                delta_m_refl=delta_m_refl,
            )
        )
        max_idx += 1

    # 4. Compute O'Connell effect for consecutive maxima pairs
    oconnell: List[OConnellResult] = []
    sorted_maxima = sorted(maxima, key=lambda m: m.t0)
    for i in range(len(sorted_maxima) - 1):
        m1 = sorted_maxima[i]
        m2 = sorted_maxima[i + 1]
        delta_m = m2.y_at_t0 - m1.y_at_t0
        oconnell.append(
            OConnellResult(
                max1_index=m1.index,
                max2_index=m2.index,
                t0_max1=m1.t0,
                t0_max2=m2.t0,
                y_max1=m1.y_at_t0,
                y_max2=m2.y_at_t0,
                delta_m=delta_m,
            )
        )

    return AnalysisResult(maxima=sorted_maxima, oconnell=oconnell)
