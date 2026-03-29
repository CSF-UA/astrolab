"""O-C curve approximation logic.

Fits O-C data to a combined polynomial trend + sinusoidal model:
    O-C = (Σ Cj·(E - E0)^j) + R·cos(2π(E - E_max) / P2)

The implementation follows the robust 4-stage architecture described in
the methodology report (o-c.tex):

    1. Linear polynomial detrending via numpy.polyfit.
    2. Generalized Lomb-Scargle periodogram (scipy, floating_mean=True)
       on the detrended residuals for initial period estimation.
    3. Constrained non-linear optimisation via Trust Region Reflective
       (trf) with physically-motivated parameter bounds and x_scale.
    4. BIC model comparison (linear-only vs linear+sinusoid) to guard
       against overfitting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lombscargle


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OCDataError(Exception):
    """Raised when O-C data cannot be loaded or parsed."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OCFitResult:
    """Stores the results of the O-C curve fit."""

    # --- fitted parameters ---
    c0: float               # constant offset
    c1: float               # linear trend coefficient
    amplitude: float        # semi-amplitude R of sinusoidal component (days)
    period: float           # long (secondary) period P2 (epochs)
    phase: float            # phase φ (radians)

    # --- data arrays ---
    time: np.ndarray        # time/epoch array used
    oc: np.ndarray          # observed O-C values
    oc_fitted: np.ndarray   # fitted O-C values at the observed epochs

    # --- diagnostics ---
    rms: float              # RMS of final residuals
    bic_linear: float       # BIC of linear-only model (k=2)
    bic_sinusoidal: float   # BIC of linear+sinusoid model (k=5)
    sinusoid_preferred: bool  # True when BIC favours the sinusoidal model
    param_errors: np.ndarray | None  # σ for each parameter (from pcov)
    cond_number: float      # condition number of the covariance matrix


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_oc_data(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load O-C data from a tab/space-separated text file.

    Expects two columns: Time/Epoch and O-C value.
    No header row. Values may use scientific notation (e.g. 3.84E-05).

    Returns sorted (time, oc) arrays with duplicates removed.
    """
    p = Path(path)
    if not p.exists():
        raise OCDataError(f"O-C data file not found: {p}")

    data_rows: list[Tuple[float, float]] = []
    with p.open("r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                oc = float(parts[1])
                data_rows.append((t, oc))
            except ValueError:
                continue

    if not data_rows:
        raise OCDataError(f"No usable data in {p}")

    data = np.array(data_rows, dtype=float)
    time = data[:, 0]
    oc = data[:, 1]

    # Sort chronologically
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    oc = oc[sort_idx]

    # Remove exact duplicates (same epoch)
    _, unique_idx = np.unique(time, return_index=True)
    time = time[unique_idx]
    oc = oc[unique_idx]

    if time.size < 4:
        raise OCDataError("Need at least 4 data points for the fit.")

    return time, oc


# ---------------------------------------------------------------------------
# Model function
# ---------------------------------------------------------------------------

def oc_model(t: np.ndarray, c0: float, c1: float, amplitude: float,
             period: float, phase: float) -> np.ndarray:
    """Combined linear trend + sinusoidal model for O-C data."""
    return c0 + c1 * t + amplitude * np.sin(2 * np.pi * t / period + phase)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_bic(n: int, rss: float, k: int) -> float:
    """Bayesian Information Criterion.

    BIC = n·ln(RSS/n) + k·ln(n)

    Parameters
    ----------
    n : number of data points
    rss : residual sum of squares
    k : number of free parameters
    """
    return n * np.log(rss / n) + k * np.log(n)


def _extract_covariance_diagnostics(
    pcov: np.ndarray,
) -> Tuple[np.ndarray | None, float]:
    """Safely extract parameter errors and condition number from pcov.

    Returns (param_errors, cond_number). Handles inf/nan gracefully.
    """
    try:
        if np.isinf(pcov).any():
            return np.inf * np.ones(pcov.shape[0]), np.inf
        cond = float(np.linalg.cond(pcov))
        errors = np.sqrt(np.diag(pcov))
        return errors, cond
    except np.linalg.LinAlgError:
        return None, np.inf


# ---------------------------------------------------------------------------
# Main fitting pipeline
# ---------------------------------------------------------------------------

def fit_oc_curve(
    time: np.ndarray,
    oc: np.ndarray,
    period_min: float | None = None,
    period_max: float | None = None,
) -> OCFitResult:
    """Fit the O-C data to a linear trend + sinusoid.

    Implements the robust 4-stage architecture:

        Stage 1 — Linear polynomial detrending.
        Stage 2 — Generalized Lomb-Scargle (floating_mean=True) for
                  initial period estimation.
        Stage 3 — Trust Region Reflective (trf) constrained optimisation
                  with dynamic bounds and x_scale.
        Stage 4 — BIC model comparison (linear vs sinusoidal).

    Args:
        period_min: Lower bound for fitting period P₂ (default: 100).
        period_max: Upper bound for fitting period P₂ (default: 3×ΔE).
    """
    n_pts = len(time)
    delta_e = time[-1] - time[0]

    # ── Stage 1: Linear detrending ─────────────────────────────────────
    trend_coeffs = np.polyfit(time, oc, 1)  # [slope, intercept]
    oc_detrended = oc - np.polyval(trend_coeffs, time)

    # ── Stage 2: Generalized Lomb-Scargle periodogram ──────────────────
    #   Frequency grid capped at 2×ΔE (report §5.2).
    #   Min period = 100 epochs (physical lower bound).
    freq_min = 1.0 / (2.0 * delta_e)   # period = 2×ΔE
    freq_max = 1.0 / 100.0             # period = 100 epochs
    n_freqs = max(1000, 5 * n_pts)
    freqs = np.linspace(freq_min, freq_max, n_freqs)
    angular_freqs = 2.0 * np.pi * freqs

    power = lombscargle(
        time, oc_detrended, angular_freqs,
        floating_mean=True,
    )

    # Collect candidate periods: top 3 GLS peaks
    top_indices = np.argsort(power)[-3:][::-1]
    candidate_periods = [1.0 / freqs[i] for i in top_indices]

    # Guess amplitude from the detrended data
    residual_range = np.ptp(oc_detrended)
    guess_amplitude = residual_range / 2.0

    # ── Stage 3: Constrained trf optimisation ──────────────────────────
    #   Dynamic bounds derived from data (report §5.3).
    p_min = period_min if period_min is not None else 100.0
    p_max = period_max if period_max is not None else 3.0 * delta_e
    lower_bounds = [-np.inf, -np.inf, 0.0,   p_min,      -np.pi]
    upper_bounds = [np.inf,  np.inf,  1.5 * residual_range,
                    p_max, 3 * np.pi]

    # Parameter scaling (report §5.4): harmonise different magnitudes.
    x_scale = [1, 1e-4, 1e-3, 1e3, 1]

    best_rms = np.inf
    best_popt: np.ndarray | None = None
    best_pcov: np.ndarray | None = None

    for p0 in candidate_periods:
        # Clamp p0 inside bounds to avoid ValueError from trf
        p0_clamped = np.clip(p0, lower_bounds[3], upper_bounds[3])
        initial = [trend_coeffs[1], trend_coeffs[0],
                   guess_amplitude, p0_clamped, 0.0]
        try:
            popt, pcov = curve_fit(
                oc_model, time, oc,
                p0=initial,
                method='trf',
                bounds=(lower_bounds, upper_bounds),
                x_scale=x_scale,
                maxfev=50000,
            )
            oc_fitted = oc_model(time, *popt)
            rms = float(np.sqrt(np.mean((oc - oc_fitted) ** 2)))
            if rms < best_rms:
                best_rms = rms
                best_popt = popt
                best_pcov = pcov
        except RuntimeError:
            continue

    if best_popt is None:
        raise OCDataError("Curve fitting failed to converge for all "
                          "candidate initial periods.")

    c0_opt, c1_opt, amplitude_opt, period_opt, phase_opt = best_popt

    # Normalise: ensure amplitude is positive
    if amplitude_opt < 0:
        amplitude_opt = -amplitude_opt
        phase_opt += np.pi

    # Normalise phase to [0, 2π)
    phase_opt = phase_opt % (2 * np.pi)

    # Compute fitted curve at observed epochs
    oc_fitted = oc_model(time, c0_opt, c1_opt, amplitude_opt,
                         period_opt, phase_opt)
    rms = float(np.sqrt(np.mean((oc - oc_fitted) ** 2)))

    # ── Stage 4: BIC model comparison ──────────────────────────────────
    #   Linear model (k=2)
    oc_linear = np.polyval(trend_coeffs, time)
    rss_linear = float(np.sum((oc - oc_linear) ** 2))
    bic_linear = _compute_bic(n_pts, rss_linear, k=2)

    #   Sinusoidal model (k=5)
    rss_sinusoidal = float(np.sum((oc - oc_fitted) ** 2))
    bic_sinusoidal = _compute_bic(n_pts, rss_sinusoidal, k=5)

    sinusoid_preferred = bic_sinusoidal < bic_linear

    # ── Covariance diagnostics ─────────────────────────────────────────
    param_errors, cond_number = _extract_covariance_diagnostics(best_pcov)

    return OCFitResult(
        c0=c0_opt,
        c1=c1_opt,
        amplitude=amplitude_opt,
        period=period_opt,
        phase=phase_opt,
        time=time,
        oc=oc,
        oc_fitted=oc_fitted,
        rms=rms,
        bic_linear=bic_linear,
        bic_sinusoidal=bic_sinusoidal,
        sinusoid_preferred=sinusoid_preferred,
        param_errors=param_errors,
        cond_number=cond_number,
    )
