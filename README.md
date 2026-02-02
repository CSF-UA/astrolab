# Astrolab

Small, self-contained Python apps for the Astrolab project. Each app lives under `apps/` and can be run directly with `uv`.

## Quick start

1) Install uv (if not already): see https://docs.astral.sh/uv/
2) From the repo root: `uv sync`
3) Run the approximation app:
   - `uv run approximation`


## Apps

- `approximation` — GUI tool to load a TESS light curve (`.tess`) and an interval file (`.txt`), pick a polynomial order (auto 3–10 or fixed 3–10), fit each interval, review results, re-fit or delete specific extrema, then export `RESULTS.txt` plus per-interval figures.

### Approximation workflow

- **Load files:** left panel buttons for light curve (`.tess`, two columns: time, magnitude; headers/# comments ignored) and intervals (`.txt`, two integers per line: start end indices).
- **Choose order:** auto tries orders 3–10 and picks the lowest SSE; fixed uses the chosen order, clamped to the available points in each interval.
- **Approximate all:** processes every interval and plots extrema on the VisPy canvas (y uses `-magnitude` so brighter goes up).
- **Review/edit:** select a row to highlight its interval; use “Re-approx selected” (uses current order choice) or “Delete selected”. “Clear results” removes all fits.
- **Save:** choose a folder to write `RESULTS.txt` and `interval_<n>.png` figures for each interval.

## Notes

- The legacy script `Extrema_timings_v2.1.py` remains unchanged for reference.
- Dependencies are declared in `pyproject.toml`; no global installs required.
