# Astrolab

Repository for apps/software/codes that are used in CSF AstroLab projects.
Each app lives under `apps/` and can be run directly with `uv`.

## Quick start

1) Install uv (if not already): see https://docs.astral.sh/uv/
2) From the repo root: `uv sync`
3) Run the approximation app:
   - `uv run approximation`
4) Run the period app:
   - `uv run period`

## Apps

- `approximation` — software for approximating light curve extremas, load a TESS light curve (`.tess`) and an interval file (`.txt`), pick a polynomial order (auto 3–10 or fixed 3–10). After that, user can approximate each interval, review results, redo fit or delete specific extrema, then export `RESULTS.txt` as well as figures for eac interval.
- `period` — GUI app for period calculation and phase curves based on the previous terminal workflow. Search/download TESS SPOC sectors via `lightkurve` or load a local light-curve file, click sector rows to view light curves, compute periodograms (auto period), enable manual peak-pick on the periodogram, and plot phase curves for checked sectors. PNGs for displayed light curves/periodograms/phase curves are saved to the selected output folder.
