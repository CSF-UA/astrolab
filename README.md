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
5) Other apps: `uv run oc_curve`, `uv run maxima_params`

## Apps

- `approximation` — times the extrema of a light curve: load a TESS light curve (`.tess`) and an interval file (`.txt`,
  e.g. from Splitter: `start end` and optionally `min`/`max`), pick a method, approximate every interval, review the
  results, refit or delete single extrema, then export `RESULTS.txt` and a figure of every interval. Methods:
  - **Auto** — brightness minima by Brat+ with a linear baseline on the interval with wings (kept if it looks like an
    eclipse), maxima and failed minima by a polynomial of BIC order;
  - **Polynomial** — order auto 3–10 (BIC) or fixed; **Exponential**; **Brat+** — the eclipse profile of Mikulášek (2015);
  - **Symmetric polynomial**, **Wall-supported line**, **Asymptotic parabola** — the near-extremum functions of MAVKA
    (Andrych & Andronov 2019) for symmetric or flat extrema, total eclipses and asymmetric maxima.
  **Wings** widen each interval on both sides (50 % for Auto and Brat+, which need the baseline around an eclipse;
  0 for the rest). Every fit also gives the 1-sigma error of its moment. Checks: `uv run python tests/test_approximation.py`,
  `xvfb-run -a uv run python tests/test_approximation_plot.py`.
- `period` — GUI app for period calculation and phase curves based on the previous terminal workflow. Search/download TESS SPOC sectors via `lightkurve` or load a local light-curve file, click sector rows to view light curves, compute periodograms (auto period), enable manual peak-pick on the periodogram, and plot phase curves for checked sectors. PNGs for displayed light curves/periodograms/phase curves are saved to the selected output folder.
