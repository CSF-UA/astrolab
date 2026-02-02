# Astrolab

Small, self-contained Python apps for the Astrolab project. Each app lives under `apps/` and can be run directly with `uv`.

## Quick start

1) Install uv (if not already): see https://docs.astral.sh/uv/
2) From the repo root: `uv sync`
3) Run the approximation app:
   - `uv run approximation`


## Apps

- `approximation` — software for approximating light curve extremas, load a TESS light curve (`.tess`) and an interval file (`.txt`), pick a polynomial order (auto 3–10 or fixed 3–10). After that, user can approximate each interval, review results, redo fit or delete specific extrema, then export `RESULTS.txt` as well as figures for eac interval.
