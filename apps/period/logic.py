from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import numpy as np

try:
    import lightkurve as lk
except Exception as exc:  # pragma: no cover - import availability is runtime/environment dependent
    lk = None
    _LIGHTKURVE_IMPORT_ERROR = exc
else:
    _LIGHTKURVE_IMPORT_ERROR = None


class PeriodAppError(Exception):
    """Base error for the period app."""


class DependencyError(PeriodAppError):
    """Raised when an optional dependency required by the app is missing."""


class DataLoadError(PeriodAppError):
    """Raised for search/download/file parsing failures."""


def lightkurve_available() -> bool:
    return lk is not None


def require_lightkurve() -> None:
    if lk is None:
        msg = "lightkurve is not installed. Run `uv sync` to install project dependencies (including lightkurve)."
        if _LIGHTKURVE_IMPORT_ERROR is not None:
            msg = f"{msg}\nOriginal import error: {_LIGHTKURVE_IMPORT_ERROR}"
        raise DependencyError(msg)


@dataclass
class SectorRecord:
    tic_id: str
    sector_number: int | None
    sector_label: str
    source_kind: str  # "lightkurve", "file", or "combined"
    source_label: str
    file_path: Path | None = None
    search_product_count: int = 0
    lightcurve: Any | None = None
    time: np.ndarray | None = None
    flux: np.ndarray | None = None
    periodogram: Any | None = None
    auto_period: float | None = None
    auto_frequency: float | None = None
    manual_period: float | None = None
    manual_frequency: float | None = None
    phase_shift: float = 0.0
    include_in_periodogram: bool = False
    include_in_phase: bool = False
    phase_selection_record: str = ""
    status: str = "Pending"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def display_sector(self) -> str:
        return self.sector_label

    @property
    def chosen_period(self) -> float | None:
        if self.manual_period is not None:
            return self.manual_period
        return self.auto_period

    @property
    def chosen_period_mode(self) -> str | None:
        if self.manual_period is not None:
            return "manual"
        if self.auto_period is not None:
            return "auto"
        return None

    @property
    def is_loaded(self) -> bool:
        return self.lightcurve is not None and self.time is not None and self.flux is not None

    @property
    def output_stem(self) -> str:
        meta_stem = self.meta.get("output_stem")
        if isinstance(meta_stem, str) and meta_stem.strip():
            return sanitize_filename(meta_stem)
        if self.sector_number is not None:
            return f"sector_{self.sector_number}"
        if self.file_path is not None:
            return f"local_{sanitize_filename(self.file_path.stem)}"
        return sanitize_filename(self.sector_label)


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe.strip("._") or "figure"


def normalize_tic_input(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise DataLoadError("TIC ID is empty.")
    if raw.upper().startswith("TIC"):
        suffix = raw[3:].strip()
        if not suffix:
            raise DataLoadError("TIC ID is empty.")
        return f"TIC {suffix}"
    return f"TIC {raw}"


def _scalar_float(value: Any) -> float:
    return float(getattr(value, "value", value))


def _array_float(values: Any) -> np.ndarray:
    arr = np.asarray(getattr(values, "value", values), dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def _extract_sector_number(row: Any) -> int | None:
    for key in ("sequence_number", "sector", "tess_sector"):
        try:
            value = row[key]
        except Exception:
            continue
        try:
            if value is None:
                continue
            if hasattr(value, "item"):
                value = value.item()
            return int(value)
        except Exception:
            continue

    for key in ("mission", "obs_id", "target_name", "productFilename"):
        try:
            text = str(row[key])
        except Exception:
            continue
        match = re.search(r"sector\D*(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _attach_lightcurve(record: SectorRecord, lightcurve: Any) -> SectorRecord:
    require_lightkurve()
    if lightcurve is None:
        raise DataLoadError("Light curve download returned no data.")

    lc = lightcurve
    if hasattr(lc, "remove_nans"):
        try:
            lc = lc.remove_nans()
        except Exception:
            pass

    time = _array_float(getattr(lc, "time"))
    flux = _array_float(getattr(lc, "flux"))
    mask = np.isfinite(time) & np.isfinite(flux)
    if not np.any(mask):
        raise DataLoadError("No finite time/flux data points available.")

    if not np.all(mask):
        try:
            lc = lc[mask]
        except Exception:
            lc = lk.LightCurve(time=time[mask], flux=flux[mask])
        time = time[mask]
        flux = flux[mask]

    record.lightcurve = lc
    record.time = np.asarray(time, dtype=float)
    record.flux = np.asarray(flux, dtype=float)
    record.status = f"Loaded ({record.time.size} pts)"
    return record


def search_spoc_sectors(tic_input: str) -> tuple[str, list[SectorRecord], int]:
    require_lightkurve()
    tic_id = normalize_tic_input(tic_input)
    try:
        search_result = lk.search_lightcurve(tic_id, author="SPOC")
    except Exception as exc:
        raise DataLoadError(f"Search failed for {tic_id}: {exc}") from exc

    if search_result is None or len(search_result) == 0:
        return tic_id, [], 0

    product_count = len(search_result)
    records_by_sector: dict[int, SectorRecord] = {}
    unnumbered_count = 0

    table = getattr(search_result, "table", [])
    for row in table:
        sector = _extract_sector_number(row)
        if sector is None:
            unnumbered_count += 1
            continue
        if sector in records_by_sector:
            records_by_sector[sector].search_product_count += 1
            continue

        mission = ""
        try:
            mission = str(row["mission"])
        except Exception:
            mission = "SPOC"
        records_by_sector[sector] = SectorRecord(
            tic_id=tic_id,
            sector_number=sector,
            sector_label=str(sector),
            source_kind="lightkurve",
            source_label="SPOC",
            search_product_count=1,
            status="Found",
            meta={"mission": mission},
        )

    if not records_by_sector and unnumbered_count > 0:
        records_by_sector[0] = SectorRecord(
            tic_id=tic_id,
            sector_number=None,
            sector_label="Unknown sector",
            source_kind="lightkurve",
            source_label="SPOC",
            search_product_count=unnumbered_count,
            status="Found",
        )

    records = [records_by_sector[key] for key in sorted(records_by_sector.keys())]
    return tic_id, records, product_count


def download_sector_lightcurve(record: SectorRecord) -> SectorRecord:
    require_lightkurve()
    if record.source_kind != "lightkurve":
        raise DataLoadError("This row is not a lightkurve search result.")
    if record.is_loaded:
        return record

    kwargs: dict[str, Any] = {"author": "SPOC"}
    if record.sector_number is not None:
        kwargs["sector"] = record.sector_number
    try:
        search_result = lk.search_lightcurve(record.tic_id, **kwargs)
    except Exception as exc:
        raise DataLoadError(f"Search failed while downloading {record.tic_id} sector {record.display_sector}: {exc}") from exc

    if search_result is None or len(search_result) == 0:
        raise DataLoadError(f"No downloadable data found for {record.tic_id} sector {record.display_sector}.")

    try:
        lc = search_result.download()
    except Exception as exc:
        raise DataLoadError(f"Download failed for {record.tic_id} sector {record.display_sector}: {exc}") from exc

    return _attach_lightcurve(record, lc)


def load_local_light_curve_file(path: str | Path, *, tic_id: str | None = None) -> SectorRecord:
    require_lightkurve()
    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"File not found: {p}")

    try:
        data = np.genfromtxt(p, comments="#")
    except Exception as exc:
        raise DataLoadError(f"Could not read file {p}: {exc}") from exc

    if data.ndim == 1:
        if data.size < 2:
            raise DataLoadError("File must contain at least two columns: time and flux.")
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise DataLoadError("File must contain at least two columns: time and flux.")

    time = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    mask = np.isfinite(time) & np.isfinite(flux)
    if not np.any(mask):
        raise DataLoadError("File contains no finite time/flux values.")

    lc = lk.LightCurve(time=time[mask], flux=flux[mask])

    stem = p.stem
    sector_number: int | None = None

    # Expected format example: TIC_294206429_S27_mag.tess
    tic_sector_match = re.search(r"(?i)(?:^|[_\s-])TIC[_\s-]*(\d+)[_\s-]*S(\d+)(?:[_\s-]|$)", stem)
    if tic_sector_match:
        local_tic_from_name = f"TIC {tic_sector_match.group(1)}"
        try:
            sector_number = int(tic_sector_match.group(2))
        except Exception:
            sector_number = None
        inferred_object = local_tic_from_name
    else:
        sector_match = re.search(r"(?i)(?:sector|S)[_\s-]*(\d+)", stem)
        if sector_match:
            try:
                sector_number = int(sector_match.group(1))
            except Exception:
                sector_number = None
        inferred_object = re.sub(r"(?i)[_\s-]*(?:sector|S)[_\s-]*\d+([_\s-]*mag)?", "", stem).strip(" _-")

    local_tic = (tic_id or inferred_object or stem or "Local").strip() or "Local"
    sector_label = str(sector_number) if sector_number is not None else f"Local ({p.stem})"

    record = SectorRecord(
        tic_id=local_tic,
        sector_number=sector_number,
        sector_label=sector_label,
        source_kind="file",
        source_label="TESS file",
        file_path=p,
        status="Loaded",
    )
    return _attach_lightcurve(record, lc)


def compute_periodogram(record: SectorRecord, oversample_factor: int = 10) -> SectorRecord:
    require_lightkurve()
    if not record.is_loaded:
        raise DataLoadError("Light curve is not loaded.")
    cached_oversample = record.meta.get("periodogram_oversample_factor")
    if record.periodogram is not None and cached_oversample == int(oversample_factor):
        return record
    try:
        periodogram = record.lightcurve.to_periodogram(oversample_factor=oversample_factor)
    except Exception as exc:
        raise DataLoadError(f"Could not compute periodogram: {exc}") from exc

    auto_period = _scalar_float(getattr(periodogram, "period_at_max_power"))
    if not np.isfinite(auto_period) or auto_period <= 0:
        raise DataLoadError("Automatic period from periodogram is invalid.")

    record.periodogram = periodogram
    record.auto_period = float(auto_period)
    record.auto_frequency = float(1.0 / auto_period)
    record.meta["periodogram_oversample_factor"] = int(oversample_factor)
    record.meta.pop("periodogram_arrays", None)
    record.status = f"{record.status.split(' | ')[0]} | Periodogram"
    return record


def get_periodogram_arrays(record: SectorRecord) -> tuple[np.ndarray, np.ndarray]:
    if record.periodogram is None:
        raise DataLoadError("Periodogram has not been computed.")
    cached = record.meta.get("periodogram_arrays")
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached[0], cached[1]
    freqs = _array_float(getattr(record.periodogram, "frequency"))
    powers = _array_float(getattr(record.periodogram, "power"))
    mask = np.isfinite(freqs) & np.isfinite(powers)
    if not np.any(mask):
        raise DataLoadError("Periodogram has no finite frequency/power values.")
    out = (freqs[mask], powers[mask])
    record.meta["periodogram_arrays"] = out
    return out


def select_manual_period_from_click(record: SectorRecord, clicked_frequency: float, tolerance: float = 0.05) -> float:
    if record.periodogram is None:
        raise DataLoadError("Compute the periodogram first.")
    freqs, powers = get_periodogram_arrays(record)
    if not np.isfinite(clicked_frequency):
        raise DataLoadError("Clicked position is not a valid frequency.")

    indices = np.where(np.abs(freqs - clicked_frequency) < tolerance)[0]
    if indices.size > 0:
        local_idx = int(indices[np.argmax(powers[indices])])
    else:
        local_idx = int(np.argmin(np.abs(freqs - clicked_frequency)))

    selected_frequency = float(freqs[local_idx])
    if not np.isfinite(selected_frequency) or selected_frequency <= 0:
        raise DataLoadError("Selected frequency is invalid.")

    record.manual_frequency = selected_frequency
    record.manual_period = float(1.0 / selected_frequency)
    return record.manual_period


def build_phase_curve_series(record: SectorRecord, period: float, phase_shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    if not record.is_loaded:
        raise DataLoadError("Light curve is not loaded.")
    if not np.isfinite(period) or period <= 0:
        raise DataLoadError("Period must be a positive finite number.")

    try:
        folded_lc = record.lightcurve.fold(period=period)
    except Exception as exc:
        raise DataLoadError(f"Could not fold light curve: {exc}") from exc

    phase = _array_float(getattr(folded_lc, "phase"))
    flux = _array_float(getattr(folded_lc, "flux"))
    mask = np.isfinite(phase) & np.isfinite(flux)
    if not np.any(mask):
        raise DataLoadError("Folded light curve contains no finite points.")

    phase = phase[mask]
    flux = flux[mask]
    phase = ((phase + float(phase_shift) + 0.5) % 1.0) - 0.5
    return phase, flux


def build_combined_lightcurve_record(records: list[SectorRecord], tic_id: str | None = None) -> SectorRecord:
    require_lightkurve()
    if not records:
        raise DataLoadError("No sector rows were selected for a combined periodogram.")

    times_list: list[np.ndarray] = []
    flux_list: list[np.ndarray] = []
    labels: list[str] = []
    for rec in records:
        if not rec.is_loaded or rec.time is None or rec.flux is None:
            raise DataLoadError(f"Sector {rec.display_sector} is not loaded.")
        mask = np.isfinite(rec.time) & np.isfinite(rec.flux)
        if not np.any(mask):
            continue
        times_list.append(np.asarray(rec.time[mask], dtype=float))
        flux_list.append(np.asarray(rec.flux[mask], dtype=float))
        labels.append(rec.display_sector)

    if not times_list:
        raise DataLoadError("Selected sectors contain no finite time/flux points.")

    time = np.concatenate(times_list)
    flux = np.concatenate(flux_list)
    order = np.argsort(time, kind="mergesort")
    time = time[order]
    flux = flux[order]

    lc = lk.LightCurve(time=time, flux=flux)
    label_short = ",".join(labels[:6])
    if len(labels) > 6:
        label_short = f"{label_short},+{len(labels) - 6}"

    use_tic = (tic_id or records[0].tic_id).strip() or records[0].tic_id
    combined = SectorRecord(
        tic_id=use_tic,
        sector_number=None,
        sector_label=f"Combined ({len(labels)}): {label_short}",
        source_kind="combined",
        source_label="Combined",
        status=f"Loaded ({time.size} pts)",
        meta={
            "combined_sector_labels": labels,
            "output_stem": "combined_sectors_" + "_".join(sanitize_filename(lbl) for lbl in labels[:12]),
        },
    )
    return _attach_lightcurve(combined, lc)


def build_phase_curve_series_common_reference(
    records: list[SectorRecord],
    period: float,
    phase_shift: float = 0.0,
    t0: float | None = None,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], float]:
    if not records:
        raise DataLoadError("No sector rows were selected for the phase curve.")
    if not np.isfinite(period) or period <= 0:
        raise DataLoadError("Period must be a positive finite number.")

    min_time: float | None = None
    for rec in records:
        if not rec.is_loaded or rec.time is None or rec.flux is None:
            raise DataLoadError(f"Sector {rec.display_sector} is not loaded.")
        mask = np.isfinite(rec.time) & np.isfinite(rec.flux)
        if not np.any(mask):
            continue
        rec_min = float(np.min(rec.time[mask]))
        min_time = rec_min if min_time is None else min(min_time, rec_min)

    if t0 is None:
        if min_time is None:
            raise DataLoadError("Selected sectors contain no finite points.")
        t0 = min_time

    combined_times: list[np.ndarray] = []
    combined_fluxes: list[np.ndarray] = []
    for rec in records:
        assert rec.time is not None and rec.flux is not None
        mask = np.isfinite(rec.time) & np.isfinite(rec.flux)
        if not np.any(mask):
            continue
        time = np.asarray(rec.time[mask], dtype=float)
        flux = np.asarray(rec.flux[mask], dtype=float)
        # Remove per-sector constant magnitude offset so sectors overlay on a common phase curve baseline.
        median_flux = float(np.nanmedian(flux))
        if np.isfinite(median_flux):
            flux = flux - median_flux
        combined_times.append(time)
        combined_fluxes.append(flux)

    if not combined_times:
        raise DataLoadError("Selected sectors contain no finite points.")

    all_time = np.concatenate(combined_times)
    all_flux = np.concatenate(combined_fluxes)
    phase = ((all_time - float(t0)) / float(period) + float(phase_shift) + 0.5) % 1.0 - 0.5
    order = np.argsort(phase, kind="mergesort")
    series = [("Combined", phase[order], all_flux[order])]
    return series, float(t0)


def format_period(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.5f}"


def format_shift(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.4f}"


def phase_selection_label(records: list[SectorRecord], selected_indices: list[int]) -> str:
    if not selected_indices:
        return ""
    if len(selected_indices) == len(records):
        return "all"
    labels = [records[i].display_sector for i in selected_indices]
    return ",".join(labels)


def build_figure_output_path(base_dir: str | Path, tic_id: str, kind: str, suffix: str) -> Path:
    base = Path(base_dir)
    target_dir = base / sanitize_filename(tic_id or "object")
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitize_filename(kind)}_{sanitize_filename(suffix)}.png"
    return target_dir / filename
