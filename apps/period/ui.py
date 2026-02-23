from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)

from . import logic


mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "path.simplify": True,
        "path.simplify_threshold": 0.5,
        "agg.path.chunksize": 10000,
    }
)


COL_PG = 0
COL_PHASE = 1
COL_SECTOR = 2
COL_SOURCE = 3
COL_STATUS = 4
COL_AUTO = 5
COL_MANUAL = 6
COL_SHIFT = 7
COL_PHASE_SET = 8


def _ro_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


def _checkbox_item(checked: bool) -> QTableWidgetItem:
    item = QTableWidgetItem("")
    flags = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    item.setFlags(flags)
    item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    return item


HEADER_PG_UNCHECKED = "☐ PG"
HEADER_PG_CHECKED = "☑ PG"
HEADER_PHASE_UNCHECKED = "☐ Phase"
HEADER_PHASE_CHECKED = "☑ Phase"


class PlotPanel(QWidget):
    MAX_LIGHTCURVE_POINTS = 120_000
    MAX_PERIODOGRAM_POINTS = 150_000
    MAX_PHASE_POINTS_PER_SERIES = 40_000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        self.current_plot_kind: str = "none"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    @staticmethod
    def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        n = int(len(x))
        if n <= max_points or max_points <= 0:
            return x, y
        step = max(1, n // max_points)
        xs = x[::step]
        ys = y[::step]
        if xs.size == 0:
            return x[:1], y[:1]
        if xs[-1] != x[-1] or ys[-1] != y[-1]:
            xs = np.concatenate((xs, x[-1:]))
            ys = np.concatenate((ys, y[-1:]))
        return xs, ys

    def clear(self) -> None:
        self.ax.clear()
        self.figure.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.94)
        self.ax.text(0.5, 0.5, "Load a sector to start", ha="center", va="center", transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.current_plot_kind = "none"
        self.canvas.draw_idle()
        self._reset_navigation_home()

    def _reset_navigation_home(self) -> None:
        # Reset toolbar history so Home goes to the currently displayed plot, not an old/blank view.
        nav_stack = getattr(self.toolbar, "_nav_stack", None)
        if nav_stack is not None and hasattr(nav_stack, "clear"):
            nav_stack.clear()
        if hasattr(self.toolbar, "push_current"):
            try:
                self.toolbar.push_current()
            except Exception:
                pass

    def save_current_figure(self, path: Path, dpi: int = 150) -> None:
        # Saved figures use a constant size independent of the window size.
        old_size = tuple(self.figure.get_size_inches())
        try:
            self.figure.set_size_inches(15.0, 8.0, forward=False)
            self.figure.savefig(path, dpi=dpi)
        finally:
            self.figure.set_size_inches(*old_size, forward=False)
            self.canvas.draw_idle()
        if hasattr(self.toolbar, "set_history_buttons"):
            try:
                self.toolbar.set_history_buttons()
            except Exception:
                pass

    def plot_light_curve(self, record: logic.SectorRecord) -> None:
        assert record.time is not None and record.flux is not None
        x, y = self._downsample_xy(record.time, record.flux, self.MAX_LIGHTCURVE_POINTS)
        self.ax.clear()
        self.figure.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.94)
        self.ax.set_axis_on()
        marker = "," if len(x) > 20_000 else "."
        ms = 1.0 if marker == "," else 2.0
        self.ax.plot(x, y, linestyle="None", marker=marker, color="black", markersize=ms, rasterized=True)
        if record.sector_number is not None:
            title = f"Light curve for {record.tic_id} (Sector {record.sector_number})"
        else:
            title = f"Light curve for {record.tic_id} ({record.display_sector})"
        if len(x) != len(record.time):
            title = f"{title} [display {len(x):,}/{len(record.time):,} pts]"
        self.ax.set_title(title)
        self.ax.set_xlabel("JD - 2 457 000")
        self.ax.set_ylabel("Magnitude, mmag")
        self.ax.grid(False)
        self.ax.invert_yaxis()
        self.current_plot_kind = "light_curve"
        self.canvas.draw_idle()
        self._reset_navigation_home()

    def plot_periodogram(self, record: logic.SectorRecord, preserve_view: bool = False) -> None:
        freqs, powers = logic.get_periodogram_arrays(record)
        plot_freqs, plot_powers = self._downsample_xy(freqs, powers, self.MAX_PERIODOGRAM_POINTS)
        old_xlim: tuple[float, float] | None = None
        old_ylim: tuple[float, float] | None = None
        if preserve_view and self.current_plot_kind == "periodogram":
            try:
                old_xlim = tuple(self.ax.get_xlim())
                old_ylim = tuple(self.ax.get_ylim())
            except Exception:
                old_xlim = None
                old_ylim = None
        self.ax.clear()
        if not preserve_view:
            self.figure.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.94)
        self.ax.set_axis_on()
        self.ax.plot(plot_freqs, plot_powers, color="black", lw=0.9, antialiased=False)
        if record.auto_frequency is not None and record.auto_period is not None:
            self.ax.axvline(
                x=record.auto_frequency,
                color="gray",
                linestyle="--",
                label=f"Auto: {record.auto_period:.5f} d",
            )
        if record.manual_frequency is not None and record.manual_period is not None:
            self.ax.axvline(
                x=record.manual_frequency,
                color="red",
                linestyle=":",
                label=f"Manual: {record.manual_period:.5f} d",
            )
        label_sector = record.display_sector
        title = f"Periodogram for {record.tic_id} ({label_sector})"
        if len(plot_freqs) != len(freqs):
            title = f"{title} [display {len(plot_freqs):,}/{len(freqs):,} pts]"
        self.ax.set_title(title)
        self.ax.set_xlabel("Frequency (1/d)")
        self.ax.set_ylabel("Power")
        if old_xlim is not None and old_ylim is not None:
            self.ax.set_xlim(*old_xlim)
            self.ax.set_ylim(*old_ylim)
        else:
            self.ax.set_xlim(0, 7)
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(loc="best")
        self.current_plot_kind = "periodogram"
        self.canvas.draw_idle()
        if not preserve_view:
            self._reset_navigation_home()

    def plot_phase_curve(
        self,
        source_record: logic.SectorRecord,
        phase_series: list[tuple[str, np.ndarray, np.ndarray]],
        period: float,
        phase_shift: float,
        t0: float | None = None,
    ) -> None:
        self.ax.clear()
        self.figure.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.90)
        self.ax.set_axis_on()
        for _idx, (_label, phase, flux) in enumerate(phase_series):
            phase_plot, flux_plot = self._downsample_xy(phase, flux, self.MAX_PHASE_POINTS_PER_SERIES)
            self.ax.plot(
                phase_plot,
                flux_plot,
                linestyle="None",
                marker=".",
                color="black",
                markersize=2.0,
                rasterized=True,
            )

        title = f"Phase curve for {source_record.tic_id} (P = {period:.5f} d, shift = {phase_shift:.4f})"
        if t0 is not None and np.isfinite(t0):
            title = f"{title}\nCommon reference t0 = {t0:.6f}"
        self.ax.set_title(title)
        self.ax.set_xlabel("Phase")
        self.ax.set_ylabel("Magnitude, mmag")
        self.ax.grid(False)
        self.ax.invert_yaxis()
        self.current_plot_kind = "phase_curve"
        self.canvas.draw_idle()
        self._reset_navigation_home()


class PeriodWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Astrolab – Period Calculator")
        self.resize(1400, 860)

        self.records: list[logic.SectorRecord] = []
        self.current_tic_id: str = ""
        self._table_updating = False
        self._phase_spin_sync = False
        self._last_phase_source_row: int | None = None
        self._last_phase_rows: list[int] = []
        self.current_periodogram_record: logic.SectorRecord | None = None
        self.current_periodogram_rows: list[int] = []
        self.current_periodogram_single_row: int | None = None
        self.last_phase_t0: float | None = None
        self._combined_pg_cache_key: tuple[Any, ...] | None = None
        self._combined_pg_cache_record: logic.SectorRecord | None = None
        self._last_plot_save_kind: str | None = None
        self._last_plot_save_suffix: str | None = None
        self._last_save_dir: str = str(Path("outputs") / "period")

        self.plot_panel = PlotPanel(self)
        self.plot_panel.clear()

        self._build_ui()
        self._update_current_periodogram_panel()
        self._connect_signals()
        self.plot_panel.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        if not logic.lightkurve_available():
            self._set_status("lightkurve is missing. Run `uv sync`, then `uv run period`.")

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter, 1)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        data_box = QGroupBox("Data", self)
        data_layout = QGridLayout(data_box)
        self.load_file_btn = QPushButton("Load .tess file(s)", self)
        data_layout.addWidget(self.load_file_btn, 0, 0, 1, 2)

        action_box = QGroupBox("Periodogram", self)
        action_layout = QGridLayout(action_box)
        self.periodogram_btn = QPushButton("Compute periodogram (PG checked / current row)", self)
        self.manual_pick_btn = QPushButton("Manual peak pick: OFF", self)
        self.manual_pick_btn.setCheckable(True)
        self.pg_source_label = QLabel("Current PG source: none", self)
        self.pg_source_label.setWordWrap(True)
        self.pg_auto_edit = QLineEdit(self)
        self.pg_auto_edit.setReadOnly(True)
        self.pg_auto_edit.setPlaceholderText("Auto period (copyable)")
        self.pg_manual_edit = QLineEdit(self)
        self.pg_manual_edit.setReadOnly(True)
        self.pg_manual_edit.setPlaceholderText("Manual period (copyable)")
        self.use_pg_auto_btn = QPushButton("Use PG auto for phase", self)
        self.use_pg_manual_btn = QPushButton("Use PG manual for phase", self)
        self.use_pg_manual_btn.setEnabled(False)
        self.save_figure_btn = QPushButton("Save Current Figure", self)

        action_layout.addWidget(self.periodogram_btn, 0, 0, 1, 2)
        action_layout.addWidget(self.manual_pick_btn, 1, 0, 1, 2)
        action_layout.addWidget(self.pg_source_label, 2, 0, 1, 2)
        action_layout.addWidget(QLabel("Auto period"), 3, 0)
        action_layout.addWidget(self.pg_auto_edit, 3, 1)
        action_layout.addWidget(QLabel("Manual period"), 4, 0)
        action_layout.addWidget(self.pg_manual_edit, 4, 1)
        action_layout.addWidget(self.use_pg_auto_btn, 5, 0, 1, 2)
        action_layout.addWidget(self.use_pg_manual_btn, 6, 0, 1, 2)

        phase_box = QGroupBox("Phase Controls", self)
        phase_layout = QGridLayout(phase_box)
        self.phase_period_spin = QDoubleSpinBox(self)
        self.phase_period_spin.setRange(1e-8, 1_000_000.0)
        self.phase_period_spin.setDecimals(8)
        self.phase_period_spin.setSingleStep(0.001)
        self.phase_period_spin.setValue(1.0)
        self.phase_shift_spin = QDoubleSpinBox(self)
        self.phase_shift_spin.setRange(-5.0, 5.0)
        self.phase_shift_spin.setDecimals(4)
        self.phase_shift_spin.setSingleStep(0.01)
        self.phase_shift_spin.setValue(0.0)
        self.phase_plot_btn = QPushButton("Plot phase curve (Phase checked / current row)", self)
        # Spin boxes are editable/copyable; users can type or use arrows.
        self.phase_period_spin.setKeyboardTracking(False)
        self.phase_shift_spin.setKeyboardTracking(False)
        phase_layout.addWidget(QLabel("Period for phase"), 0, 0)
        phase_layout.addWidget(self.phase_period_spin, 0, 1)
        phase_layout.addWidget(QLabel("Phase shift"), 1, 0)
        phase_layout.addWidget(self.phase_shift_spin, 1, 1)
        phase_layout.addWidget(self.phase_plot_btn, 2, 0, 1, 2)

        table_box = QGroupBox("Sectors", self)
        table_layout = QVBoxLayout(table_box)
        self.table = QTableWidget(0, 9, self)
        self.table.setHorizontalHeaderLabels(
            [
                HEADER_PG_UNCHECKED,
                HEADER_PHASE_UNCHECKED,
                "Sector",
                "Source",
                "Status",
                "Period auto",
                "Period manual",
                "Phase shift",
                "Phase sectors",
            ]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        self.table.setColumnWidth(COL_PG, 64)
        self.table.setColumnWidth(COL_PHASE, 86)
        self.table.setColumnWidth(COL_SECTOR, 100)
        self.table.setColumnWidth(COL_SOURCE, 90)
        self.table.setColumnWidth(COL_STATUS, 180)
        self.table.setColumnWidth(COL_AUTO, 120)
        self.table.setColumnWidth(COL_MANUAL, 130)
        self.table.setColumnWidth(COL_SHIFT, 90)
        self.table.setColumnWidth(COL_PHASE_SET, 220)
        table_layout.addWidget(self.table)

        self.status_label = QLabel("Search a TIC or load a local file to begin.")
        self.status_label.setWordWrap(True)

        controls_widget = QWidget(self)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        controls_layout.addWidget(data_box)
        controls_layout.addWidget(action_box)
        controls_layout.addWidget(phase_box)
        controls_layout.addWidget(self.save_figure_btn)
        controls_layout.addStretch()

        controls_scroll = QScrollArea(self)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setWidget(controls_widget)

        table_panel = QWidget(self)
        table_panel_layout = QVBoxLayout(table_panel)
        table_panel_layout.setContentsMargins(0, 0, 0, 0)
        table_panel_layout.setSpacing(8)
        table_panel_layout.addWidget(table_box, 1)
        table_panel_layout.addWidget(self.status_label)

        left_splitter = QSplitter(Qt.Orientation.Vertical, self)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.addWidget(controls_scroll)
        left_splitter.addWidget(table_panel)
        left_splitter.setStretchFactor(0, 2)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setSizes([560, 280])

        left_layout.addWidget(left_splitter, 1)

        splitter.addWidget(left)
        splitter.addWidget(self.plot_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([540, 1080])

    def _connect_signals(self) -> None:
        self.load_file_btn.clicked.connect(self._load_local_file)
        self.periodogram_btn.clicked.connect(self._compute_and_show_periodogram)
        self.manual_pick_btn.toggled.connect(self._on_manual_pick_toggled)
        self.use_pg_auto_btn.clicked.connect(self._use_current_pg_auto_for_phase)
        self.use_pg_manual_btn.clicked.connect(self._use_current_pg_manual_for_phase)
        self.save_figure_btn.clicked.connect(self._save_current_figure)
        self.phase_plot_btn.clicked.connect(self._plot_phase_curve)
        self.phase_shift_spin.valueChanged.connect(self._on_phase_shift_value_changed)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.horizontalHeader().sectionClicked.connect(self._on_table_header_clicked)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _show_error(self, title: str, message: str) -> None:
        self._set_status(message)
        QMessageBox.critical(self, title, message)

    def _with_busy_cursor(self, fn, *args, **kwargs):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return fn(*args, **kwargs)
        finally:
            QApplication.restoreOverrideCursor()

    def _update_current_periodogram_panel(self) -> None:
        rec = self.current_periodogram_record
        if rec is None:
            self.pg_source_label.setText("Current PG source: none")
            self.pg_auto_edit.clear()
            self.pg_manual_edit.clear()
            self.use_pg_auto_btn.setEnabled(False)
            self.use_pg_manual_btn.setEnabled(False)
            return
        self.pg_source_label.setText(f"Current PG source: {rec.display_sector}")
        auto_text = f"{rec.auto_period:.8f} d" if rec.auto_period is not None else "—"
        manual_text = f"{rec.manual_period:.8f} d" if rec.manual_period is not None else "—"
        self.pg_auto_edit.setText(auto_text if auto_text != "—" else "")
        self.pg_manual_edit.setText(manual_text if manual_text != "—" else "")
        self.use_pg_auto_btn.setEnabled(rec.auto_period is not None)
        self.use_pg_manual_btn.setEnabled(rec.manual_period is not None)

    def _all_rows_checked(self, column: int) -> bool:
        if not self.records:
            return False
        if column == COL_PG:
            return all(r.include_in_periodogram for r in self.records)
        if column == COL_PHASE:
            return all(r.include_in_phase for r in self.records)
        return False

    def _update_header_select_labels(self) -> None:
        item_pg = self.table.horizontalHeaderItem(COL_PG)
        item_phase = self.table.horizontalHeaderItem(COL_PHASE)
        if item_pg is not None:
            item_pg.setText(HEADER_PG_CHECKED if self._all_rows_checked(COL_PG) else HEADER_PG_UNCHECKED)
        if item_phase is not None:
            item_phase.setText(HEADER_PHASE_CHECKED if self._all_rows_checked(COL_PHASE) else HEADER_PHASE_UNCHECKED)

    def _on_table_header_clicked(self, section: int) -> None:
        if section == COL_PG:
            self._set_pg_selection_for_all(not self._all_rows_checked(COL_PG))
        elif section == COL_PHASE:
            self._set_phase_selection_for_all(not self._all_rows_checked(COL_PHASE))

    def _use_current_pg_auto_for_phase(self) -> None:
        rec = self.current_periodogram_record
        if rec is None or rec.auto_period is None:
            self._set_status("No current periodogram auto period is available.")
            return
        self.phase_period_spin.setValue(float(rec.auto_period))
        self._set_status(f"Phase period set to current periodogram auto period: {rec.auto_period:.8f} d")

    def _use_current_pg_manual_for_phase(self) -> None:
        rec = self.current_periodogram_record
        if rec is None or rec.manual_period is None:
            self._set_status("No current periodogram manual period is available.")
            return
        self.phase_period_spin.setValue(float(rec.manual_period))
        self._set_status(f"Phase period set to current periodogram manual period: {rec.manual_period:.8f} d")

    def _clear_records(self) -> None:
        self.records = []
        self.current_tic_id = ""
        self._last_phase_source_row = None
        self._last_phase_rows = []
        self.last_phase_t0 = None
        self.current_periodogram_record = None
        self.current_periodogram_rows = []
        self.current_periodogram_single_row = None
        self._combined_pg_cache_key = None
        self._combined_pg_cache_record = None
        self._table_updating = True
        self.table.setRowCount(0)
        self._table_updating = False
        self._update_header_select_labels()
        self.plot_panel.clear()
        self._update_current_periodogram_panel()

    def _populate_table(self) -> None:
        self._table_updating = True
        self.table.setRowCount(len(self.records))
        for row_idx, record in enumerate(self.records):
            self._refresh_table_row(row_idx, record)
        self._table_updating = False
        self._update_header_select_labels()
        if self.records:
            self.table.selectRow(0)

    def _refresh_table_row(self, row_idx: int, record: logic.SectorRecord) -> None:
        current_pg_item = self.table.item(row_idx, COL_PG)
        if current_pg_item is None:
            self.table.setItem(row_idx, COL_PG, _checkbox_item(record.include_in_periodogram))
        else:
            current_pg_item.setCheckState(
                Qt.CheckState.Checked if record.include_in_periodogram else Qt.CheckState.Unchecked
            )

        current_phase_item = self.table.item(row_idx, COL_PHASE)
        if current_phase_item is None:
            self.table.setItem(row_idx, COL_PHASE, _checkbox_item(record.include_in_phase))
        else:
            current_phase_item.setCheckState(
                Qt.CheckState.Checked if record.include_in_phase else Qt.CheckState.Unchecked
            )

        values = {
            COL_SECTOR: record.display_sector,
            COL_SOURCE: record.source_label,
            COL_STATUS: record.status,
            COL_AUTO: logic.format_period(record.auto_period),
            COL_MANUAL: logic.format_period(record.manual_period),
            COL_SHIFT: logic.format_shift(record.phase_shift),
            COL_PHASE_SET: record.phase_selection_record,
        }
        for col, text in values.items():
            item = self.table.item(row_idx, col)
            if item is None:
                self.table.setItem(row_idx, col, _ro_item(text))
            else:
                item.setText(text)

    def _current_row(self) -> int | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.records):
            return None
        return row

    def _current_record(self) -> tuple[int, logic.SectorRecord] | None:
        row = self._current_row()
        if row is None:
            return None
        return row, self.records[row]

    def _search_tic(self) -> None:
        self._set_status("TIC search is disabled in this version. Load local .tess file(s) instead.")

    def _load_local_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open .tess light curve file(s)",
            "",
            "TESS light curve files (*.tess)",
        )
        if not paths:
            return
        loaded_records: list[logic.SectorRecord] = []
        errors: list[str] = []
        for p in paths:
            try:
                rec = self._with_busy_cursor(logic.load_local_light_curve_file, p, tic_id=None)
            except Exception as exc:
                errors.append(f"{Path(p).name}: {exc}")
                continue
            loaded_records.append(rec)

        if not loaded_records:
            self._show_error("File load error", "\n".join(errors) if errors else "No files were loaded.")
            return

        loaded_records.sort(key=lambda r: (r.sector_number is None, r.sector_number if r.sector_number is not None else 10**9, r.display_sector))

        self._clear_records()
        self.records = loaded_records
        self.current_tic_id = loaded_records[0].tic_id
        self._populate_table()
        msg = f"Loaded {len(loaded_records)} .tess file(s). Click sector rows to preview the light curve."
        if errors:
            msg += f" {len(errors)} file(s) failed to load."
        self._set_status(msg)

    def _download_current(self) -> None:
        self._set_status("Sector download is disabled in this version. Load local .tess file(s) instead.")

    def _ensure_record_loaded(self, row_idx: int) -> bool:
        record = self.records[row_idx]
        if record.is_loaded:
            return True
        if record.source_kind == "file":
            return record.is_loaded
        try:
            self._with_busy_cursor(logic.download_sector_lightcurve, record)
        except logic.PeriodAppError as exc:
            record.status = "Download error"
            self._table_updating = True
            self._refresh_table_row(row_idx, record)
            self._table_updating = False
            self._show_error("Download error", str(exc))
            return False
        except Exception as exc:
            record.status = "Download error"
            self._table_updating = True
            self._refresh_table_row(row_idx, record)
            self._table_updating = False
            self._show_error("Download error", str(exc))
            return False

        self._table_updating = True
        self._refresh_table_row(row_idx, record)
        self._table_updating = False
        return True

    def _on_table_selection_changed(self) -> None:
        current = self._current_record()
        if current is None:
            return
        row_idx, record = current
        self._phase_spin_sync = True
        self.phase_shift_spin.setValue(float(record.phase_shift))
        self._phase_spin_sync = False

        if self._ensure_record_loaded(row_idx):
            self._show_record_light_curve(row_idx, autosave=False)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._table_updating:
            return
        row = item.row()
        if row < 0 or row >= len(self.records):
            return
        if item.column() == COL_PG:
            self.records[row].include_in_periodogram = item.checkState() == Qt.CheckState.Checked
        elif item.column() == COL_PHASE:
            self.records[row].include_in_phase = item.checkState() == Qt.CheckState.Checked
        self._update_header_select_labels()

    def _show_record_light_curve(self, row_idx: int, autosave: bool = False) -> None:
        record = self.records[row_idx]
        if not record.is_loaded:
            if not self._ensure_record_loaded(row_idx):
                return
        self.plot_panel.plot_light_curve(record)
        self._last_plot_save_kind = "light_curve"
        self._last_plot_save_suffix = record.output_stem
        self._set_status(f"Displayed light curve for {record.tic_id} ({record.display_sector}).")

    def _show_current_light_curve(self) -> None:
        current = self._current_record()
        if current is None:
            self._show_error("No row selected", "Select a sector row first.")
            return
        row_idx, _ = current
        self._show_record_light_curve(row_idx, autosave=True)

    def _compute_and_show_periodogram(self) -> None:
        current = self._current_record()
        if current is None:
            self._show_error("No row selected", "Select a sector row first.")
            return
        row_idx, _ = current
        pg_rows = self._selected_periodogram_rows()
        if not pg_rows:
            pg_rows = [row_idx]

        for idx in pg_rows:
            if not self._ensure_record_loaded(idx):
                return

        try:
            if len(pg_rows) == 1:
                pg_record = self.records[pg_rows[0]]
                self._with_busy_cursor(logic.compute_periodogram, pg_record)
                self.current_periodogram_single_row = pg_rows[0]
                self._table_updating = True
                self._refresh_table_row(pg_rows[0], pg_record)
                self._table_updating = False
            else:
                selected_records = [self.records[idx] for idx in pg_rows]
                cache_key = self._combined_pg_key(pg_rows)
                if self._combined_pg_cache_key == cache_key and self._combined_pg_cache_record is not None:
                    pg_record = self._combined_pg_cache_record
                else:
                    pg_record = self._with_busy_cursor(
                        logic.build_combined_lightcurve_record,
                        selected_records,
                        self.current_tic_id or selected_records[0].tic_id,
                    )
                    self._with_busy_cursor(logic.compute_periodogram, pg_record)
                    self._combined_pg_cache_key = cache_key
                    self._combined_pg_cache_record = pg_record
                self.current_periodogram_single_row = None
        except logic.PeriodAppError as exc:
            self._show_error("Periodogram error", str(exc))
            return
        except Exception as exc:
            self._show_error("Periodogram error", str(exc))
            return

        self.current_periodogram_record = pg_record
        self.current_periodogram_rows = list(pg_rows)
        self._update_current_periodogram_panel()
        if pg_record.auto_period is not None:
            self.phase_period_spin.setValue(float(pg_record.auto_period))

        self.plot_panel.plot_periodogram(pg_record, preserve_view=False)
        self._last_plot_save_kind = "periodogram"
        self._last_plot_save_suffix = pg_record.output_stem
        row_count = len(pg_rows)
        if row_count == 1:
            scope_text = f"sector {pg_record.display_sector}"
        else:
            scope_text = f"{row_count} selected sectors"
        self._set_status(
            f"Auto period for {scope_text}: {pg_record.auto_period:.8f} d. "
            "Manual peak pick can be enabled now."
        )

    def _on_manual_pick_toggled(self, checked: bool) -> None:
        self.manual_pick_btn.setText(f"Manual peak pick: {'ON' if checked else 'OFF'}")
        if checked:
            self._set_status("Manual peak picking is ON. Click a peak on the periodogram to set the manual period.")

    def _on_canvas_click(self, event: Any) -> None:
        if not self.manual_pick_btn.isChecked():
            return
        if event is None or event.inaxes is not self.plot_panel.ax:
            return
        if self.plot_panel.current_plot_kind != "periodogram":
            return
        if event.xdata is None:
            return
        record = self.current_periodogram_record
        if record is None:
            self._set_status("Compute a periodogram first, then enable manual peak pick.")
            return
        try:
            manual_period = logic.select_manual_period_from_click(record, float(event.xdata))
        except logic.PeriodAppError as exc:
            self._show_error("Manual period selection", str(exc))
            return
        except Exception as exc:
            self._show_error("Manual period selection", str(exc))
            return

        if self.current_periodogram_single_row is not None:
            row_idx = self.current_periodogram_single_row
            table_record = self.records[row_idx]
            table_record.manual_period = record.manual_period
            table_record.manual_frequency = record.manual_frequency
            self._table_updating = True
            self._refresh_table_row(row_idx, table_record)
            self._table_updating = False
        self._update_current_periodogram_panel()
        self.plot_panel.plot_periodogram(record, preserve_view=True)
        self._last_plot_save_kind = "periodogram"
        self._last_plot_save_suffix = record.output_stem
        self._set_status(
            f"Manual period for current periodogram source ({record.display_sector}): {manual_period:.8f} d "
            "(click again to change)."
        )

    def _selected_phase_rows(self) -> list[int]:
        selected = [idx for idx, record in enumerate(self.records) if record.include_in_phase]
        return selected

    def _selected_periodogram_rows(self) -> list[int]:
        return [idx for idx, record in enumerate(self.records) if record.include_in_periodogram]

    def _combined_pg_key(self, rows: list[int]) -> tuple[Any, ...]:
        parts: list[Any] = ["combined-pg", len(rows)]
        for idx in rows:
            rec = self.records[idx]
            if rec.time is None or len(rec.time) == 0:
                parts.extend((idx, 0, None, None))
                continue
            parts.extend(
                (
                    idx,
                    int(len(rec.time)),
                    round(float(rec.time[0]), 10),
                    round(float(rec.time[-1]), 10),
                )
            )
        return tuple(parts)

    def _set_pg_selection_for_all(self, checked: bool) -> None:
        self._table_updating = True
        for row_idx, record in enumerate(self.records):
            record.include_in_periodogram = checked
            self._refresh_table_row(row_idx, record)
        self._table_updating = False
        self._update_header_select_labels()
        if checked:
            self._set_status("All sector rows are selected for the next periodogram.")
        else:
            self._set_status("Periodogram sector selection cleared.")

    def _set_phase_selection_for_all(self, checked: bool) -> None:
        self._table_updating = True
        for row_idx, record in enumerate(self.records):
            record.include_in_phase = checked
            self._refresh_table_row(row_idx, record)
        self._table_updating = False
        self._update_header_select_labels()
        if checked:
            self._set_status("All sector rows are selected for the next phase-curve plot.")
        else:
            self._set_status("Phase-curve sector selection cleared.")

    def _on_phase_shift_value_changed(self, value: float) -> None:
        if self._phase_spin_sync:
            return
        current = self._current_record()
        if current is None:
            return
        row_idx, record = current
        record.phase_shift = float(value)
        self._table_updating = True
        self._refresh_table_row(row_idx, record)
        self._table_updating = False

        if self.plot_panel.current_plot_kind == "phase_curve" and self._last_phase_source_row == row_idx and self._last_phase_rows:
            self._plot_phase_curve(
                save_png=False,
                source_row_override=self._last_phase_source_row,
                phase_rows_override=list(self._last_phase_rows),
            )

    def _replot_phase_current_context(self) -> None:
        if self._last_phase_source_row is None or not self._last_phase_rows:
            self._set_status("No phase curve has been plotted yet.")
            return
        self._plot_phase_curve(
            save_png=True,
            source_row_override=self._last_phase_source_row,
            phase_rows_override=list(self._last_phase_rows),
        )

    def _plot_phase_curve(
        self,
        _checked: bool = False,
        save_png: bool = True,
        source_row_override: int | None = None,
        phase_rows_override: list[int] | None = None,
    ) -> None:
        if source_row_override is not None:
            source_row = source_row_override
            if source_row < 0 or source_row >= len(self.records):
                self._show_error("Phase curve", "Saved phase source row is no longer available.")
                return
        else:
            current = self._current_record()
            if current is None:
                self._show_error("No row selected", "Select a row (used for table recording/default selection).")
                return
            source_row, _ = current

        if not self._ensure_record_loaded(source_row):
            return

        period = float(self.phase_period_spin.value())
        if not np.isfinite(period) or period <= 0:
            self._show_error("Invalid period", "Phase period must be a positive number.")
            return

        if phase_rows_override is not None:
            selected_rows = [idx for idx in phase_rows_override if 0 <= idx < len(self.records)]
        else:
            selected_rows = self._selected_phase_rows()
        if not selected_rows:
            selected_rows = [source_row]
            self.records[source_row].include_in_phase = True
            self._table_updating = True
            self._refresh_table_row(source_row, self.records[source_row])
            self._table_updating = False

        for row_idx in selected_rows:
            if not self._ensure_record_loaded(row_idx):
                return

        phase_shift = float(self.phase_shift_spin.value())
        selected_records = [self.records[idx] for idx in selected_rows]
        try:
            phase_series, t0 = logic.build_phase_curve_series_common_reference(
                selected_records,
                period=period,
                phase_shift=phase_shift,
            )
        except logic.PeriodAppError as exc:
            self._show_error("Phase curve error", str(exc))
            return
        except Exception as exc:
            self._show_error("Phase curve error", str(exc))
            return

        selection_label = logic.phase_selection_label(self.records, selected_rows)
        rows_to_update = sorted(set(selected_rows + [source_row]))
        self._table_updating = True
        for row_idx in rows_to_update:
            rec = self.records[row_idx]
            rec.phase_shift = phase_shift
            rec.phase_selection_record = selection_label
            self._refresh_table_row(row_idx, rec)
        self._table_updating = False

        period_source_record = self.current_periodogram_record or self.records[source_row]
        self.plot_panel.plot_phase_curve(
            period_source_record,
            phase_series,
            period=period,
            phase_shift=phase_shift,
            t0=t0,
        )
        self._last_phase_source_row = source_row
        self._last_phase_rows = list(selected_rows)
        self.last_phase_t0 = t0
        self._last_plot_save_kind = "phase_curve"
        self._last_plot_save_suffix = f"{period_source_record.output_stem}_{selection_label or 'selected'}"

        period_source_text = (
            f"current periodogram ({period_source_record.display_sector})"
            if self.current_periodogram_record is not None
            else "manual phase period field"
        )
        msg = (
            f"Phase curve plotted with period {period:.8f} d from {period_source_text}; "
            f"phase sectors: {selection_label or self.records[source_row].display_sector}; "
            f"shared t0 = {t0:.6f}; shift = {phase_shift:.4f}."
        )
        self._set_status(msg)

    def _save_current_figure(self) -> None:
        if self.plot_panel.current_plot_kind == "none" or not self._last_plot_save_kind or not self._last_plot_save_suffix:
            self._set_status("There is no plot to save yet.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Choose folder to save current figure", self._last_save_dir)
        if not folder:
            return
        self._last_save_dir = folder

        file_name = f"{logic.sanitize_filename(self._last_plot_save_kind)}_{logic.sanitize_filename(self._last_plot_save_suffix)}.png"
        path = Path(folder) / file_name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.plot_panel.save_current_figure(path, dpi=150)
        except Exception as exc:
            self._show_error("Save figure error", str(exc))
            return
        self._set_status(f"Saved {self._last_plot_save_kind} figure: {path}")
