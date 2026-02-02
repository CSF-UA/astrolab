from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import traceback

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import logic
from .logic import FitResult, Interval
from .plot import PlotWidget


class ApproximationWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Astrolab – Extremum Approximation")
        self.resize(1300, 800)

        self.times = None
        self.mags = None
        self.intervals: List[Interval] = []
        self.results: List[FitResult] = []
        self.light_curve_path: Optional[Path] = None
        self.intervals_path: Optional[Path] = None

        self.plot = PlotWidget(self)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        file_box = QGroupBox("Input files", self)
        file_layout = QGridLayout(file_box)
        self.load_lc_btn = QPushButton("Load light curve (.tess)")
        self.load_iv_btn = QPushButton("Load intervals (.txt)")
        self.light_curve_label = QLabel("No light curve loaded")
        self.intervals_label = QLabel("No intervals loaded")
        file_layout.addWidget(self.load_lc_btn, 0, 0)
        file_layout.addWidget(self.light_curve_label, 0, 1)
        file_layout.addWidget(self.load_iv_btn, 1, 0)
        file_layout.addWidget(self.intervals_label, 1, 1)

        order_box = QGroupBox("Polynomial order", self)
        order_layout = QHBoxLayout(order_box)
        self.order_auto_btn = QPushButton("Auto 3–10")
        self.order_auto_btn.setCheckable(True)
        self.order_auto_btn.setChecked(True)
        self.order_spinner = QSpinBox()
        self.order_spinner.setRange(3, 10)
        self.order_spinner.setValue(5)
        self.order_spinner.setEnabled(False)
        order_layout.addWidget(self.order_auto_btn)
        order_layout.addWidget(QLabel("or fixed"))
        order_layout.addWidget(self.order_spinner)
        order_layout.addStretch()

        action_box = QGroupBox("Actions", self)
        action_layout = QGridLayout(action_box)
        self.approx_btn = QPushButton("Approximate all")
        self.reapprox_btn = QPushButton("Re-approx selected")
        self.delete_btn = QPushButton("Delete selected")
        self.clear_btn = QPushButton("Clear results")
        self.save_btn = QPushButton("Save RESULTS + figures")
        action_layout.addWidget(self.approx_btn, 0, 0, 1, 2)
        action_layout.addWidget(self.reapprox_btn, 1, 0)
        action_layout.addWidget(self.delete_btn, 1, 1)
        action_layout.addWidget(self.clear_btn, 2, 0)
        action_layout.addWidget(self.save_btn, 2, 1)

        self.table = QTableWidget(0, 7, self)
        self.table.setHorizontalHeaderLabels(["#", "Start", "End", "Order", "T0", "Type", "SSE"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(260)

        self.status_label = QLabel("Load a light curve and intervals to begin.")
        self.status_label.setWordWrap(True)

        left_layout.addWidget(file_box)
        left_layout.addWidget(order_box)
        left_layout.addWidget(action_box)
        left_layout.addWidget(self.table, 1)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch()

        main_layout.addWidget(left_panel, 0)
        main_layout.addWidget(self.plot, 1)

        self.load_lc_btn.clicked.connect(self._load_light_curve)
        self.load_iv_btn.clicked.connect(self._load_intervals)
        self.order_auto_btn.toggled.connect(self._toggle_order_mode)
        self.approx_btn.clicked.connect(self._run_approximation)
        self.reapprox_btn.clicked.connect(self._reapprox_selected)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.clear_btn.clicked.connect(self._clear_results)
        self.save_btn.clicked.connect(self._save_outputs)
        self.table.itemSelectionChanged.connect(self._highlight_selected_interval)

    def _toggle_order_mode(self, checked: bool) -> None:
        self.order_spinner.setEnabled(not checked)

    def _current_order_choice(self) -> Union[int, str]:
        if self.order_auto_btn.isChecked():
            return "auto"
        return int(self.order_spinner.value())

    def _load_light_curve(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select light curve (.tess)", "", "TESS files (*.tess);;All files (*.*)")
        if not path:
            return
        try:
            times, mags = logic.load_light_curve(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self.times, self.mags = times, mags
        self.light_curve_path = Path(path)
        self.light_curve_label.setText(self.light_curve_path.name)
        try:
            self.plot.set_light_curve(self.times, self.mags)
            self.plot.set_intervals(self.times, self.mags, self.intervals)
        except Exception:
            QMessageBox.critical(self, "Plot error", traceback.format_exc())
            return
        self.status_label.setText(
            f"Loaded {len(self.times):,} points. "
            f"time: {self.times.min():.6f}…{self.times.max():.6f}, "
            f"mag: {self.mags.min():.4f}…{self.mags.max():.4f}. "
            "Load intervals next."
        )

    def _load_intervals(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select intervals (.txt)", "", "Text files (*.txt);;All files (*.*)")
        if not path:
            return
        try:
            intervals = logic.load_intervals(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self.intervals = intervals
        self.intervals_path = Path(path)
        self.intervals_label.setText(self.intervals_path.name)
        if self.times is not None and self.mags is not None:
            try:
                self.plot.set_intervals(self.times, self.mags, self.intervals)
            except Exception:
                QMessageBox.critical(self, "Plot error", traceback.format_exc())
                return
        self.status_label.setText(f"Loaded {len(self.intervals)} intervals. Ready to approximate.")

    def _ensure_inputs(self) -> bool:
        if self.times is None or self.mags is None:
            QMessageBox.warning(self, "Missing data", "Load a light curve first.")
            return False
        if not self.intervals:
            QMessageBox.warning(self, "Missing intervals", "Load an intervals file.")
            return False
        return True

    def _run_approximation(self) -> None:
        if not self._ensure_inputs():
            return
        try:
            self.results = logic.approximate_all(self.times, self.mags, self.intervals, self._current_order_choice())
        except Exception as exc:
            QMessageBox.critical(self, "Approximation failed", str(exc))
            return
        self._refresh_results_table()
        self.plot.set_results(self.results)
        self.status_label.setText("Approximation finished. Review results below.")

    def _refresh_results_table(self) -> None:
        rows = logic.results_to_table_rows(self.results)
        self.table.setRowCount(len(rows))
        for r, row_data in enumerate(rows):
            for c, value in enumerate(row_data):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, item)

    def _selected_result(self) -> Optional[FitResult]:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.results):
            return None
        return self.results[row]

    def _reapprox_selected(self) -> None:
        if not self._ensure_inputs():
            return
        target = self._selected_result()
        if target is None:
            QMessageBox.information(self, "No selection", "Select a row in the results table.")
            return
        try:
            updated = logic.recompute_result(self.times, self.mags, target, self._current_order_choice())
        except Exception as exc:
            QMessageBox.critical(self, "Re-approx failed", str(exc))
            return
        self.results[target.index] = updated
        self._refresh_results_table()
        self.plot.set_results(self.results)
        self.status_label.setText(f"Recomputed interval {target.index + 1}.")

    def _delete_selected(self) -> None:
        target = self._selected_result()
        if target is None:
            QMessageBox.information(self, "No selection", "Select a row to delete.")
            return
        del self.results[target.index]
        for idx, res in enumerate(self.results):
            res.index = idx
        self._refresh_results_table()
        self.plot.set_results(self.results)
        self.plot.highlight_interval(self.times if self.times is not None else None, self.mags if self.mags is not None else None, None)
        self.status_label.setText(f"Removed interval {target.index + 1}.")

    def _clear_results(self) -> None:
        self.results = []
        self.table.setRowCount(0)
        self.plot.set_results([])
        self.status_label.setText("Results cleared.")

    def _highlight_selected_interval(self) -> None:
        target = self._selected_result()
        if self.times is None or self.mags is None:
            return
        if target is None:
            self.plot.highlight_interval(self.times, self.mags, None)
        else:
            self.plot.highlight_interval(self.times, self.mags, target.interval)

    def _save_outputs(self) -> None:
        if not self.results:
            QMessageBox.information(self, "Nothing to save", "Run an approximation first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select folder to save outputs")
        if not directory:
            return
        try:
            results_path = logic.save_results_and_figures(self.results, self.times, self.mags, directory)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.status_label.setText(f"Saved RESULTS.txt and figures to {results_path.parent}")
        QMessageBox.information(self, "Saved", f"Saved to {results_path.parent}")
