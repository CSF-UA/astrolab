from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import traceback

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import logic
from .logic import FitResult, Interval
from .plot import PlotWidget


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None, elide_mode: Qt.TextElideMode = Qt.ElideMiddle) -> None:
        super().__init__(parent)
        self._full_text = text
        self._elide_mode = elide_mode
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setFullText(text)

    def setFullText(self, text: str) -> None:
        self._full_text = text
        self._update_elision()

    def fullText(self) -> str:
        return self._full_text

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_elision()

    def _update_elision(self) -> None:
        metrics = self.fontMetrics()
        elided = metrics.elidedText(self._full_text, self._elide_mode, max(0, self.contentsRect().width()))
        super().setText(elided)


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
        self.light_curve_label = ElidedLabel("No light curve loaded", self)
        self.intervals_label = ElidedLabel("No intervals loaded", self)
        file_layout.addWidget(self.load_lc_btn, 0, 0)
        file_layout.addWidget(self.light_curve_label, 0, 1)
        file_layout.addWidget(self.load_iv_btn, 1, 0)
        file_layout.addWidget(self.intervals_label, 1, 1)
        file_layout.setColumnStretch(1, 1)

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

        view_box = QGroupBox("View controls", self)
        view_layout = QGridLayout(view_box)
        self.x_zoom_value = QDoubleSpinBox(self)
        self.x_zoom_value.setRange(0.2, 5.0)
        self.x_zoom_value.setSingleStep(0.1)
        self.x_zoom_value.setDecimals(2)
        self.x_zoom_value.setValue(1.0)
        self.x_zoom_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.x_zoom_slider.setRange(20, 500)
        self.x_zoom_slider.setSingleStep(5)
        self.x_zoom_slider.setPageStep(10)
        self.x_zoom_slider.setValue(100)
        self.x_zoom_plus_btn = QPushButton("+")
        self.x_zoom_minus_btn = QPushButton("-")

        self.y_zoom_value = QDoubleSpinBox(self)
        self.y_zoom_value.setRange(0.2, 5.0)
        self.y_zoom_value.setSingleStep(0.1)
        self.y_zoom_value.setDecimals(2)
        self.y_zoom_value.setValue(1.0)
        self.y_zoom_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.y_zoom_slider.setRange(20, 500)
        self.y_zoom_slider.setSingleStep(5)
        self.y_zoom_slider.setPageStep(10)
        self.y_zoom_slider.setValue(100)
        self.y_zoom_plus_btn = QPushButton("+")
        self.y_zoom_minus_btn = QPushButton("-")

        self.reset_view_btn = QPushButton("Reset view")

        view_layout.addWidget(QLabel("X:"), 0, 0)
        view_layout.addWidget(self.x_zoom_value, 0, 1)
        view_layout.addWidget(self.x_zoom_slider, 1, 0, 1, 2)
        view_layout.addWidget(self.x_zoom_plus_btn, 2, 0)
        view_layout.addWidget(self.x_zoom_minus_btn, 2, 1)

        view_layout.addWidget(QLabel("Y:"), 3, 0)
        view_layout.addWidget(self.y_zoom_value, 3, 1)
        view_layout.addWidget(self.y_zoom_slider, 4, 0, 1, 2)
        view_layout.addWidget(self.y_zoom_plus_btn, 5, 0)
        view_layout.addWidget(self.y_zoom_minus_btn, 5, 1)
        view_layout.addWidget(self.reset_view_btn, 6, 0, 1, 2)

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
        left_layout.addWidget(view_box)
        left_layout.addWidget(self.table, 1)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch()

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 920])
        main_layout.addWidget(splitter)

        self.load_lc_btn.clicked.connect(self._load_light_curve)
        self.load_iv_btn.clicked.connect(self._load_intervals)
        self.order_auto_btn.toggled.connect(self._toggle_order_mode)
        self.approx_btn.clicked.connect(self._run_approximation)
        self.reapprox_btn.clicked.connect(self._reapprox_selected)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.clear_btn.clicked.connect(self._clear_results)
        self.save_btn.clicked.connect(self._save_outputs)
        self.x_zoom_slider.valueChanged.connect(self._x_zoom_slider_changed)
        self.y_zoom_slider.valueChanged.connect(self._y_zoom_slider_changed)
        self.x_zoom_value.valueChanged.connect(self._x_zoom_spin_changed)
        self.y_zoom_value.valueChanged.connect(self._y_zoom_spin_changed)
        self.x_zoom_plus_btn.clicked.connect(self._x_zoom_in)
        self.x_zoom_minus_btn.clicked.connect(self._x_zoom_out)
        self.y_zoom_plus_btn.clicked.connect(self._y_zoom_in)
        self.y_zoom_minus_btn.clicked.connect(self._y_zoom_out)
        self.reset_view_btn.clicked.connect(self.plot.reset_view)
        self.reset_view_btn.clicked.connect(self._reset_zoom_controls)
        self.table.itemSelectionChanged.connect(self._highlight_selected_interval)

    def _toggle_order_mode(self, checked: bool) -> None:
        self.order_spinner.setEnabled(not checked)

    @staticmethod
    def _zoom_slider_to_value(slider_value: int) -> float:
        return slider_value / 100.0

    @staticmethod
    def _zoom_value_to_slider(zoom_value: float) -> int:
        return int(round(zoom_value * 100.0))

    def _apply_zoom_controls(self) -> None:
        self.plot.set_zoom_factors(float(self.x_zoom_value.value()), float(self.y_zoom_value.value()))

    def _x_zoom_slider_changed(self, slider_value: int) -> None:
        zoom_value = self._zoom_slider_to_value(slider_value)
        self.x_zoom_value.blockSignals(True)
        self.x_zoom_value.setValue(zoom_value)
        self.x_zoom_value.blockSignals(False)
        self._apply_zoom_controls()

    def _y_zoom_slider_changed(self, slider_value: int) -> None:
        zoom_value = self._zoom_slider_to_value(slider_value)
        self.y_zoom_value.blockSignals(True)
        self.y_zoom_value.setValue(zoom_value)
        self.y_zoom_value.blockSignals(False)
        self._apply_zoom_controls()

    def _x_zoom_spin_changed(self, zoom_value: float) -> None:
        self.x_zoom_slider.blockSignals(True)
        self.x_zoom_slider.setValue(self._zoom_value_to_slider(float(zoom_value)))
        self.x_zoom_slider.blockSignals(False)
        self._apply_zoom_controls()

    def _y_zoom_spin_changed(self, zoom_value: float) -> None:
        self.y_zoom_slider.blockSignals(True)
        self.y_zoom_slider.setValue(self._zoom_value_to_slider(float(zoom_value)))
        self.y_zoom_slider.blockSignals(False)
        self._apply_zoom_controls()

    def _x_zoom_in(self) -> None:
        self.x_zoom_value.setValue(min(self.x_zoom_value.value() + self.x_zoom_value.singleStep(), self.x_zoom_value.maximum()))

    def _x_zoom_out(self) -> None:
        self.x_zoom_value.setValue(max(self.x_zoom_value.value() - self.x_zoom_value.singleStep(), self.x_zoom_value.minimum()))

    def _y_zoom_in(self) -> None:
        self.y_zoom_value.setValue(min(self.y_zoom_value.value() + self.y_zoom_value.singleStep(), self.y_zoom_value.maximum()))

    def _y_zoom_out(self) -> None:
        self.y_zoom_value.setValue(max(self.y_zoom_value.value() - self.y_zoom_value.singleStep(), self.y_zoom_value.minimum()))

    def _reset_zoom_controls(self) -> None:
        self.x_zoom_value.blockSignals(True)
        self.y_zoom_value.blockSignals(True)
        self.x_zoom_slider.blockSignals(True)
        self.y_zoom_slider.blockSignals(True)
        self.x_zoom_value.setValue(1.0)
        self.y_zoom_value.setValue(1.0)
        self.x_zoom_slider.setValue(100)
        self.y_zoom_slider.setValue(100)
        self.x_zoom_value.blockSignals(False)
        self.y_zoom_value.blockSignals(False)
        self.x_zoom_slider.blockSignals(False)
        self.y_zoom_slider.blockSignals(False)

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
        self.light_curve_label.setFullText(self.light_curve_path.name)
        self.light_curve_label.setToolTip(str(self.light_curve_path))
        try:
            self.plot.set_light_curve(self.times, self.mags)
            self.plot.set_intervals(self.times, self.mags, self.intervals)
            self._reset_zoom_controls()
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
        self.intervals_label.setFullText(self.intervals_path.name)
        self.intervals_label.setToolTip(str(self.intervals_path))
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
