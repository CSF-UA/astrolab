from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vispy import app as vispy_app
vispy_app.use_app("pyside6")

from vispy import scene
from vispy.scene import visuals

from .logic import FitResult, Interval


class PlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="white", parent=self)
        self.canvas.native.setParent(self)

        grid = self.canvas.central_widget.add_grid(margin=10)
        self.view = grid.add_view(row=0, col=1, camera="panzoom")
        self.view.camera.interactive = True

        self.x_axis = scene.AxisWidget(orientation="bottom")
        self.y_axis = scene.AxisWidget(orientation="left")
        self.x_axis.stretch = (1, 0.1)
        self.y_axis.stretch = (0.1, 1)
        grid.add_widget(scene.Widget(), row=1, col=0)
        grid.add_widget(self.y_axis, row=0, col=0)
        grid.add_widget(self.x_axis, row=1, col=1)
        self.x_axis.link_view(self.view)
        self.y_axis.link_view(self.view)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)

        empty = np.empty((0, 2), dtype=float)

        self.light_curve = visuals.Markers(parent=self.view.scene)
        self.light_curve.set_gl_state(depth_test=False)
        self.interval_lines: list[visuals.Line] = []
        self.selected_line = visuals.Line(empty, color=(0.10, 0.40, 0.85, 1), width=3, parent=self.view.scene)
        self.selected_line.set_gl_state(depth_test=False)
        self.extrema_markers = visuals.Markers(parent=self.view.scene)
        self.extrema_markers.set_gl_state(depth_test=False)
        self.extrema_lines = visuals.Line(empty, color=(0.80, 0.20, 0.20, 0.7), width=2, connect="segments", parent=self.view.scene)
        self.extrema_lines.set_gl_state(depth_test=False)

        self.y_limits = (0.0, 1.0)

        self.light_curve.set_data(empty, face_color=(0.1, 0.1, 0.1, 0.95), edge_color=None, size=6)
        self.selected_line.set_data(empty)
        self.extrema_markers.set_data(empty)
        self.extrema_lines.set_data(empty)

    def set_light_curve(self, times: np.ndarray, mags: np.ndarray) -> None:
        if len(times) == 0:
            return
        inv_mags = -mags
        pos = np.column_stack((times, inv_mags)).astype(np.float32, copy=False)
        self.light_curve.set_data(pos, face_color=(0.1, 0.1, 0.1, 0.95), edge_color=None, size=6)
        ymin, ymax = float(np.min(inv_mags)), float(np.max(inv_mags))
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        span_x = float(times.max() - times.min())
        span_y = float(ymax - ymin)
        pad_x = span_x * 0.02 if span_x > 0 else 1.0
        pad_y = span_y * 0.05 if span_y > 0 else 1.0
        self.y_limits = (ymin - pad_y, ymax + pad_y)
        x_min = float(times.min()) - pad_x
        x_max = float(times.max()) + pad_x
        y_min, y_max = self.y_limits
        self.view.camera.rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        self.canvas.update()

    def set_intervals(self, times: np.ndarray, mags: np.ndarray, intervals: list[Interval]) -> None:
        for line in self.interval_lines:
            line.parent = None
        self.interval_lines = []
        if len(times) == 0 or not intervals:
            return
        for iv in intervals:
            seg_x = times[iv.start : iv.end]
            seg_y = -mags[iv.start : iv.end]
            line = visuals.Line(
                np.column_stack((seg_x, seg_y)).astype(np.float32, copy=False),
                color=(0.50, 0.50, 0.50, 0.9),
                width=2,
                parent=self.view.scene,
            )
            line.set_gl_state(depth_test=False)
            self.interval_lines.append(line)
        self.canvas.update()

    def set_results(self, results: list[FitResult]) -> None:
        if not results:
            self.extrema_markers.set_data(np.empty((0, 2)))
            self.extrema_lines.set_data(np.empty((0, 2)))
            self.canvas.update()
            return
        points = np.array([[res.t0, -res.y_at_t0] for res in results], dtype=np.float32)
        self.extrema_markers.set_data(points, edge_color=(0.8, 0.2, 0.2, 1), face_color=(0.95, 0.75, 0.75, 0.9), size=8)
        lines = []
        ymin, ymax = self.y_limits
        for res in results:
            lines.append([res.t0, ymin])
            lines.append([res.t0, ymax])
        self.extrema_lines.set_data(np.array(lines, dtype=np.float32))
        self.canvas.update()

    def highlight_interval(self, times: np.ndarray, mags: np.ndarray, interval: Interval | None) -> None:
        if interval is None:
            self.selected_line.set_data(np.empty((0, 2)))
            self.canvas.update()
            return
        seg_x = times[interval.start : interval.end]
        seg_y = -mags[interval.start : interval.end]
        if len(seg_x) == 0:
            self.selected_line.set_data(np.empty((0, 2)))
            self.canvas.update()
            return
        data = np.column_stack((seg_x, seg_y)).astype(np.float32, copy=False)
        self.selected_line.set_data(data)
        self.canvas.update()
