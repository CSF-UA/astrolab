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

        axis_kwargs = {
            "axis_color": (0.0, 0.0, 0.0, 1.0),
            "tick_color": (0.35, 0.35, 0.35, 1.0),
            "text_color": "black",
            "tick_font_size": 10,
            "axis_font_size": 12,
            "axis_label_margin": 45,
        }
        self.x_axis = scene.AxisWidget(orientation="bottom", axis_label="Time (JD-2457000)", **axis_kwargs)
        self.y_axis = scene.AxisWidget(orientation="left", axis_label="Magnitude (mmag)", **axis_kwargs)
        self.x_axis.height_min = 70
        self.y_axis.width_min = 90
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
        self.interval_lines: list[visuals.Line] = []  # kept for compatibility; we no longer draw connecting lines
        self.selected_region = visuals.Rectangle(
            center=(0.0, 0.0),
            width=1.0,
            height=1.0,
            color=(0.6, 0.6, 0.6, 0.15),
            border_color=(0.4, 0.4, 0.4, 0.4),
            parent=self.view.scene,
        )
        self.selected_region.visible = False
        self.extrema_markers = visuals.Markers(parent=self.view.scene)
        self.extrema_markers.set_gl_state(depth_test=False)
        self.extrema_lines = visuals.Line(empty, color=(0.80, 0.20, 0.20, 0.7), width=2, connect="segments", parent=self.view.scene)
        self.extrema_lines.set_gl_state(depth_test=False)
        self.fit_lines: list[visuals.Line] = []

        self.y_limits = (0.0, 1.0)
        self.times: np.ndarray | None = None
        self.mags: np.ndarray | None = None
        self.interval_colors = [
            (0.86, 0.37, 0.34, 0.9),
            (0.31, 0.53, 0.85, 0.9),
            (0.33, 0.73, 0.51, 0.9),
            (0.87, 0.59, 0.32, 0.9),
            (0.69, 0.40, 0.82, 0.9),
            (0.21, 0.65, 0.75, 0.9),
            (0.92, 0.46, 0.73, 0.9),
            (0.55, 0.55, 0.55, 0.9),
        ]

        self.light_curve.set_data(empty, face_color=(0.1, 0.1, 0.1, 0.95), edge_color=None, size=6)
        self.extrema_markers.set_data(empty)
        self.extrema_lines.set_data(empty)

    def set_light_curve(self, times: np.ndarray, mags: np.ndarray) -> None:
        if len(times) == 0:
            return
        inv_mags = -mags
        pos = np.column_stack((times, inv_mags)).astype(np.float32, copy=False)
        self.times = times
        self.mags = mags
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
        # we no longer draw connecting lines; only recolor points by interval
        if self.times is None:
            self.times = times
        if self.mags is None:
            self.mags = mags
        if len(times) == 0 or not intervals:
            return
        # color points by interval
        colors = np.tile(np.array([[0.1, 0.1, 0.1, 0.85]], dtype=np.float32), (len(times), 1))
        for idx, iv in enumerate(intervals):
            c = self.interval_colors[idx % len(self.interval_colors)]
            colors[iv.start : iv.end] = c
        if self.times is not None and self.mags is not None:
            inv_mags = -self.mags
            pos = np.column_stack((self.times, inv_mags)).astype(np.float32, copy=False)
            self.light_curve.set_data(pos, face_color=colors, edge_color=None, size=6)
        self.canvas.update()

    def set_results(self, results: list[FitResult]) -> None:
        for ln in self.fit_lines:
            ln.parent = None
        self.fit_lines = []
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

        # draw fit curves per interval
        if self.times is not None and self.mags is not None:
            for res in results:
                seg_x = self.times[res.interval.start : res.interval.end]
                dense_x = np.linspace(seg_x.min(), seg_x.max(), 600, dtype=np.float32)
                dense_x_centered = dense_x - res.x_mean
                dense_y = np.polyval(res.coefficients, dense_x_centered)
                dense_y = -dense_y
                color_idx = res.index % len(self.interval_colors)
                ln = visuals.Line(
                    np.column_stack((dense_x, dense_y)).astype(np.float32, copy=False),
                    color=self.interval_colors[color_idx],
                    width=3.5,
                    parent=self.view.scene,
                )
                ln.set_gl_state(depth_test=False)
                self.fit_lines.append(ln)

        self.canvas.update()

    def highlight_interval(self, times: np.ndarray, mags: np.ndarray, interval: Interval | None) -> None:
        if interval is None:
            self.canvas.update()
            self.selected_region.visible = False
            return
        seg_x = times[interval.start : interval.end]
        seg_y = -mags[interval.start : interval.end]
        if len(seg_x) == 0:
            self.canvas.update()
            return
        y0, y1 = self.y_limits
        center = ((seg_x.min() + seg_x.max()) / 2.0, (y0 + y1) / 2.0)
        self.selected_region.center = center
        self.selected_region.width = float(seg_x.max() - seg_x.min())
        self.selected_region.height = float(y1 - y0)
        self.selected_region.visible = True
        self.canvas.update()
