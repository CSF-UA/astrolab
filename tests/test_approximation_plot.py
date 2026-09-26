"""Self-checks for the approximation plot's zoom and axes (needs a display).
Run: xvfb-run -a uv run python tests/test_approximation_plot.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from apps.approximation.plot import PlotWidget  # noqa: E402


def rect(w):
    r = w.view.camera.rect
    return r.left, r.bottom, r.width, r.height


def test_zoom_keeps_flipped_magnitude_axis():
    w = PlotWidget()
    t = np.linspace(1325, 1350, 2000)
    w.set_light_curve(t, 60 * np.exp(-((t - 1337) / 0.1) ** 2))  # an eclipse in magnitudes
    x0, y0, w0, h0 = rect(w)
    assert h0 < 0  # fainter (larger magnitude) at the bottom
    for zx, zy in ((2.0, 1.0), (1.0, 2.0), (0.5, 3.0)):
        w.reset_view()
        w.set_zoom_factors(zx, zy)
        x, y, wd, h = rect(w)
        assert np.isclose(wd, w0 / zx) and np.isclose(h, h0 / zy), (zx, zy, wd, h)  # was h = 1e-9: y collapsed
        assert np.isclose(x + wd / 2, x0 + w0 / 2) and np.isclose(y + h / 2, y0 + h0 / 2)  # zoom about the centre


def test_axis_labels_follow_a_resize():
    w = PlotWidget()
    w.resize(900, 500)
    w.show()
    t = np.linspace(1325, 1350, 2000)
    w.set_light_curve(t, 60 * np.exp(-((t - 1337) / 0.1) ** 2))
    w.resize(1500, 800)  # a maximised window: the labels were left over the stretched axes
    for _ in range(10):
        app.processEvents()
    for ax, k in ((w.x_axis, 0), (w.y_axis, 1)):
        shown = ax.node_transform(w.view.scene).map(ax._axis_ends())[:, k]
        assert np.allclose(ax.axis.domain, shown), (ax.orientation, ax.axis.domain, shown)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
