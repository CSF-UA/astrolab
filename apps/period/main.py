from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    __package__ = "apps.period"

from apps.period.ui import PeriodWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # light like the white plots, also under a dark system theme (Ubuntu): Fusion's palette
    # follows the system colour scheme unless the app asks for the light one
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    app.setPalette(app.style().standardPalette())
    app.setApplicationName("Astrolab Period Calculator")
    window = PeriodWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

