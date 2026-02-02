from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from vispy import app as vispy_app

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    __package__ = "apps.approximation"

from apps.approximation.ui import ApproximationWindow


def main() -> None:
    vispy_app.use_app("pyside6")
    app = QApplication(sys.argv)
    app.setApplicationName("Astrolab Approximation")
    window = ApproximationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
