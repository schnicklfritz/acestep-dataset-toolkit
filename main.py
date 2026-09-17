# main.py
# 🚀 Pure Assembly & Execution Layer for the ACE-Step Toolkit

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from dataset_manager import DatasetManager
from modules.wheel_guard import install_wheel_guard

if __name__ == "__main__":
    # Handle high-DPI font scaling and layout adjustments across multiple displays natively
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    # Stop the scroll wheel from silently changing combos/spinboxes/sliders the
    # cursor passes over while scrolling the dataset. Focus unlocks them.
    install_wheel_guard(app)
    window = DatasetManager()
    window.show()
    sys.exit(app.exec())

