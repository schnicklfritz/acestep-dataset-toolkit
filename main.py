# main.py
# 🚀 Pure Assembly & Execution Layer for the ACE-Step Toolkit

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from dataset_manager import DatasetManager

if __name__ == "__main__":
    # Handle high-DPI font scaling and layout adjustments across multiple displays natively
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    window = DatasetManager()
    window.show()
    sys.exit(app.exec())

