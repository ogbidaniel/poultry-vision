#!/usr/bin/env python3
"""
PoseLabeler - Multi-Animal Pose Annotation Tool
Label images frames.

Author: Daniel Ogbuigwe
"""

import sys
from pathlib import Path

# Add labeler directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui.main_window import MainWindow


def main():
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("PoseLabeler")
    app.setOrganizationName("PoultryVision")
    app.setApplicationVersion("1.0.0")
    
    # Set application style
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
