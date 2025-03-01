import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg

class TwoGraphsWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Histogram and Distribution")
        self.setGeometry(100, 100, 800, 600)

        # Create a main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout()
        self.main_widget.setLayout(layout)

        # Create two PyQtGraph PlotWidgets
        self.histogram_graph = pg.PlotWidget(title="Histogram ")
        self.distribution_graph = pg.PlotWidget(title="Distribution")

        # Add graphs to the layout
        layout.addWidget(self.histogram_graph)
        layout.addWidget(self.distribution_graph)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TwoGraphsWindow()
    window.show()
    sys.exit(app.exec_())
