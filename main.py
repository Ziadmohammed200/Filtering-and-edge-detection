import sys
import os
import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDesktopWidget
import pyqtgraph as pg
from Filters import filter
from PyQt5.QtCore import Qt

Ui_MainWindow, QtBaseClass = uic.loadUiType("untitled.ui")

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # Adjust window size
        screen_size = QDesktopWidget().screenGeometry()
        width = screen_size.width()
        height = screen_size.height()
        self.setGeometry(0, 0, width, height - 100)

        # Connect Browse button to function
        self.pushButton_browse.clicked.connect(self.load_image)

        # Store image path
        self.image_path = None

        self.original_image = None  # Store the loaded image
        self.output_image = None    # Store the output image

        # Connect ComboBox to noise function
        self.comboBox_noise.currentIndexChanged.connect(self.apply_noise)

        # Connect ComboBox to filters function
        self.comboBox_lowpass.currentIndexChanged.connect(self.Apply_Filters)

        self.pushButton_reset.clicked.connect(self.reset_program)

        self.checkBox_normalize.stateChanged.connect(self.toggle_normalization)




    def load_image(self):
        """Open file dialog to load an image and display it on plot_original."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")

        if file_path:
            file_path = os.path.abspath(file_path)  # Get absolute path
            file_path = file_path.replace("\\", "/")  # Ensure correct path format

            if not os.path.exists(file_path):
                print("❌ Error: File not found!", file_path)
                return
            
            self.reset_program()

            self.original_image = cv2.imread(file_path )

            if self.original_image is None:
                print("❌ Error: Cannot load image. Check file format and permissions.")
                return

            self.display_image(self.original_image, self.plot_original)  # Show image on original plot

#############################################################################################
    def apply_noise(self):
        """Apply selected noise type from comboBox_noise."""
        if self.original_image is None:
            print("⚠ No image loaded!")
            return

        selected_noise = self.comboBox_noise.currentText()

        if selected_noise == "Uniform Noise":
            # Generate uniform noise
            noise = np.random.uniform(-100, 100, self.original_image.shape).astype(np.int16)

        elif selected_noise == "Gaussian Noise":
            # Generate Gaussian noise
            mean = 0
            stddev = 25
            noise = np.random.normal(mean, stddev, self.original_image.shape).astype(np.int16)
            
        elif selected_noise == "Salt & Pepper Noise":
            self.output_image = self.add_salt_pepper_noise(self.output_image if self.output_image is not None else self.original_image)
            self.display_image(self.output_image, self.plot_output)
            return  # Return early as no need for np.clip
        elif selected_noise == "No Noise":
            self.output_image = self.original_image
            self.display_image(self.output_image, self.plot_output)


        else:
            return  # If no valid noise type is selected, do nothing

        # Apply noise
        if self.output_image is None:
            self.output_image = np.clip(self.original_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        else:
            self.output_image = np.clip(self.output_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        self.display_image(self.output_image, self.plot_output)  # Show noisy image

        


    def add_salt_pepper_noise(self, image, salt_prob=0.02, pepper_prob=0.02):
        """Apply Salt & Pepper Noise to an image."""
        noisy_image = np.copy(image)
        total_pixels = image.size

        # Add salt (white pixels)
        num_salt = int(total_pixels * salt_prob)
        salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape]
        noisy_image[salt_coords[0], salt_coords[1]] = 255

        # Add pepper (black pixels)
        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_image

########################################################################################################################
    def Apply_Filters(self):
        """Apply selected filter type from comboBox_filters."""
        if self.original_image is None:
            print("⚠ No image loaded!")
            return

        selected_filter = self.comboBox_lowpass.currentText()

        # Ensure we are working on the correct image
        input_image = self.output_image if self.output_image is not None else self.original_image

        # Convert to grayscale if not already
        if len(input_image.shape) == 3:  # Check if the image is colored (RGB/BGR)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Apply the selected filter
        filter_functions = {
            "Average": filter.average_filter,
            "Gaussian": filter.Gaussian_Filter,
            "Median": filter.median_filter
        }

        if selected_filter in filter_functions:
            self.output_image = filter_functions[selected_filter](input_image)
            self.display_image(self.output_image, self.plot_output)
        else:
            print(f"⚠ Unknown filter selected: {selected_filter}")
################################################################################################
    def toggle_normalization(self, state):
        """Apply normalization when the checkbox is checked."""
        if state == Qt.Checked:
            # Store a backup of the current output image before normalizing
            self.backup_image = self.output_image.copy() if self.output_image is not None else self.original_image.copy()
            self.normalization()
        else:
            # Reset to the original output image
            self.output_image = self.backup_image.copy()
            self.display_image(self.output_image, self.plot_output)
    
    
    def normalization(self):
        """Apply mean normalization to an image."""
        if self.output_image is None and self.original_image is None:
            print("⚠ No image available for normalization!")
            return
        
        input_image = self.output_image if self.output_image is not None else self.original_image
        input_image = input_image.astype(np.float32)  # Convert to float for processing
        max_val = np.max(input_image)

        if max_val == 0:  # Prevent division by zero
            print("⚠ Warning: Max pixel value is zero, normalization skipped!")
            return

        # normalized_image = (image - mean) / (max_val - min_val)  # Mean normalization
        normalized_image = (input_image*255)/max_val  # Mean normalization
        self.output_image=normalized_image
        self.display_image(self.output_image, self.plot_output)
            
#########################################################################################################################
    def display_image(self, image , plot_widget):
        """Display the loaded image in plot_original (PyQtGraph)."""
        plot_widget.clear()  # Clear previous image
        img_item = pg.ImageItem(image)  # Convert image to pyqtgraph format
        img_item.setTransform(pg.QtGui.QTransform().rotate(-90))  # Rotate 90 degrees
        plot_widget.addItem(img_item)  # Add image to plot
###############################################################################################################
    def reset_program(self):
        """Reset the program to its initial state."""
        self.output_image = None  # Clear output image
        self.original_image = None  # Clear original image
        self.checkBox_normalize.setChecked(False)  # Uncheck normalization checkbox
        self.comboBox_noise.setCurrentIndex(0)  # Reset filter selection
        self.comboBox_lowpass.setCurrentIndex(0)  # Reset filter selection
        self.plot_output.clear()  # Clear output display
        self.plot_original.clear()  # Clear input display
        print("🔄 Program reset successfully!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
