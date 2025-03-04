import sys
import os
import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDesktopWidget
import pyqtgraph as pg
from Filters import filter
from Edge_detector import edge_detector
from PyQt5.QtCore import Qt
from image_processing import image_process
from Histogram_eq_graphs import TwoGraphsWindow

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

        # Connect Gray Scale button 
        self.pushButton_grayscale.clicked.connect(self.convert2gray)

        self.pushButton_disthist.clicked.connect(self.plot_histogram_and_distribution)


        # Store image path
        self.image_path = None

        self.original_image = None  # Store the loaded image
        self.output_image = None    # Store the output image
        self.last_state = None
        self.second_image=None
        self.iscolored=False
        self.two_graphs = TwoGraphsWindow()

        # Connect ComboBox to noise function
        self.comboBox_noise.currentIndexChanged.connect(self.apply_noise)

        self.comboBox_freq.currentIndexChanged.connect(self.freq_filter)


        # Connect ComboBox to filters function
        self.comboBox_lowpass.currentIndexChanged.connect(self.Apply_Filters)

        self.pushButton_reset.clicked.connect(self.reset_program)

        self.checkBox_normalize.stateChanged.connect(self.toggle_normalization)

        self.checkBox_toggle.stateChanged.connect(self.toggle_switch)

        # Connect ComboBox to edge detection function
        self.comboBox_edge.currentIndexChanged.connect(self.apply_edge_detection)
        self.comboBox_edge.setItemText(0,'No edge detection')

        self.checkBox_equalize.stateChanged.connect(self.toggle_equalize)

        self.spinBox_cutoff.valueChanged.connect(self.update_cutoff)
        self.spinBox_cutoff.setValue(30)
        self.cutoff_value=30
        self.Switch=False


        # Kernels initialization
        self.kernel_sobel_x =[[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]

        self.kernel_sobel_y = [[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]]

        self.kernel_prewitt_x = [[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]]

        self.kernel_prewitt_y = [[-1,-1,-1],
                                 [0,0,0],
                                 [1,1,1]]

        self.kernel_robert_x = [[1, 0],
                                [0, -1]]

        self.kernel_robert_y = [[0, 1],
                                [-1, 0]]

        self.edge_detector = edge_detector()

        self.freq_dict = {}




    def load_image(self):
        """Open file dialog to load an image and display it accordingly."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")

        if file_path:
            file_path = os.path.abspath(file_path).replace("\\", "/")  # Normalize file path

            if not os.path.exists(file_path):
                print("❌ Error: File not found!", file_path)
                return

            # Track button state
            if not hasattr(self, 'image_stage'):
                self.image_stage = 0  # Initialize stage tracker

            if self.image_stage == 0:
                # First load: Show image in plot_original
                self.reset_program()
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    print("❌ Error: Cannot load image. Check file format and permissions.")
                    return
                self.display_image(self.original_image, self.plot_original)
                self.pushButton_browse.setText("Browse Another Image And Hyprid It")  # Change button text
                self.image_stage = 1  # Move to next stage

            elif self.image_stage == 1:
                # Second load: Show image in plot_second and reset button text
                self.second_image = cv2.imread(file_path)
                if self.second_image is None:
                    print("❌ Error: Cannot load image. Check file format and permissions.")
                    return
                self.display_image(self.second_image, self.plot_second)
                self.pushButton_browse.setText("Browse")  # Reset button text
                self.image_stage = 0 # Move to next stage
                self.Hyprid()


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
            "Gaussian": filter.gaussian_filter,
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

    def apply_edge_detection(self,index):
        if index == 0:
            self.output_image = self.original_image
        elif index == 1:
            if self.output_image is None:
                self.output_image = self.edge_detector.apply_edge_detection(self.original_image,self.kernel_sobel_x,self.kernel_sobel_y,kernel_size=3)
            else:
                self.output_image = self.edge_detector.apply_edge_detection(self.output_image,self.kernel_sobel_x,self.kernel_sobel_y,kernel_size=3)
        elif index == 2 :
            if self.output_image is None:
                self.output_image = self.edge_detector.apply_edge_detection(self.original_image,self.kernel_robert_x,self.kernel_robert_y,kernel_size=2)
            else:
                self.output_image = self.edge_detector.apply_edge_detection(self.output_image,self.kernel_robert_x,self.kernel_robert_y,kernel_size=2)
        elif index == 3 :
            if self.output_image is None:
                self.output_image = self.edge_detector.apply_canny_edge_detection(self.original_image)
            else:
                self.output_image = self.edge_detector.apply_canny_edge_detection(self.output_image)
        else:
            if self.output_image is None:
                self.output_image = self.edge_detector.apply_edge_detection(self.original_image, self.kernel_prewitt_x,
                                                                            self.kernel_prewitt_y,kernel_size=3)
            else:
                self.output_image = self.edge_detector.apply_edge_detection(self.output_image, self.kernel_prewitt_x,
                                                                            self.kernel_prewitt_y,kernel_size=3)
        self.display_image(self.output_image, self.plot_output)
###########################################################################################################
    def plot_histogram_and_distribution(self):
        self.freq_dict = self.edge_detector.form_histogram_dict(self.original_image)
        self.edge_detector.plot_histogram(self.freq_dict,self.two_graphs.histogram_graph)
        cdf = self.edge_detector.call_cdf()
        self.edge_detector.plot_cdf(cdf,self.two_graphs.distribution_graph)
        self.two_graphs.show()
    
    def toggle_equalize(self, state):
        """Apply equalize when the checkbox is checked."""
        if state == Qt.Checked:
            # Store a backup of the current output image before normalizing
            self.backup_image = self.output_image.copy() if self.output_image is not None else self.original_image.copy()
            self.equalize()
        else:
            # Reset to the original output image
            self.output_image = self.backup_image.copy()
            self.display_image(self.output_image, self.plot_output)

    def equalize(self):
        if self.original_image is not None:
            self.freq_dict = self.edge_detector.form_histogram_dict(self.original_image)
            self.output_image = self.edge_detector.equalize(self.original_image,self.freq_dict)
        self.display_image(self.output_image, self.plot_output)
####################################################################################################
    def convert2gray(self):
        """Apply selected filter type from comboBox_filters."""
        if self.original_image is None:
            print("⚠ No image loaded!")
            return
        self.output_image=image_process.convert_rgb_to_gray(self.original_image)
        self.iscolored=True
        self.display_image(self.output_image, self.plot_output)

    def darw_hist(self):
        if self.iscolored == True:
            image_process.plot_histogram_with_cdf(self.original_image)

########################################################################################################
    def update_cutoff(self):
        self.cutoff_value = self.spinBox_cutoff.value()
        print(f"Updated Cutoff: {self.cutoff_value}")
    
    def freq_filter(self):
        """Apply selected filter type from comboBox_filters."""
        if self.original_image is None:
            print("⚠ No image loaded!")
            return

        selected_filter = self.comboBox_freq.currentText()

        # Ensure we are working on the correct image
        input_image = self.output_image if self.output_image is not None else self.original_image

        # Convert to grayscale if not already
        if len(input_image.shape) == 3:  # Check if the image is colored (RGB/BGR)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        if selected_filter == "Low Pass Filter":
            self.output_image=filter.Frequncey_filter(input_image,"low",self.cutoff_value)
        elif selected_filter == "High Pass Filter":
            self.output_image=filter.Frequncey_filter(input_image,"high",self.cutoff_value)
        elif selected_filter == "Freq Domain Filter":
            self.output_image=self.original_image

        self.display_image(self.output_image, self.plot_output)
###########################################################################################

    def toggle_switch(self, state):
        if state == Qt.Checked:
            self.switch = True
        else:
            self.switch = False
    def Hyprid(self):
        if self.original_image is None or self.second_image is None:
            print("⚠ Missing one or two images")
            return
        

        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.second_image = cv2.cvtColor(self.second_image, cv2.COLOR_BGR2GRAY)


        
        Hypird_image=filter.create_hybrid_image(self.original_image, self.second_image,switch=self.Switch)
        self.display_image(Hypird_image, self.plot_hyprid)


    
    
        


###########################################################################################################
    def reset_program(self):
        """Reset the program to its initial state."""
        self.pushButton_browse.setText("Browse")  # Reset button text
        self.image_stage=0
        self.iscolored=False
        self.output_image = None  # Clear output image
        self.original_image = None  # Clear original image
        self.second_image=None    # Clear second image
        self.cutoff_value=30
        self.Switch=False
        self.checkBox_normalize.setChecked(False)  # Uncheck normalization checkbox
        self.comboBox_noise.setCurrentIndex(0)  # Reset filter selection
        self.comboBox_lowpass.setCurrentIndex(0)  # Reset filter selection
        self.comboBox_edge.setCurrentIndex(0)
        self.checkBox_equalize.setChecked(False)
        self.plot_output.clear()  # Clear output display
        self.plot_original.clear()  # Clear input display
        self.plot_second.clear()
        self.plot_hyprid.clear()

        print("🔄 Program reset successfully!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
