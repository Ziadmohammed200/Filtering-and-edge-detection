import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QGraphicsPathItem
from PyQt5.QtGui import QPainterPath, QPen
from PyQt5.QtCore import Qt

class edge_detector:
    def __init__(self):
        self.cdf = {}

    def apply_edge_detection(self, image,kernel_x,kernel_y,kernel_size):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_x = np.array(kernel_x, dtype=np.float32)
        kernel_y = np.array(kernel_y, dtype=np.float32)
        image_padded = np.pad(image, pad_width=1, mode='constant', constant_values=0)
        height, width = image.shape
        output_image = np.zeros((height, width), dtype=np.float32)
        for y_coord in range(height):
            for x_coord in range(width):
                image_part = image_padded[y_coord:y_coord + kernel_size, x_coord:x_coord + kernel_size]
                image_part_np = np.array(image_part)
                sobel_x = np.sum(kernel_x * image_part_np)
                sobel_y = np.sum(kernel_y * image_part_np)
                resultant_sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                output_image[y_coord, x_coord] = resultant_sobel
        max_value = np.max(output_image)
        if max_value > 0:
            output_image = (output_image / max_value) * 255
        output_image = np.uint8(output_image)
        return output_image

    def apply_canny_edge_detection(self, image):
        edge = cv2.Canny(image, 100, 200)
        return edge

    def form_histogram_dict(self,image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pixels_list = {}
        for y_coord in range(image.shape[0]):
            for x_coord in range(image.shape[1]):
                if image[y_coord, x_coord] in pixels_list:
                    pixels_list[int(image[y_coord, x_coord])] += 1
                else:
                    pixels_list[int(image[y_coord, x_coord])] = 1
        return pixels_list

    def equalize(self,image, pixels_list):
        probability_density_func = {}
        cumulative_distribution_func = {}
        new_pixels_value = {}
        total_freq = sum(pixels_list.values())
        image_copy = image.copy()
        cumulative_sum = 0
        for key in pixels_list:
            probability_density_func[key] = pixels_list[key] / total_freq

        pdf_keys = sorted(probability_density_func.keys())
        for key in pdf_keys:
            cumulative_sum += probability_density_func[key]
            cumulative_distribution_func[key] = cumulative_sum
        self.cdf = cumulative_distribution_func

        for key in cumulative_distribution_func:
            new_pixels_value[key] = int(round(cumulative_distribution_func[key] * 255))

        for key in new_pixels_value:
            image_copy[image == key] = new_pixels_value[key]
        return image_copy
    def call_cdf(self):
        return self.cdf
    def plot_histogram(self,freq_dict,graph):
        graph.clear()
        graph.setLabel("left", "Frequency")
        graph.setLabel("bottom", "Pixel Intensity (0-255)")
        # graph.showGrid(x=True, y=True)

        pixel_values = list(freq_dict.keys())
        frequencies = list(freq_dict.values())

        # Plot histogram using a bar chart
        bg = pg.BarGraphItem(x=pixel_values, height=frequencies, width=1, brush="blue")
        graph.addItem(bg)



    def plot_cdf(self,cdf_dict, scene):
        scene.clear()

        # Create a path for the CDF line
        path = QPainterPath()
        sorted_items = sorted(cdf_dict.items())



        # Draw lines connecting the points
        for pixel, cdf_value in sorted_items:
            path.lineTo(pixel, cdf_value)  # Scale for visibility

        # Create a QGraphicsPathItem and add it to the scene
        path_item = QGraphicsPathItem(path)
        path_item.setPen(QPen(Qt.blue, 0.01))  # Set line color and width
        scene.addItem(path_item)







