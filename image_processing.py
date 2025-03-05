import cv2
import numpy as np
import matplotlib.pyplot as plt
from  Filters import filter 

class image_process:
    def __init__(self):
        pass
    def convert_rgb_to_gray(image):
        """Manually convert an RGB image to grayscale using the luminance formula."""
        b, g, r = cv2.split(image)  # Split into channels
        grayscale = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)  # Luminance formula
        return grayscale

    def plot_hist_4color(image):
        # Split channels
        R, G, B = cv2.split(image)

        # Function to calculate histogram and CDF
        def calculate_histogram_cdf(channel):
            hist, bins = np.histogram(channel.flatten(), bins=256, range=[0,256])
            cdf = hist.cumsum()  # Compute cumulative sum
            cdf = cdf / float(cdf.max())  # Normalize CDF to [0,1]
            return hist, cdf

        # Compute histogram and CDF for each channel
        hist_R, cdf_R = calculate_histogram_cdf(R)
        hist_G, cdf_G = calculate_histogram_cdf(G)
        hist_B, cdf_B = calculate_histogram_cdf(B)

        # Plot the results
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Plot histograms
        axes[0, 0].bar(range(256), hist_R, color='red', alpha=0.7)
        axes[0, 0].set_title('Red Histogram')
        axes[0, 1].bar(range(256), hist_G, color='green', alpha=0.7)
        axes[0, 1].set_title('Green Histogram')
        axes[0, 2].bar(range(256), hist_B, color='blue', alpha=0.7)
        axes[0, 2].set_title('Blue Histogram')

        # Plot CDFs
        axes[1, 0].plot(cdf_R, color='red')
        axes[1, 0].set_title('Red CDF')
        axes[1, 1].plot(cdf_G, color='green')
        axes[1, 1].set_title('Green CDF')
        axes[1, 2].plot(cdf_B, color='blue')
        axes[1, 2].set_title('Blue CDF')

        plt.tight_layout()
        plt.show()
###################################################################################################################3

