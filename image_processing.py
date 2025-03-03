import cv2
import numpy as np
import matplotlib.pyplot as plt

class image_process:
    def __init__(self):
        pass
    def convert_rgb_to_gray(image):
        """Manually convert an RGB image to grayscale using the luminance formula."""
        b, g, r = cv2.split(image)  # Split into channels
        grayscale = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)  # Luminance formula
        return grayscale

    def plot_histogram_with_cdf(image):
        """Plot R, G, B histograms and their cumulative distribution functions (CDFs)."""
        
        # Split channels
        b, g, r = cv2.split(image)  # OpenCV loads images in BGR order

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns

        # Function to plot histogram and CDF
        def plot_channel_histogram(ax_hist, ax_cdf, channel, color, label):
            """Helper function to plot histogram and CDF for each channel."""
            hist, bins = np.histogram(channel.flatten(), bins=256, range=[0, 256])
            cdf = hist.cumsum()  # Compute CDF
            cdf_normalized = cdf * hist.max() / cdf.max()  # Normalize for visualization

            # Plot histogram
            ax_hist.plot(hist, color=color)
            ax_hist.set_title(f"{label} Histogram")
            ax_hist.set_xlim([0, 256])

            # Plot CDF
            ax_cdf.plot(cdf_normalized, color=color, linestyle="--")
            ax_cdf.set_title(f"{label} CDF")
            ax_cdf.set_xlim([0, 256])

        # Plot histograms and CDFs for R, G, B channels
        plot_channel_histogram(axes[0, 0], axes[1, 0], r, 'red', "Red")    # Red histogram & CDF
        plot_channel_histogram(axes[0, 1], axes[1, 1], g, 'green', "Green")  # Green histogram & CDF
        plot_channel_histogram(axes[0, 2], axes[1, 2], b, 'blue', "Blue")  # Blue histogram & CDF

        plt.tight_layout()
        plt.show()