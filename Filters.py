import numpy as np
import cv2

class filter:
    @staticmethod
    def average_filter(image, size=3):
        if image is None:
            raise ValueError("Error: Image not found or cannot be loaded.")
        if size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        
        kernel = np.ones((size, size), dtype=float) / (size * size)
        pad = size // 2
        row, col = image.shape
        filtered_image = np.zeros((row, col), dtype=np.uint8)

        for i in range(pad, row - pad):
            for j in range(pad, col - pad):
                neighborhood = image[i - pad : i + pad + 1, j - pad : j + pad + 1]
                filtered_image[i, j] = int(np.sum(neighborhood * kernel))

        return filtered_image

    @staticmethod
    def gaussian_filter(image, std=1, size=3):
        if image is None:
            raise ValueError("Error: Image not found or cannot be loaded.")
        if size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")

        kernel = np.zeros((size, size), dtype=float)
        center = size // 2

        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = (1 / (2 * np.pi * std**2)) * np.exp(-(x**2 + y**2) / (2 * std**2))

        kernel /= np.sum(kernel)  # Normalize kernel

        pad = size // 2
        row, col = image.shape
        filtered_image = np.zeros((row, col), dtype=np.uint8)

        for i in range(pad, row - pad):
            for j in range(pad, col - pad):
                neighborhood = image[i - pad : i + pad + 1, j - pad : j + pad + 1]
                filtered_image[i, j] = int(np.sum(neighborhood * kernel))

        return filtered_image

    @staticmethod
    def median_filter(image, size=3):
        if image is None:
            raise ValueError("Error: Image not found or cannot be loaded.")
        if size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        
        pad = size // 2
        row, col = image.shape
        filtered_image = np.zeros((row, col), dtype=np.uint8)

        for i in range(pad, row - pad):
            for j in range(pad, col - pad):
                neighborhood = image[i - pad : i + pad + 1, j - pad : j + pad + 1]
                filtered_image[i, j] = np.median(neighborhood)

        return filtered_image

    @staticmethod
    def frequency_filter(image, filter_type, d=30):
        if image is None:
            raise ValueError("Error: Image not found or cannot be loaded.")
        
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.zeros((rows, cols), np.uint8)
        if filter_type == 'low':
            mask[crow-d:crow+d, ccol-d:ccol+d] = 1
        elif filter_type == 'high':
            mask[:] = 1
            mask[crow-d:crow+d, ccol-d:ccol+d] = 0

        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_filtered = np.abs(np.fft.ifft2(f_ishift))

        return img_filtered.astype(np.uint8)

##########################################################################


    @staticmethod
    def resize_images(image1, image2):
        """ Resize image2 to match image1 dimensions. """
        return cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    @staticmethod
    def fft_transform(image):
        """ Compute the FFT and shift the zero frequency component to the center. """
        dft = np.fft.fft2(image)
        dft_shift = np.fft.fftshift(dft)
        return dft_shift

    @staticmethod
    def ifft_transform(dft_shift):
        """ Compute the inverse FFT to get the image back in spatial domain. """
        dft_ishift = np.fft.ifftshift(dft_shift)
        image_back = np.fft.ifft2(dft_ishift)
        return np.abs(image_back)

    @staticmethod
    def create_low_pass_filter(shape, cutoff):
        """ Create a low-pass filter with a given cutoff frequency. """
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (center_col, center_row), cutoff, 1, thickness=-1)
        return mask

    @staticmethod
    def create_high_pass_filter(shape, cutoff):
        """ Create a high-pass filter by inverting a low-pass filter. """
        low_pass = filter.create_low_pass_filter(shape, cutoff)
        high_pass = 1 - low_pass
        return high_pass

    @staticmethod
    def apply_frequency_filter(image, filter_mask):
        """ Apply a frequency domain filter (low-pass or high-pass). """
        dft_shift = filter.fft_transform(image)
        filtered_dft = dft_shift * filter_mask
        return filter.ifft_transform(filtered_dft)

    @staticmethod
    def create_hybrid_image(image1, image2, low_cutoff=30, high_cutoff=40, switch=False):
        """
        Create a hybrid image using frequency domain filtering.
        
        Parameters:
        - image1: First input image (used for low frequencies).
        - image2: Second input image (used for high frequencies).
        - low_cutoff: Cutoff frequency for low-pass filter.
        - high_cutoff: Cutoff frequency for high-pass filter.
        - switch: If True, swap roles of image1 and image2.
        """

        # Convert images to grayscale if they are not already
        if len(image1.shape) == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if len(image2.shape) == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Resize images if needed
        if image1.shape != image2.shape:
            image2 = filter.resize_images(image1, image2)

        if switch:
            image1, image2 = image2, image1

        # Create frequency domain filters
        low_pass_filter = filter.create_low_pass_filter(image1.shape, low_cutoff)
        high_pass_filter = filter.create_high_pass_filter(image2.shape, high_cutoff)

        # Apply filters in frequency domain
        low_pass_result = filter.apply_frequency_filter(image1, low_pass_filter)
        high_pass_result = filter.apply_frequency_filter(image2, high_pass_filter)

        # Normalize both images to range [0,1]
        low_pass_result = (low_pass_result - low_pass_result.min()) / (low_pass_result.max() - low_pass_result.min())
        high_pass_result = (high_pass_result - high_pass_result.min()) / (high_pass_result.max() - high_pass_result.min())

        # Scale the contributions manually (alpha = 0.6, beta = 0.4 for better effect)
        alpha, beta = 0.6, 0.4
        hybrid_image = alpha * low_pass_result + beta * high_pass_result

        # Normalize the final result to 0-255
        hybrid_image = cv2.normalize(hybrid_image, None, 0, 255, cv2.NORM_MINMAX)

        return np.uint8(hybrid_image)

