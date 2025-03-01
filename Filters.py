import numpy as np
import cv2


class filter:
    def average_filter(image,size=3):
        if image is None:
            raise ValueError("Error: Image not found or cannot be loaded.")
        if size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        
        kernel = np.ones((size, size), dtype=float) / (size * size)
        pad=size//2

        
        row , col = image.shape
        filtered_image = np.zeros((row, col), dtype=np.uint8)  # Create a new image

        for i in range(pad, row - pad):  
            for j in range(pad, col - pad):
                # Extract the neighborhood region based on the kernel size
                neighborhood = image[i - pad : i + pad + 1, j - pad : j + pad + 1]
                
                # Compute the average using the kernel
                res = np.sum(neighborhood * kernel)
                

                # Assign the new value
                filtered_image[i, j] = int(res)

        return filtered_image

    #####################################################################################################

    def Gaussian_Filter(image , std=0,size=3 ):
        if std == 0 :
            std=1
        if image is None:
            raise ValueError("Error: Image not found or cannot be loaded.")
        if size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        kernel = np.ones((size, size), dtype=float)
        center = size // 2

        # Compute Gaussian values
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center  # Shift indices relative to center
                kernel[i, j] = (1 / (2 * np.pi * (std**2))) * np.exp(-(x**2 + y**2) / (2 * (std**2)))

        # Normalize the kernel
        kernel /= np.sum(kernel)

        row , col = image.shape
        filtered_image = np.zeros((row, col), dtype=np.uint8)  # Create a new image

        for i in range(center, row - center):  
            for j in range(center, col - center):
                # Extract the neighborhood region based on the kernel size
                neighborhood = image[i - center : i + center + 1, j - center : j + center + 1]
                
                # Compute the average using the kernel
                res = np.sum(neighborhood * kernel)
                

                # Assign the new value
                filtered_image[i, j] = int(res)

        return filtered_image


    ##################################################################################################



    def median_filter(image , size=3):
        if image is None:
            raise ValueError("Error: Image not found or cannot be loaded.")
        if size % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        
        pad = size//2

        row , col = image.shape
        filtered_image = np.zeros((row, col), dtype=np.uint8)  # Create a new image


        for i in range(pad, row - pad):  
            for j in range(pad, col - pad):
                # Extract the neighborhood region based on the kernel size
                neighborhood = image[i - pad : i + pad + 1, j - pad : j + pad + 1]
                median_value = np.median(neighborhood)
                

                # Assign the new value
                filtered_image[i, j] = median_value 

        return filtered_image

    

    