# Filtering and Edge Detection Project

## Overview
This project explores fundamental image processing techniques, focusing on noise addition, filtering, and edge detection. The goal is to enhance image quality, detect significant features, and analyze pixel distributions using various computer vision methodologies.

## Implemented Tasks

### 1️⃣ Noise Addition
- **Uniform Noise**
- **Gaussian Noise**
- **Salt & Pepper Noise**

### 2️⃣ Image Filtering (Noise Reduction)
- **Average Filtering** - Reduces noise by replacing each pixel value with the average of its neighboring pixels, leading to a smoothing effect.
- **Gaussian Filtering** - Blurs images and removes noise while preserving edges better than average filtering.
- **Median Filtering** - Replaces each pixel value with the median value of the neighboring pixels, effectively reducing salt-and-pepper noise.

### 3️⃣ Edge Detection

#### 🔹 Sobel Edge Detection
Sobel edge detection emphasizes edges in an image by computing intensity gradients.

**Steps Involved:**
1. Apply Gaussian filter to smooth the image.
2. Convolve the image with vertical and horizontal derivative kernels.
3. Compute intensity gradients by combining vertical and horizontal edges.

#### 🔹 Roberts Operator
The Roberts Operator highlights regions of high spatial gradient, corresponding to edges.

**Steps Involved:**
1. Compute the difference between diagonal neighboring pixels.
2. Use two 2x2 kernels to approximate the gradient magnitude.
3. Combine results to emphasize edges.

#### 🔹 Prewitt Operator
The Prewitt Operator is similar to Sobel edge detection but uses fixed convolution kernels for detecting horizontal and vertical edges.

**Steps Involved:**
1. Apply Gaussian filter for noise reduction.
2. Convolve the image with Prewitt kernels.
3. Combine results to obtain the final edge-detected image.

#### 🔹 Canny Edge Detector
The Canny Edge Detector is a multi-stage algorithm designed for accurate and noise-resistant edge detection.

**Steps Involved:**
1. Apply Gaussian smoothing to reduce noise.
2. Compute intensity gradients using Sobel filters.
3. Apply non-maximum suppression to thin edges.
4. Use double thresholding to classify strong and weak edges.
5. Apply edge tracking by hysteresis to finalize detected edges.

### 4️⃣ Histogram Analysis & Equalization
- Compute and visualize histograms.
- Generate cumulative distribution functions (CDFs).
- Apply histogram equalization for contrast enhancement.

### 5️⃣ Image Normalization
- Scale pixel values to a specific range for improved processing.

### 6️⃣ Thresholding Techniques
- **Global Thresholding** - Fixed value thresholding.
- **Local Thresholding** - Adaptive thresholding methods.

### 7️⃣ Grayscale Conversion & RGB Histogram Analysis
- Convert color images to grayscale.
- Extract and plot histograms for R, G, and B channels with their cumulative distribution functions.

### 8️⃣ Frequency Domain Filtering
- **High-pass filtering** - Enhances edges and fine details.
- **Low-pass filtering** - Reduces noise and smoothens images.

### 9️⃣ Hybrid Image Generation
- Combines low-frequency and high-frequency components from two images to create a hybrid effect.

## 📊 Results & Analysis
- All processed images are stored in the `Results/` folder.
- `Report.pdf` provides a comprehensive explanation of methods, comparisons, and observations.
- Visualizations demonstrate the effects of filtering, edge detection, histogram transformations, and hybrid image synthesis.

---

## 🏆 Contributors
**Team 10**  
Prepared for: **Computer Vision**

📌 *Feel free to contribute and enhance this project!*

---

### 🔗 Connect
For any queries or contributions, reach out via GitHub Issues or Pull Requests.

---

⭐ *If you found this project helpful, give it a star!*

