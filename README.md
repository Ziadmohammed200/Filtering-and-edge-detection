# Filtering and Edge Detection Project

## Overview
This project explores fundamental image processing techniques, focusing on noise addition, filtering, and edge detection. The goal is to enhance image quality, detect significant features, and analyze pixel distributions using various computer vision methodologies.

## Implemented Tasks

### 1️⃣ Noise Addition
- **Uniform Noise**
![uniform_noise](https://github.com/user-attachments/assets/85d96360-0e87-4321-abff-5bba80d3beba)

- **Gaussian Noise**
![Gaussian_noise](https://github.com/user-attachments/assets/885b92cf-4b17-4a02-b9df-64f66ce60bef)

- **Salt & Pepper Noise**
![salt pepper_nose](https://github.com/user-attachments/assets/8323c180-5a13-421d-afef-71b3c88d1256)


### 2️⃣ Image Filtering (Noise Reduction)
- **Average Filtering** - Reduces noise by replacing each pixel value with the average of its neighboring pixels, leading to a smoothing effect.
![Average_filter](https://github.com/user-attachments/assets/674638f1-2ade-4074-abda-479bc1ae234f)

- **Gaussian Filtering** - Blurs images and removes noise while preserving edges better than average filtering.
![Gaussian_filter](https://github.com/user-attachments/assets/21501f6a-0233-48e2-8d3d-f128476a60cd)

- **Median Filtering** - Replaces each pixel value with the median value of the neighboring pixels, effectively reducing salt-and-pepper noise.
![median_filter](https://github.com/user-attachments/assets/7aa2ed7f-21a6-44ed-b8e4-2bde66e30529)


### 3️⃣ Edge Detection

#### 🔹 Sobel Edge Detection
Sobel edge detection emphasizes edges in an image by computing intensity gradients.

**Steps Involved:**
1. Apply Gaussian filter to smooth the image.
2. Convolve the image with vertical and horizontal derivative kernels.
3. Compute intensity gradients by combining vertical and horizontal edges.
![sobel](https://github.com/user-attachments/assets/34126abc-51b0-44a5-aa5a-cbd16fc0d158)


#### 🔹 Roberts Operator
The Roberts Operator highlights regions of high spatial gradient, corresponding to edges.

**Steps Involved:**
1. Compute the difference between diagonal neighboring pixels.
2. Use two 2x2 kernels to approximate the gradient magnitude.
3. Combine results to emphasize edges.
![robert](https://github.com/user-attachments/assets/35b4cb2a-6ad6-4899-838b-b7d1a0b537b6)


#### 🔹 Prewitt Operator
The Prewitt Operator is similar to Sobel edge detection but uses fixed convolution kernels for detecting horizontal and vertical edges.

**Steps Involved:**
1. Apply Gaussian filter for noise reduction.
2. Convolve the image with Prewitt kernels.
3. Combine results to obtain the final edge-detected image.
![prewitt](https://github.com/user-attachments/assets/ca10a16e-10d5-4482-bf11-fad67db073df)


#### 🔹 Canny Edge Detector
The Canny Edge Detector is a multi-stage algorithm designed for accurate and noise-resistant edge detection.

**Steps Involved:**
1. Apply Gaussian smoothing to reduce noise.
2. Compute intensity gradients using Sobel filters.
3. Apply non-maximum suppression to thin edges.
4. Use double thresholding to classify strong and weak edges.
5. Apply edge tracking by hysteresis to finalize detected edges.
![canny](https://github.com/user-attachments/assets/d71443dd-454f-4e36-8303-dd6bdf265106)


### 4️⃣ Histogram Analysis & Equalization
- Compute and visualize histograms.
- Generate cumulative distribution functions (CDFs).
- Apply histogram equalization for contrast enhancement.

### 5️⃣ Image Normalization
- Scale pixel values to a specific range for improved processing.

### 6️⃣ Thresholding Techniques
- **Global Thresholding** - Fixed value thresholding.
![global](https://github.com/user-attachments/assets/a00a8e56-8abf-40e3-bf69-7878805c13f9)

- **Local Thresholding** - Adaptive thresholding methods.
![local](https://github.com/user-attachments/assets/b0fbdccc-8d75-4454-afd4-a5c18d6ea47d)


### 7️⃣ Grayscale Conversion & RGB Histogram Analysis
- Convert color images to grayscale.
- Extract and plot histograms for R, G, and B channels with their cumulative distribution functions.
![gray](https://github.com/user-attachments/assets/90a1c4a9-b16c-48a8-928c-7e70772f2488)


### 8️⃣ Frequency Domain Filtering
- **High-pass filtering** - Enhances edges and fine details.
![high_pass](https://github.com/user-attachments/assets/e59ae0cd-e794-4b8d-8f07-473296f9f9d8)

- **Low-pass filtering** - Reduces noise and smoothens images.
![low_pass](https://github.com/user-attachments/assets/1184b2fe-5ceb-4909-bb9d-7b863271a3e6)



### 9️⃣ Hybrid Image Generation
- Combines low-frequency and high-frequency components from two images to create a hybrid effect.
![hybrid](https://github.com/user-attachments/assets/bc2eeff1-f60d-4f45-b104-42985c3df896)


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

