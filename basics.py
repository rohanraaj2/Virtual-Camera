# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
from numba import njit # conda install numba
import numpy as np
import cv2
from scipy import stats
from scipy.stats import entropy
import math

@njit
def histogram_figure_numba(np_img):
    '''
    Jit compiled function to increase performance.
    Use some loops insteads of purely numpy functions.
    If you face some compile errors using @njit, see: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
    In case you dont need performance boosts, remove the njit flag above the function
    Do not use cv2 functions together with @njit
    '''
    height, width, channels = np_img.shape
    
    # Initialize histogram arrays for RGB channels
    r_bars = np.zeros(256)
    g_bars = np.zeros(256)
    b_bars = np.zeros(256)
    
    # Calculate histograms for each channel
    for i in range(height):
        for j in range(width):
            r_val = np_img[i, j, 0]
            g_val = np_img[i, j, 1] 
            b_val = np_img[i, j, 2]
            
            r_bars[r_val] += 1
            g_bars[g_val] += 1
            b_bars[b_val] += 1
    
    # Normalize the histogram values (optional, for better visualization)
    total_pixels = height * width
    r_bars = r_bars / total_pixels * 3  # Scale for visualization
    g_bars = g_bars / total_pixels * 3
    b_bars = b_bars / total_pixels * 3
    
    return r_bars, g_bars, b_bars


####
### Statistical Analysis Functions
####

def calculate_mean_rgb(np_img):
    """
    Calculate the mean value for each RGB channel
    """
    r_mean = np.mean(np_img[:, :, 0])
    g_mean = np.mean(np_img[:, :, 1])
    b_mean = np.mean(np_img[:, :, 2])
    return r_mean, g_mean, b_mean

def calculate_mode_rgb(np_img):
    """
    Calculate the mode (most frequent value) for each RGB channel
    """
    r_mode = stats.mode(np_img[:, :, 0].flatten(), keepdims=True)[0][0]
    g_mode = stats.mode(np_img[:, :, 1].flatten(), keepdims=True)[0][0]
    b_mode = stats.mode(np_img[:, :, 2].flatten(), keepdims=True)[0][0]
    return r_mode, g_mode, b_mode

def calculate_std_rgb(np_img):
    """
    Calculate the standard deviation for each RGB channel
    """
    r_std = np.std(np_img[:, :, 0])
    g_std = np.std(np_img[:, :, 1])
    b_std = np.std(np_img[:, :, 2])
    return r_std, g_std, b_std

def calculate_max_rgb(np_img):
    """
    Calculate the maximum value for each RGB channel
    """
    r_max = np.max(np_img[:, :, 0])
    g_max = np.max(np_img[:, :, 1])
    b_max = np.max(np_img[:, :, 2])
    return r_max, g_max, b_max

def calculate_min_rgb(np_img):
    """
    Calculate the minimum value for each RGB channel
    """
    r_min = np.min(np_img[:, :, 0])
    g_min = np.min(np_img[:, :, 1])
    b_min = np.min(np_img[:, :, 2])
    return r_min, g_min, b_min

####
### Transformation Functions
####

def apply_linear_transformation(np_img, alpha=1.2, beta=30):
    """
    Apply linear transformation: output = alpha * input + beta
    Alpha controls contrast, beta controls brightness
    """
    # Apply transformation to each channel
    transformed = cv2.convertScaleAbs(np_img, alpha=alpha, beta=beta)
    return transformed

def calculate_entropy_rgb(np_img):
    """
    Calculate entropy for each RGB channel
    Entropy measures the randomness/information content
    """
    r_entropy = entropy(np.histogram(np_img[:, :, 0], bins=256)[0] + 1e-10)
    g_entropy = entropy(np.histogram(np_img[:, :, 1], bins=256)[0] + 1e-10)
    b_entropy = entropy(np.histogram(np_img[:, :, 2], bins=256)[0] + 1e-10)
    return r_entropy, g_entropy, b_entropy

def apply_histogram_equalization(np_img):
    """
    Apply histogram equalization to improve contrast
    """
    # Convert to YUV color space for better results
    yuv = cv2.cvtColor(np_img, cv2.COLOR_RGB2YUV)
    
    # Apply histogram equalization to the Y (luminance) channel
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    
    # Convert back to RGB
    equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return equalized

####
### Filter Functions
####

def apply_edge_detection(np_img):
    """
    Apply Sobel edge detection filter
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    
    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.clip(sobel_combined, 0, 255).astype(np.uint8)
    
    # Convert back to RGB (grayscale edges)
    edges_rgb = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2RGB)
    return edges_rgb

def apply_gaussian_blur(np_img, kernel_size=15):
    """
    Apply Gaussian blur filter
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(np_img, (kernel_size, kernel_size), 0)
    return blurred

def apply_sharpen_filter(np_img):
    """
    Apply sharpening filter using kernel convolution
    """
    # Define sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Apply filter to each channel
    sharpened = cv2.filter2D(np_img, -1, kernel)
    return sharpened

def apply_gabor_filter(np_img, theta=0, frequency=0.6):
    """
    Apply Gabor filter for texture analysis
    """
    # Convert to grayscale
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    
    # Create Gabor kernel
    kernel = cv2.getGaborKernel((31, 31), 4, theta, 2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
    
    # Apply Gabor filter
    gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    
    # Convert back to RGB
    gabor_rgb = cv2.cvtColor(gabor_response, cv2.COLOR_GRAY2RGB)
    return gabor_rgb

####
### Utility Functions
####

def get_image_statistics(np_img):
    """
    Get comprehensive image statistics for display
    Returns a list of strings suitable for overlay display
    """
    r_mean, g_mean, b_mean = calculate_mean_rgb(np_img)
    r_mode, g_mode, b_mode = calculate_mode_rgb(np_img)
    r_std, g_std, b_std = calculate_std_rgb(np_img)
    r_max, g_max, b_max = calculate_max_rgb(np_img)
    r_min, g_min, b_min = calculate_min_rgb(np_img)
    r_entropy, g_entropy, b_entropy = calculate_entropy_rgb(np_img)
    
    stats_text = [
        f"Mean: R={r_mean:.1f} G={g_mean:.1f} B={b_mean:.1f}",
        f"Mode: R={r_mode} G={g_mode} B={b_mode}",
        f"Std: R={r_std:.1f} G={g_std:.1f} B={b_std:.1f}",
        f"Max: R={r_max} G={g_max} B={b_max}",
        f"Min: R={r_min} G={g_min} B={b_min}",
        f"Entropy: R={r_entropy:.2f} G={g_entropy:.2f} B={b_entropy:.2f}"
    ]
    
    return stats_text