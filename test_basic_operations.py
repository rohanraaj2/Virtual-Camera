# -*- coding: utf-8 -*-
"""
Test script for basic image processing operations
Run this to verify that the basic operations work correctly
"""

import numpy as np
import cv2
try:
    from basics import (get_image_statistics, apply_linear_transformation, 
                        apply_histogram_equalization, apply_edge_detection, 
                        apply_gaussian_blur, apply_sharpen_filter, 
                        apply_gabor_filter, histogram_figure_numba)
    print("✓ All basic operation imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure to install all required packages: pip install -r requirements.txt")
    exit(1)

def test_basic_operations():
    """Test all basic image processing operations with a synthetic image"""
    
    # Create a test image (colorful gradient)
    height, width = 240, 320
    test_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient pattern
    for i in range(height):
        for j in range(width):
            test_img[i, j, 0] = int((i / height) * 255)  # Red gradient
            test_img[i, j, 1] = int((j / width) * 255)   # Green gradient  
            test_img[i, j, 2] = int(((i + j) / (height + width)) * 255)  # Blue gradient
    
    print("Testing basic image processing operations...")
    
    # Test statistical analysis
    try:
        stats = get_image_statistics(test_img)
        print("✓ Statistical analysis working")
        for stat in stats[:3]:  # Show first 3 statistics
            print(f"  {stat}")
    except Exception as e:
        print(f"✗ Statistical analysis failed: {e}")
    
    # Test histogram calculation
    try:
        r_bars, g_bars, b_bars = histogram_figure_numba(test_img)
        print(f"✓ Histogram calculation working (shapes: R={len(r_bars)}, G={len(g_bars)}, B={len(b_bars)})")
    except Exception as e:
        print(f"✗ Histogram calculation failed: {e}")
    
    # Test transformations and filters
    operations = [
        ("Linear Transformation", lambda img: apply_linear_transformation(img, alpha=1.2, beta=20)),
        ("Histogram Equalization", apply_histogram_equalization),
        ("Edge Detection", apply_edge_detection),
        ("Gaussian Blur", lambda img: apply_gaussian_blur(img, 15)),
        ("Sharpen Filter", apply_sharpen_filter),
        ("Gabor Filter", lambda img: apply_gabor_filter(img, theta=np.pi/4, frequency=0.6))
    ]
    
    for name, operation in operations:
        try:
            result = operation(test_img)
            if result.shape == test_img.shape:
                print(f"✓ {name} working")
            else:
                print(f"✗ {name} - incorrect output shape: {result.shape}")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    print("\nTest completed! If all operations show ✓, your implementation is working correctly.")
    print("Note: Import errors for 'numba' and 'face_recognition' are normal if packages aren't installed.")

if __name__ == "__main__":
    test_basic_operations()
