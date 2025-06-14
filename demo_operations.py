import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing import ImageProcessor

def demonstrate_operations():
    """
    Demonstrate all implemented image processing operations
    """
    # Load a sample image or capture from camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. Using sample image instead.")
        # Create a sample image
        sample_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        ret, sample_img = cap.read()
        cap.release()
        if not ret:
            sample_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize processor
    processor = ImageProcessor()
    
    print("=== Demonstrating Image Processing Operations ===\n")
    
    # 1. Statistical Analysis
    print("1. Statistical Analysis:")
    stats = processor.statistical_analysis(sample_img)
    for channel, data in stats.items():
        print(f"   {channel.upper()} Channel:")
        print(f"     Mean: {data['mean']:.2f}")
        print(f"     Mode: {data['mode']}")
        print(f"     Std Dev: {data['std']:.2f}")
        print(f"     Max: {data['max']}")
        print(f"     Min: {data['min']}")
    print()
    
    # 2. Linear Transformation
    print("2. Linear Transformation (α=1.2, β=30):")
    transformed = processor.linear_transformation(sample_img, alpha=1.2, beta=30)
    print("   Applied successfully - increases contrast and brightness")
    print()
    
    # 3. Entropy Calculation
    print("3. Entropy Calculation:")
    entropy = processor.calculate_entropy(sample_img)
    for channel, value in entropy.items():
        print(f"   {channel.upper()} Channel Entropy: {value:.3f}")
    print()
    
    # 4. Histogram Equalization
    print("4. Histogram Equalization:")
    equalized = processor.histogram_equalization(sample_img)
    print("   Applied successfully - enhances contrast")
    print()
    
    # 5. Filters
    print("5. Filters:")
    filters = [
        ("Edge Detection (Canny)", processor.edge_detection_filter),
        ("Gaussian Blur", processor.blur_filter),
        ("Sharpen", processor.sharpen_filter),
        ("Sobel Edge Detection", processor.sobel_filter),
        ("Gabor Filter", processor.gabor_filter)
    ]
    
    for filter_name, filter_func in filters:
        try:
            filtered = filter_func(sample_img)
            print(f"   ✓ {filter_name}: Applied successfully")
        except Exception as e:
            print(f"   ✗ {filter_name}: Error - {e}")
    print()
    
    # 6. Create visual comparison
    print("6. Creating visual comparison...")
    
    # Apply all operations in sequence
    final_processed, final_stats, final_entropy = processor.process_frame(sample_img, "edge")
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Image Processing Operations Demonstration', fontsize=16)
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Linear Transformation
    axes[0, 1].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Linear Transformation')
    axes[0, 1].axis('off')
    
    # Histogram Equalization
    axes[0, 2].imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Histogram Equalized')
    axes[0, 2].axis('off')
    
    # Edge Detection
    edge_result = processor.edge_detection_filter(sample_img)
    axes[1, 0].imshow(cv2.cvtColor(edge_result, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Edge Detection')
    axes[1, 0].axis('off')
    
    # Blur
    blur_result = processor.blur_filter(sample_img)
    axes[1, 1].imshow(cv2.cvtColor(blur_result, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Gaussian Blur')
    axes[1, 1].axis('off')
    
    # Final processed
    axes[1, 2].imshow(cv2.cvtColor(final_processed, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Full Processing Pipeline')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('processing_demonstration.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ All operations demonstrated successfully!")
    print("✓ Visual comparison saved as 'processing_demonstration.png'")

if __name__ == "__main__":
    demonstrate_operations()