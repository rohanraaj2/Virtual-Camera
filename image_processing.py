import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from skimage.filters import gabor
from skimage.feature import canny

class ImageProcessor:
    """
    Image processing class implementing all required basic operations
    for the Virtual Camera project
    """
    
    def __init__(self):
        """Initialize the image processor"""
        self.last_histogram = None
        
    def statistical_analysis(self, frame):
        """
        Perform statistical analysis on each RGB channel
        Returns: dict with statistics for each channel
        """
        stats_dict = {}
        
        # Split into RGB channels
        b, g, r = cv2.split(frame)
        channels = {'red': r, 'green': g, 'blue': b}
        
        for channel_name, channel_data in channels.items():
            # Flatten the channel for calculations
            flat_channel = channel_data.flatten()
            
            stats_dict[channel_name] = {
                'mean': np.mean(flat_channel),
                'mode': stats.mode(flat_channel, keepdims=True)[0][0],
                'std': np.std(flat_channel),
                'max': np.max(flat_channel),
                'min': np.min(flat_channel)
            }
            
        return stats_dict
    
    def linear_transformation(self, frame, alpha=1.2, beta=30):
        """
        Apply linear transformation: new_pixel = alpha * old_pixel + beta
        Args:
            frame: input image
            alpha: contrast multiplier
            beta: brightness offset
        """
        # Apply transformation and clip values to valid range
        transformed = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return transformed
    
    def calculate_entropy(self, frame):
        """
        Calculate entropy for each RGB channel
        Returns: dict with entropy values
        """
        entropy_dict = {}
        
        # Split into RGB channels
        b, g, r = cv2.split(frame)
        channels = {'red': r, 'green': g, 'blue': b}
        
        for channel_name, channel_data in channels.items():
            # Calculate histogram
            hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))
            
            # Normalize histogram to get probabilities
            hist = hist / hist.sum()
            
            # Remove zero probabilities to avoid log(0)
            hist = hist[hist > 0]
            
            # Calculate entropy: -sum(p * log2(p))
            entropy = -np.sum(hist * np.log2(hist))
            entropy_dict[channel_name] = entropy
            
        return entropy_dict
    
    def plot_histogram(self, frame, title="RGB Histogram"):
        """
        Create histogram plot for RGB channels
        Returns: histogram plot as image array
        """
        # Split into RGB channels
        b, g, r = cv2.split(frame)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate histograms
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        
        # Plot histograms
        ax.plot(hist_r, color='red', alpha=0.7, label='Red')
        ax.plot(hist_g, color='green', alpha=0.7, label='Green')
        ax.plot(hist_b, color='blue', alpha=0.7, label='Blue')
        
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convert plot to image array
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return plot_img
    
    def histogram_equalization(self, frame):
        """
        Apply histogram equalization to each channel separately
        """
        # Split into RGB channels
        b, g, r = cv2.split(frame)
        
        # Apply histogram equalization to each channel
        r_eq = cv2.equalizeHist(r)
        g_eq = cv2.equalizeHist(g)
        b_eq = cv2.equalizeHist(b)
        
        # Merge channels back
        equalized = cv2.merge([b_eq, g_eq, r_eq])
        return equalized
    
    def edge_detection_filter(self, frame):
        """
        Apply Canny edge detection filter
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Convert back to 3-channel for consistency
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges_colored
    
    def blur_filter(self, frame, kernel_size=15):
        """
        Apply Gaussian blur filter
        """
        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        return blurred
    
    def sharpen_filter(self, frame):
        """
        Apply sharpening filter using convolution
        """
        # Define sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        # Apply filter
        sharpened = cv2.filter2D(frame, -1, kernel)
        return sharpened
    
    def sobel_filter(self, frame):
        """
        Apply Sobel edge detection filter
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Sobel filter in X and Y directions
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine X and Y gradients
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize to 0-255 range
        sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
        
        # Convert back to 3-channel
        sobel_colored = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)
        return sobel_colored
    
    def gabor_filter(self, frame, frequency=0.6):
        """
        Apply Gabor filter
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gabor filter
        filtered_real, _ = gabor(gray, frequency=frequency)
        
        # Normalize to 0-255 range
        filtered_normalized = np.uint8((filtered_real - filtered_real.min()) / 
                                     (filtered_real.max() - filtered_real.min()) * 255)
        
        # Convert back to 3-channel
        gabor_colored = cv2.cvtColor(filtered_normalized, cv2.COLOR_GRAY2BGR)
        return gabor_colored
    
    def process_frame(self, frame, filter_type="edge"):
        """
        Main processing function that applies all required operations
        Args:
            frame: input frame from camera
            filter_type: type of filter to apply ("edge", "blur", "sharpen", "sobel", "gabor")
        """
        # 1. Statistical Analysis (print to console for monitoring)
        stats = self.statistical_analysis(frame)
        
        # 2. Linear Transformation
        transformed = self.linear_transformation(frame, alpha=1.1, beta=20)
        
        # 3. Calculate Entropy
        entropy = self.calculate_entropy(transformed)
        
        # 4. Histogram Equalization
        equalized = self.histogram_equalization(transformed)
        
        # 5. Apply selected filter
        if filter_type == "edge":
            filtered = self.edge_detection_filter(equalized)
        elif filter_type == "blur":
            filtered = self.blur_filter(equalized)
        elif filter_type == "sharpen":
            filtered = self.sharpen_filter(equalized)
        elif filter_type == "sobel":
            filtered = self.sobel_filter(equalized)
        elif filter_type == "gabor":
            filtered = self.gabor_filter(equalized)
        else:
            filtered = self.edge_detection_filter(equalized)  # Default to edge detection
        
        return filtered, stats, entropy
    
    def display_statistics(self, frame):
        """
        Create a visual overlay showing statistics on the frame
        """
        stats = self.statistical_analysis(frame)
        entropy = self.calculate_entropy(frame)
        
        # Create text overlay
        overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White text
        thickness = 1
        
        y_offset = 30
        
        # Display statistics for each channel
        for i, (channel, data) in enumerate(stats.items()):
            text = f"{channel.upper()}: Mean={data['mean']:.1f}, Std={data['std']:.1f}"
            cv2.putText(overlay, text, (10, y_offset + i * 20), font, font_scale, color, thickness)
        
        # Display entropy
        entropy_text = f"Entropy - R:{entropy['red']:.2f} G:{entropy['green']:.2f} B:{entropy['blue']:.2f}"
        cv2.putText(overlay, entropy_text, (10, y_offset + 80), font, font_scale, color, thickness)
        
        return overlay