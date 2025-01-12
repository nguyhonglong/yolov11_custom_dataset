import cv2
import numpy as np

def enhance_lowlight(image_path, alpha=1.5, beta=30):
    """
    Enhance a low-light image using contrast and brightness adjustment
    
    Args:
        image_path (str): Path to the input image
        alpha (float): Contrast control (1.0-3.0)
        beta (int): Brightness control (0-100)
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    # Merge channels
    lab = cv2.merge((l,a,b))

    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply contrast and brightness adjustment
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    return enhanced

def save_enhanced_image(input_path, output_path, alpha=1.5, beta=30):
    """
    Enhance a low-light image and save the result
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save enhanced image
        alpha (float): Contrast control
        beta (int): Brightness control
    """
    enhanced = enhance_lowlight(input_path, alpha=alpha, beta=beta)
    cv2.imwrite(output_path, enhanced)
