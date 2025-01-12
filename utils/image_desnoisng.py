import cv2
import numpy as np

def denoise_image(image_path, h=10, template_window_size=7, search_window_size=21):
    """
    Denoise an image using Non-Local Means Denoising algorithm
    
    Args:
        image_path (str): Path to the input image
        h (float): Filter strength (higher h = more denoising but more loss of detail)
        template_window_size (int): Size of template patch used for weight calculation
        search_window_size (int): Size of window used to compute weighted average
        
    Returns:
        numpy.ndarray: Denoised image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    # Apply Non-Local Means Denoising for each channel
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=h,
        hColor=h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )
    
    return denoised

def save_denoised_image(input_path, output_path, h=10, template_window_size=7, search_window_size=21):
    """
    Denoise an image and save the result
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save denoised image
        h (float): Filter strength
        template_window_size (int): Size of template patch
        search_window_size (int): Size of search window
    """
    denoised = denoise_image(
        input_path, 
        h=h,
        template_window_size=template_window_size,
        search_window_size=search_window_size
    )
    cv2.imwrite(output_path, denoised)
