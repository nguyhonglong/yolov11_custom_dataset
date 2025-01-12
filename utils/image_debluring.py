import cv2
import numpy as np

def deblur_image(image_path, kernel_size=3, sigma=1.5):
    """
    Deblur an image using Wiener deconvolution
    
    Args:
        image_path (str): Path to the input image
        kernel_size (int): Size of the Gaussian kernel (must be odd)
        sigma (float): Standard deviation for Gaussian kernel
        
    Returns:
        numpy.ndarray: Deblurred image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
        
    # Convert to float for processing
    img_float = img.astype(np.float32) / 255.0
    
    # Create a Gaussian kernel to simulate blur
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel)
    
    # Perform Wiener deconvolution for each channel
    deblurred = np.zeros_like(img_float)
    for i in range(3):  # Process each color channel
        # FFT of image and kernel
        img_fft = np.fft.fft2(img_float[:,:,i])
        kernel_fft = np.fft.fft2(kernel, s=img_float[:,:,i].shape)
        
        # Wiener deconvolution
        K = 0.01  # Noise to signal ratio
        deblurred[:,:,i] = np.abs(np.fft.ifft2(
            np.conj(kernel_fft) * img_fft / (np.abs(kernel_fft)**2 + K)
        ))
    
    # Clip values to valid range and convert back to uint8
    deblurred = np.clip(deblurred * 255, 0, 255).astype(np.uint8)
    
    return deblurred

def save_deblurred_image(input_path, output_path, kernel_size=3, sigma=1.5):
    """
    Deblur an image and save the result
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save deblurred image
        kernel_size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation for Gaussian kernel
    """
    deblurred = deblur_image(input_path, kernel_size, sigma)
    cv2.imwrite(output_path, deblurred)
