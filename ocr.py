import cv2
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

def preprocess_image(image):
    """
    Preprocess the image for better feature extraction.
    
    Args:
        image: Grayscale input image
        
    Returns:
        Preprocessed binary image
    """
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary

def find_stem_and_quadrants(binary_image):
    """
    Find the main vertical stem of the Cistercian numeral and divide into quadrants.
    
    Args:
        binary_image: Preprocessed binary image
        
    Returns:
        Dictionary with stem coordinates and quadrant boundaries
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning("No contours found in the image")
        return None
    
    # Find the main contour (the Cistercian symbol)
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Find the approximate center of the symbol
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Estimate the stem position (vertical line through center)
    # In a proper Cistercian numeral, the stem divides the symbol vertically
    stem_x = center_x
    stem_top = y
    stem_bottom = y + h
    
    # Define the four quadrants
    quadrants = {
        'top-left': (x, y, stem_x, center_y),
        'top-right': (stem_x, y, x + w, center_y),
        'bottom-left': (x, center_y, stem_x, y + h),
        'bottom-right': (stem_x, center_y, x + w, y + h)
    }
    
    return {
        'stem': (stem_x, stem_top, stem_x, stem_bottom),
        'quadrants': quadrants
    }

def detect_features_in_quadrant(binary_image, quadrant_coords):
    """
    Detect features in a specific quadrant to determine the digit.
    
    Args:
        binary_image: Preprocessed binary image
        quadrant_coords: (x1, y1, x2, y2) coordinates of the quadrant
        
    Returns:
        Estimated digit for the quadrant
    """
    x1, y1, x2, y2 = quadrant_coords
    quadrant_img = binary_image[y1:y2, x1:x2]
    
    # Skip empty quadrants
    if np.sum(quadrant_img) < 100:  # Threshold for considering a quadrant empty
        return 0
    
    # Extract features
    # For a simple implementation, we'll use basic shape properties
    
    # Find contours in the quadrant
    contours, _ = cv2.findContours(quadrant_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    # Count number of lines (approximated by contours)
    num_contours = len(contours)
    
    # Check for horizontal lines
    horizontal_lines = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 2*h:  # Width significantly greater than height
            horizontal_lines += 1
    
    # Check for diagonal lines
    diagonal_lines = 0
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 2:  # Line-like shape
            pt1, pt2 = approx[0][0], approx[1][0]
            angle = abs(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180 / np.pi)
            if 30 < angle < 60 or 120 < angle < 150:  # Diagonal angles
                diagonal_lines += 1
    
    # Simple heuristic recognition (simplified for demo purposes)
    if horizontal_lines == 1 and diagonal_lines == 0:
        return 1
    elif horizontal_lines == 0 and diagonal_lines == 1:
        if quadrant_img[0, 0] > 0:  # Top-left has pixels
            return 2
        else:
            return 6
    elif horizontal_lines == 1 and diagonal_lines == 1:
        return 3 if quadrant_img[0, 0] > 0 else 7
    elif horizontal_lines == 2 and diagonal_lines == 0:
        return 4
    elif np.sum(quadrant_img) > 0.5 * quadrant_img.size:  # Very filled quadrant
        return 5
    elif num_contours == 2 and diagonal_lines == 2:
        return 8
    elif num_contours == 1 and cv2.contourArea(contours[0]) > 0.4 * quadrant_img.size:
        return 9
    
    # Default fallback
    return 0

def recognize_cistercian_numeral(image):
    """
    Recognize a Cistercian numeral in the image and return the corresponding number.
    
    Args:
        image: Input image containing a Cistercian numeral
        
    Returns:
        Recognized number (0-9999)
    """
    try:
        # Preprocess the image
        binary_image = preprocess_image(image)
        
        # Find the stem and quadrants
        structure = find_stem_and_quadrants(binary_image)
        if not structure:
            logger.warning("Could not identify structure in the image")
            return 0
        
        # Extract digits from each quadrant
        quadrants = structure['quadrants']
        digits = {
            'units': detect_features_in_quadrant(binary_image, quadrants['bottom-right']),
            'tens': detect_features_in_quadrant(binary_image, quadrants['top-right']),
            'hundreds': detect_features_in_quadrant(binary_image, quadrants['bottom-left']),
            'thousands': detect_features_in_quadrant(binary_image, quadrants['top-left'])
        }
        
        # Combine digits to form the number
        number = (
            digits['thousands'] * 1000 +
            digits['hundreds'] * 100 +
            digits['tens'] * 10 +
            digits['units']
        )
        
        return number
    
    except Exception as e:
        logger.error(f"Error in Cistercian numeral recognition: {str(e)}")
        # Return default value in case of error
        return 0
