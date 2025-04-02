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
    
    # Resize image to ensure consistent processing
    image = cv2.resize(image, (300, 400))
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Further clean up with closing operation
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return binary

def find_stem_and_quadrants(binary_image):
    """
    Find the main vertical stem of the Cistercian numeral and divide into quadrants.
    
    Args:
        binary_image: Preprocessed binary image
        
    Returns:
        Dictionary with stem coordinates and quadrant boundaries
    """
    # If no contours are found, create default quadrants based on image size
    height, width = binary_image.shape
    
    # Try to find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning("No contours found in the image, using default quadrants")
        # Use the entire image and divide it into quadrants
        x, y = 0, 0
        w, h = width, height
    else:
        # Combine all contours to get the bounding rectangle of the entire symbol
        all_points = np.concatenate([cnt for cnt in contours])
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Ensure we have a reasonable size
        if w < 20 or h < 20:
            logger.warning("Contour too small, using default quadrants")
            x, y = 0, 0
            w, h = width, height
    
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
    
    # Debug: Draw quadrant boundaries on a copy of the image for visualization
    # visual_debug = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
    # cv2.line(visual_debug, (stem_x, stem_top), (stem_x, stem_bottom), (0, 255, 0), 2)
    # cv2.line(visual_debug, (x, center_y), (x + w, center_y), (0, 255, 0), 2)
    # cv2.imwrite('/tmp/debug_quadrants.png', visual_debug)
    
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
    # Ensure coordinates are integers
    x1, y1, x2, y2 = map(int, quadrant_coords)
    
    # Ensure coordinates are within image bounds
    height, width = binary_image.shape
    x1 = max(0, min(x1, width-1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height-1))
    y2 = max(0, min(y2, height))
    
    # Extract the quadrant image
    if x2 <= x1 or y2 <= y1:
        return 0  # Invalid quadrant
    
    quadrant_img = binary_image[y1:y2, x1:x2]
    
    # Skip empty or nearly empty quadrants
    if np.sum(quadrant_img) < 100:  # Threshold for considering a quadrant empty
        return 0
    
    # Extract features
    # Find contours in the quadrant
    contours, _ = cv2.findContours(quadrant_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    # Count number of distinct shapes
    num_contours = len(contours)
    
    # Analyze contour shapes
    horizontal_lines = 0
    vertical_lines = 0
    diagonal_lines = 0
    rectangles = 0
    
    # Get quadrant dimensions for relative measurements
    q_height, q_width = quadrant_img.shape
    quadrant_area = q_height * q_width
    filled_area = np.sum(quadrant_img > 0)
    fill_ratio = filled_area / quadrant_area if quadrant_area > 0 else 0
    
    for contour in contours:
        # Basic shape properties
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(contour)
        
        # Identify shape type
        if aspect_ratio > 2.5:  # Width significantly greater than height
            horizontal_lines += 1
        elif aspect_ratio < 0.4:  # Height significantly greater than width
            vertical_lines += 1
        elif 0.8 < aspect_ratio < 1.2 and area > 0.1 * quadrant_area:
            rectangles += 1  # Approximately square shape
            
        # Check for diagonal lines
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 2:  # Line-like shape
            pt1, pt2 = approx[0][0], approx[1][0]
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            
            if dx == 0:
                continue  # Vertical line, already counted
                
            angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
            if 30 < angle < 60 or 120 < angle < 150:  # Diagonal angles
                diagonal_lines += 1
    
    # Improved digit recognition based on feature combination
    # Digit 1: horizontal line
    if horizontal_lines == 1 and vertical_lines == 0 and diagonal_lines == 0:
        return 1
    
    # Digit 2: diagonal line (top-left to bottom-right or top-right to bottom-left)
    elif horizontal_lines == 0 and vertical_lines == 0 and diagonal_lines == 1:
        # Check for direction based on quadrant location
        if 'top-right' in str(quadrant_coords) or 'bottom-left' in str(quadrant_coords):
            return 2
        else:
            return 6
    
    # Digit 3: Horizontal line with diagonal (like an arrowhead)
    elif horizontal_lines >= 1 and diagonal_lines >= 1 and vertical_lines == 0:
        # Direction depends on quadrant
        if 'top-right' in str(quadrant_coords) or 'bottom-left' in str(quadrant_coords):
            return 3
        else:
            return 7
    
    # Digit 4: Horizontal line with vertical extension (L shape)
    elif (horizontal_lines >= 1 and vertical_lines >= 1) or num_contours == 2:
        return 4
    
    # Digit 5: Rectangle or high fill ratio indicating a complex shape
    elif rectangles >= 1 or fill_ratio > 0.4:
        return 5
    
    # Digit 6: Diagonal line (flipped compared to 2)
    elif horizontal_lines == 0 and vertical_lines == 0 and diagonal_lines == 1:
        # This check is redundant with digit 2, but we'll keep it for clarity
        return 6
    
    # Digit 7: Like digit 3 but flipped
    elif horizontal_lines >= 1 and diagonal_lines >= 1:
        return 7
    
    # Digit 8: V shape or two diagonal lines
    elif diagonal_lines >= 2:
        return 8
    
    # Digit 9: Circle or square shape
    elif rectangles >= 1 or (num_contours == 1 and cv2.contourArea(contours[0]) > 0.3 * quadrant_area):
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
