import cv2
import numpy as np
import logging

# Set up logging with DEBUG level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def preprocess_image(image):
    """
    Preprocess the image for better feature extraction.
    
    Args:
        image: Input image (color or grayscale)
        
    Returns:
        Preprocessed binary image
    """
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image to ensure consistent processing
    # The dimensions should match what we use for drawing (300x400)
    image = cv2.resize(image, (300, 400))
    
    # Log the image size after resizing
    logger.debug(f"Resized image shape: {image.shape}")
    
    # Normalize the image to improve contrast
    # For images from canvas, they might be faint, so normalize helps
    normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    
    # Use Otsu's thresholding to automatically determine the threshold value
    # This works better for hand-drawn images with varying intensity
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Also try adaptive thresholding which works better for scanned/printed symbols
    binary_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Combine the two thresholding methods - take the more prominent result
    binary = cv2.bitwise_or(binary_otsu, binary_adaptive)
    
    # Noise removal and cleanup
    kernel = np.ones((3, 3), np.uint8)
    
    # MORPH_OPEN: removes small noise outside the main foreground objects
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # MORPH_CLOSE: fills small holes inside foreground objects
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Dilate slightly to ensure lines are connected
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # Log the number of non-zero pixels to check if the image has content
    non_zero = np.count_nonzero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]
    logger.debug(f"Binary image has {non_zero} non-zero pixels out of {total_pixels} ({non_zero/total_pixels:.2%})")
    
    return binary

def find_stem_and_quadrants(binary_image):
    """
    Find the main vertical stem of the Cistercian numeral and divide into quadrants.
    
    Args:
        binary_image: Preprocessed binary image
        
    Returns:
        Dictionary with stem coordinates and quadrant boundaries
    """
    # Get image dimensions
    height, width = binary_image.shape
    logger.debug(f"Image dimensions: {width}x{height}")
    
    # Try to find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours or len(contours) == 0:
        logger.warning("No contours found in the image, using default quadrants")
        # Use the entire image and divide it into quadrants
        x, y = 0, 0
        w, h = width, height
    else:
        # Count the number of contours found
        logger.debug(f"Found {len(contours)} contours")
        
        # Combine all contours to get the bounding rectangle of the entire symbol
        all_points = np.concatenate([cnt for cnt in contours])
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Log the bounding rectangle
        logger.debug(f"Bounding rectangle: x={x}, y={y}, width={w}, height={h}")
        
        # Ensure we have a reasonable size
        if w < 20 or h < 20:
            logger.warning(f"Contour too small (w={w}, h={h}), using default quadrants")
            x, y = 0, 0
            w, h = width, height
    
    # Find the approximate center of the symbol
    center_x = x + w // 2
    center_y = y + h // 2
    logger.debug(f"Center point: ({center_x}, {center_y})")
    
    # Estimate the stem position (vertical line through center)
    # In a proper Cistercian numeral, the stem divides the symbol vertically
    stem_x = center_x
    stem_top = y
    stem_bottom = y + h
    
    # Define the four quadrants with clear naming
    # Each quadrant is defined as (x1, y1, x2, y2) coordinates
    # Make sure these are integers to avoid indexing issues
    quadrants = {
        'top-left': (int(x), int(y), int(stem_x), int(center_y)),
        'top-right': (int(stem_x), int(y), int(x + w), int(center_y)),
        'bottom-left': (int(x), int(center_y), int(stem_x), int(y + h)),
        'bottom-right': (int(stem_x), int(center_y), int(x + w), int(y + h))
    }
    
    # Log the quadrant coordinates for debugging
    for name, coords in quadrants.items():
        logger.debug(f"Quadrant {name}: {coords}")
    
    # Debug: Draw quadrant boundaries on a copy of the image for visualization
    visual_debug = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
    cv2.line(visual_debug, (stem_x, stem_top), (stem_x, stem_bottom), (0, 255, 0), 2)
    cv2.line(visual_debug, (x, center_y), (x + w, center_y), (0, 255, 0), 2)
    # We're not saving the debug image here as we don't have write permissions
    # but we can uncomment this if needed for debugging
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
    
    # Simplify the feature detection for more reliable results
    # We'll check the most distinctive features first
    
    # Debug: Log the quadrant features
    logger.debug(f"Quadrant features: horiz={horizontal_lines}, vert={vertical_lines}, diag={diagonal_lines}, rect={rectangles}, contours={num_contours}, fill_ratio={fill_ratio:.2f}")
    
    # Add more info about the quadrant 
    quadrant_name = "unknown"
    if "top-left" in str(quadrant_coords):
        quadrant_name = "top-left"
    elif "top-right" in str(quadrant_coords):
        quadrant_name = "top-right"
    elif "bottom-left" in str(quadrant_coords):
        quadrant_name = "bottom-left"
    elif "bottom-right" in str(quadrant_coords):
        quadrant_name = "bottom-right"
    
    logger.debug(f"Processing quadrant: {quadrant_name}")
    
    # Very simple check: if the quadrant is mostly empty, return 0
    if fill_ratio < 0.02 or num_contours == 0:
        logger.debug(f"Quadrant {quadrant_name} is mostly empty, returning 0")
        return 0
    
    # SIMPLIFIED LOGIC FOR MORE RELIABLE DETECTION
    
    # Digit 1: Simple horizontal line - most common and distinctive
    if horizontal_lines >= 1 and vertical_lines == 0 and diagonal_lines == 0:
        logger.debug(f"Detected digit 1 in {quadrant_name}")
        return 1
    
    # Diagonal lines detection - for digits 2 and 6
    # Let's try a simpler approach first by just checking for diagonal lines
    if diagonal_lines >= 1 and horizontal_lines == 0 and vertical_lines == 0:
        # Digit 2: Diagonal from stem going up-right or up-left depending on quadrant
        # Digit 6: Diagonal from stem going down-right or down-left depending on quadrant
        # In simplest implementation:
        if quadrant_name in ["top-right", "bottom-left"]:
            logger.debug(f"Detected digit 2 in {quadrant_name}")
            return 2
        else:
            logger.debug(f"Detected digit 6 in {quadrant_name}")
            return 6
    
    # Digit 3 and 7: Horizontal line with diagonal
    if horizontal_lines >= 1 and diagonal_lines >= 1:
        if quadrant_name in ["top-right", "bottom-left"]:
            logger.debug(f"Detected digit 3 in {quadrant_name}")
            return 3
        else:
            logger.debug(f"Detected digit 7 in {quadrant_name}")
            return 7
    
    # Digit 4: L-shape (horizontal and vertical lines)
    if horizontal_lines >= 1 and vertical_lines >= 1:
        logger.debug(f"Detected digit 4 in {quadrant_name}")
        return 4
    
    # Digit 5: Higher complexity or fill ratio
    if rectangles >= 1 or fill_ratio > 0.3 or num_contours >= 3:
        logger.debug(f"Detected digit 5 in {quadrant_name}")
        return 5
    
    # Digit 8: V-shape or multiple diagonal lines
    if diagonal_lines >= 2:
        logger.debug(f"Detected digit 8 in {quadrant_name}")
        return 8
    
    # Digit 9: Square/rectangle shape or high area coverage
    if rectangles >= 1 or fill_ratio > 0.25:
        logger.debug(f"Detected digit 9 in {quadrant_name}")
        return 9
    
    # If we detect any significant contours but couldn't identify a digit,
    # determine the most likely digit based on simplified rules
    if vertical_lines >= 1:
        return 1  # Vertical lines might be misdetected horizontals
    elif diagonal_lines >= 1:
        return 2  # Default to simplest diagonal digit
    elif horizontal_lines >= 1:
        return 1  # Default to simplest digit
    elif fill_ratio > 0.1:
        return 5  # Default to filled shape
    
    # Fall back to basic rule - if there's anything in the quadrant
    # but we couldn't classify it, return 1 as the safest guess
    if np.sum(quadrant_img) > 0:
        logger.debug(f"Fallback: Detected digit 1 in {quadrant_name}")
        return 1
    
    # Nothing meaningful detected
    logger.debug(f"No digit detected in {quadrant_name}")
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
        
        # Save a debug copy of the preprocessed image
        # cv2.imwrite('/tmp/preprocessed.png', binary_image)
        
        # Find the stem and quadrants
        structure = find_stem_and_quadrants(binary_image)
        if not structure:
            logger.warning("Could not identify structure in the image")
            return 0
        
        # Extract digits from each quadrant
        quadrants = structure['quadrants']
        
        # Debug: Check quadrant coordinates
        logger.debug(f"Quadrant coordinates: {quadrants}")
        
        # Extract digits properly from each quadrant
        # The quadrant placement must match exactly how we draw them in cistercian_utils.py
        # Units = bottom right
        # Tens = top right
        # Hundreds = bottom left
        # Thousands = top left
        units_digit = detect_features_in_quadrant(binary_image, quadrants['bottom-right'])
        tens_digit = detect_features_in_quadrant(binary_image, quadrants['top-right'])
        hundreds_digit = detect_features_in_quadrant(binary_image, quadrants['bottom-left'])
        thousands_digit = detect_features_in_quadrant(binary_image, quadrants['top-left'])
        
        # Debug: Log detected digits
        logger.debug(f"Detected digits: units={units_digit}, tens={tens_digit}, hundreds={hundreds_digit}, thousands={thousands_digit}")
        
        # Combine digits to form the number
        number = (
            thousands_digit * 1000 +
            hundreds_digit * 100 +
            tens_digit * 10 +
            units_digit
        )
        
        return number
    
    except Exception as e:
        logger.error(f"Error in Cistercian numeral recognition: {str(e)}")
        # Return default value in case of error
        return 0
