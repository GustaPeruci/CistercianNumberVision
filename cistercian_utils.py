import cv2
import numpy as np
import base64
from PIL import Image
import io
import math

def decode_base64_image(base64_str):
    """Decode a base64 image string to a numpy array."""
    # Remove header if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # Decode base64 string
    img_data = base64.b64decode(base64_str)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode to image
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img

def encode_image_to_base64(img):
    """Encode a numpy array image to a base64 string."""
    _, buffer = cv2.imencode('.png', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

def create_blank_image(width=300, height=400):
    """Create a blank white image with given dimensions."""
    img = np.ones((height, width), np.uint8) * 255
    return img

def draw_cistercian_symbol(img, number):
    """
    Draw a Cistercian numeral on the image for the given number (0-9999).
    
    The Cistercian numeral system uses a vertical stem with different marks
    in four quadrants to represent units, tens, hundreds, and thousands.
    """
    if number < 0 or number > 9999:
        raise ValueError("Number must be between 0 and 9999")
    
    # Image dimensions and positions
    height, width = img.shape
    center_x = width // 2
    center_y = height // 2
    stem_height = height // 1.5
    stem_top = center_y - stem_height // 2
    stem_bottom = center_y + stem_height // 2
    line_thickness = 3
    
    # Draw the vertical stem
    cv2.line(img, (center_x, int(stem_top)), (center_x, int(stem_bottom)), 0, line_thickness)
    
    # Extract digits
    digits = [0, 0, 0, 0]  # initialize with zeros [units, tens, hundreds, thousands]
    temp = number
    for i in range(4):
        digits[i] = temp % 10
        temp //= 10
    
    # Draw units (bottom right)
    draw_digit(img, digits[0], center_x, stem_bottom, 'bottom-right', line_thickness)
    
    # Draw tens (top right)
    draw_digit(img, digits[1], center_x, stem_top, 'top-right', line_thickness)
    
    # Draw hundreds (bottom left)
    draw_digit(img, digits[2], center_x, stem_bottom, 'bottom-left', line_thickness)
    
    # Draw thousands (top left)
    draw_digit(img, digits[3], center_x, stem_top, 'top-left', line_thickness)
    
    return img

def draw_digit(img, digit, center_x, y_pos, quadrant, thickness):
    """
    Draw a digit (0-9) in the specified quadrant.
    quadrant: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    """
    if digit == 0:
        return  # No symbol for 0
    
    # Determine direction based on quadrant
    x_dir = 1 if 'right' in quadrant else -1
    y_dir = 1 if 'bottom' in quadrant else -1
    
    # Size of the digit symbols
    symbol_size = 50  # Can be adjusted
    
    # Draw the digit symbol based on its value
    if digit == 1:
        # Horizontal line from stem
        cv2.line(img, (center_x, y_pos), (center_x + x_dir * symbol_size, y_pos), 0, thickness)
    
    elif digit == 2:
        # Diagonal line from stem (up)
        end_x = center_x + x_dir * symbol_size
        end_y = y_pos - symbol_size
        cv2.line(img, (center_x, y_pos), (end_x, end_y), 0, thickness)
    
    elif digit == 3:
        # Diagonal line from stem (up) with horizontal line
        end_x = center_x + x_dir * symbol_size
        end_y = y_pos - symbol_size
        cv2.line(img, (center_x, y_pos), (end_x, end_y), 0, thickness)
        cv2.line(img, (center_x, y_pos), (center_x + x_dir * symbol_size, y_pos), 0, thickness)
    
    elif digit == 4:
        # Horizontal line with vertical extension down
        cv2.line(img, (center_x, y_pos), (center_x + x_dir * symbol_size, y_pos), 0, thickness)
        cv2.line(img, (center_x + x_dir * symbol_size, y_pos), 
                 (center_x + x_dir * symbol_size, y_pos + symbol_size), 0, thickness)
    
    elif digit == 5:
        # Symbol for 5 (combination of 1 and 4)
        cv2.line(img, (center_x, y_pos), (center_x + x_dir * symbol_size, y_pos), 0, thickness)
        cv2.line(img, (center_x + x_dir * symbol_size, y_pos), 
                 (center_x + x_dir * symbol_size, y_pos + symbol_size), 0, thickness)
        cv2.line(img, (center_x + x_dir * symbol_size, y_pos + symbol_size), 
                 (center_x, y_pos + symbol_size), 0, thickness)
    
    elif digit == 6:
        # Diagonal line from stem (down)
        end_x = center_x + x_dir * symbol_size
        end_y = y_pos + symbol_size
        cv2.line(img, (center_x, y_pos), (end_x, end_y), 0, thickness)
    
    elif digit == 7:
        # Diagonal line from stem (down) with horizontal line
        end_x = center_x + x_dir * symbol_size
        end_y = y_pos + symbol_size
        cv2.line(img, (center_x, y_pos), (end_x, end_y), 0, thickness)
        cv2.line(img, (center_x, y_pos), (center_x + x_dir * symbol_size, y_pos), 0, thickness)
    
    elif digit == 8:
        # V shape
        mid_x = center_x + x_dir * (symbol_size // 2)
        end_x = center_x + x_dir * symbol_size
        cv2.line(img, (center_x, y_pos), (mid_x, y_pos + symbol_size), 0, thickness)
        cv2.line(img, (mid_x, y_pos + symbol_size), (end_x, y_pos), 0, thickness)
    
    elif digit == 9:
        # Circle or square (using a small rectangle for simplicity)
        top_left = (center_x, y_pos - symbol_size // 2)
        bottom_right = (center_x + x_dir * symbol_size, y_pos + symbol_size // 2)
        if x_dir < 0:  # Adjust for left quadrants
            top_left = (center_x + x_dir * symbol_size, y_pos - symbol_size // 2)
            bottom_right = (center_x, y_pos + symbol_size // 2)
        cv2.rectangle(img, top_left, bottom_right, 0, thickness)

def number_to_cistercian_image(number):
    """Convert a number to a Cistercian numeral image and return as base64."""
    # Validate input
    if number < 0 or number > 9999:
        raise ValueError("Number must be between 0 and 9999")
    
    # Create a blank image
    img = create_blank_image(300, 400)
    
    # Draw the Cistercian numeral
    img = draw_cistercian_symbol(img, number)
    
    # Return base64 encoded image
    return encode_image_to_base64(img)
