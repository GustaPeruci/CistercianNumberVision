import cv2
import numpy as np
import base64
from PIL import Image
import os

def decode_base64_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # Decode base64 string
    img_data = base64.b64decode(base64_str)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode to image
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # print(f"Decoded image shape: {img.shape}")
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
    stem_height = int(height // 1.5)  # Ensure integer values
    stem_top = int(center_y - stem_height // 2)
    stem_bottom = int(center_y + stem_height // 2)
    line_thickness = 3
    
    # Draw the vertical stem
    cv2.line(img, (center_x, stem_top), (center_x, stem_bottom), 0, line_thickness)
    
    # Extract digits - we need to handle each place value correctly according to Cistercian rules:
    # Units (1-9): TOP RIGHT quadrant
    # Tens (10-90): TOP LEFT quadrant
    # Hundreds (100-900): BOTTOM RIGHT quadrant
    # Thousands (1000-9000): BOTTOM LEFT quadrant
    units = number % 10
    tens = (number // 10) % 10
    hundreds = (number // 100) % 10
    thousands = (number // 1000) % 10
    
    # Draw units (TOP RIGHT)
    draw_digit(img, units, center_x, stem_top, 'top-right', line_thickness)
    
    # Draw tens (TOP LEFT)
    draw_digit(img, tens, center_x, stem_top, 'top-left', line_thickness)
    
    # Draw hundreds (BOTTOM RIGHT)
    draw_digit(img, hundreds, center_x, stem_bottom, 'bottom-right', line_thickness)
    
    # Draw thousands (BOTTOM LEFT)
    draw_digit(img, thousands, center_x, stem_bottom, 'bottom-left', line_thickness)
    
    return img

def draw_digit(img, digit, center_x, y_pos, quadrant, thickness):
    """
    Draw a digit (0-9) in the specified quadrant according to Cistercian numeral system.
    quadrant: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    """
    if digit == 0:
        return  # No symbol for 0
    
    # Ensure coordinates are integers
    center_x = int(center_x)
    y_pos = int(y_pos)
    
    # Determine direction based on quadrant
    x_dir = 1 if 'right' in quadrant else -1
    y_dir = 1 if 'bottom' in quadrant else -1
    
    # Size of the digit symbols
    symbol_size = 50  # Can be adjusted
    
    # Draw the digit symbol based on its value and quadrant
    # The symbols match the reference image provided
    
    if digit == 1:
        # 1: Horizontal line from stem
        end_x = int(center_x + x_dir * symbol_size)
        cv2.line(img, (center_x, y_pos), (end_x, y_pos), 0, thickness)
    
    elif digit == 2:
        horiz_end_x = int(center_x + x_dir * symbol_size)
        vert_end_y = int(y_pos - y_dir * symbol_size)
        
        cv2.line(img, (center_x, vert_end_y), (horiz_end_x, vert_end_y), 0, thickness)   
    
    elif digit == 3:        
        end_x = int(center_x + x_dir * symbol_size)
        end_y = int(y_pos - y_dir * symbol_size)
        cv2.line(img, (center_x, y_pos), (end_x, end_y), 0, thickness)
    
    elif digit == 4:
        end_x = int(center_x + x_dir * symbol_size)
        end_y = int(y_pos - y_dir * symbol_size)
        cv2.line(img, (center_x, end_y), (end_x, y_pos), 0, thickness)
    
    elif digit == 5:
        # 5: Diagonal with horizontal at end (like \_)
        # Diagonal line
        end_x = int(center_x + x_dir * symbol_size)
        end_y = int(y_pos - y_dir * symbol_size)
        cv2.line(img, (center_x, end_y), (end_x, y_pos), 0, thickness)

        cv2.line(img, (center_x, y_pos), (end_x, y_pos), 0, thickness)
    
    elif digit == 6:
        horiz_end_x = int(center_x + x_dir * symbol_size)
        
        vert_end_y = int(y_pos - y_dir * symbol_size)
        cv2.line(img, (horiz_end_x, y_pos), (horiz_end_x, vert_end_y), 0, thickness)
    
    elif digit == 7:
        horiz_end_x = int(center_x + x_dir * symbol_size)
        cv2.line(img, (center_x, y_pos), (horiz_end_x, y_pos), 0, thickness)
        
        # The full-height vertical line at the end
        vert_end_y = int(y_pos - y_dir * symbol_size)
        cv2.line(img, (horiz_end_x, y_pos), (horiz_end_x, vert_end_y), 0, thickness)
    
    
    elif digit == 8:
        
        horiz_end_x = int(center_x + x_dir * symbol_size)
        vert_end_y = int(y_pos - y_dir * symbol_size)
        
        cv2.line(img, (center_x, vert_end_y), (horiz_end_x, vert_end_y), 0, thickness)    
        cv2.line(img, (horiz_end_x, y_pos), (horiz_end_x, vert_end_y), 0, thickness)
    
    elif digit == 9:
        # 9: Square/rectangle shape
        horiz_end_x = int(center_x + x_dir * symbol_size)
        vert_end_y = int(y_pos - y_dir * symbol_size)
        
        # Draw the rectangle
        cv2.line(img, (center_x, y_pos), (horiz_end_x, y_pos), 0, thickness)  # Bottom
        cv2.line(img, (horiz_end_x, y_pos), (horiz_end_x, vert_end_y), 0, thickness)  # Right
        cv2.line(img, (horiz_end_x, vert_end_y), (center_x, vert_end_y), 0, thickness)  # Top
        # Left side is the stem itself
        
def save_image_to_file(img, filename):
    """Save the image to a file."""
    # Ensure the directory exists
    os.makedirs('created_numbers', exist_ok=True)
    cv2.imwrite('created_numbers/' + filename, img)

def number_to_cistercian_image(number):
    """Convert a number to a Cistercian numeral image and return as base64."""
    # Validate input
    if number < 0 or number > 9999:
        raise ValueError("Number must be between 0 and 9999")
    
    # Create a blank image
    img = create_blank_image(300, 400)
    
    # Draw the Cistercian numeral
    img = draw_cistercian_symbol(img, number)
    
    save_image_to_file(img, f'cistercian_{number}.png')
    
    # Return base64 encoded image
    return encode_image_to_base64(img)

def generate_all_cistercian_templates():
    output_dir = 'created_numbers'
    os.makedirs(output_dir, exist_ok=True)
    
    for number in range(1, 10000):
        img = create_blank_image(300, 400)
        img = draw_cistercian_symbol(img, number)
        filename = f'{output_dir}/cistercian_{number}.png'
        cv2.imwrite(filename, img)
        if number % 100 == 0:
            print(f'{number} imagens geradas...')
    
    print("✅ Todas as 9999 imagens foram geradas com sucesso na pasta 'created_numbers/'")

generate_all_cistercian_templates()


