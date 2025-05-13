import cv2
import numpy as np
import base64
from PIL import Image
import os

def decode_base64_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    
    nparr = np.frombuffer(img_data, np.uint8)
    
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    return img

def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

def create_blank_image(width=300, height=400):
    img = np.ones((height, width), np.uint8) * 255
    return img

def draw_cistercian_symbol(img, number):
    if number < 0 or number > 9999:
        raise ValueError("Number must be between 0 and 9999")

    height, width = img.shape
    center_x = width // 2
    center_y = height // 2
    stem_height = int(height // 1.5)
    stem_top = int(center_y - stem_height // 2)
    stem_bottom = int(center_y + stem_height // 2)
    line_thickness = 3
    
    cv2.line(img, (center_x, stem_top), (center_x, stem_bottom), 0, line_thickness)
    
    units = number % 10
    tens = (number // 10) % 10
    hundreds = (number // 100) % 10
    thousands = (number // 1000) % 10
    
    draw_digit(img, units, center_x, stem_top, 'top-right', line_thickness)
    
    draw_digit(img, tens, center_x, stem_top, 'top-left', line_thickness)
    
    draw_digit(img, hundreds, center_x, stem_bottom, 'bottom-right', line_thickness)

    draw_digit(img, thousands, center_x, stem_bottom, 'bottom-left', line_thickness)
    
    return img

def draw_digit(img, digit, center_x, y_pos, quadrant, thickness):
   
    if digit == 0:
        return  
    
    center_x = int(center_x)
    y_pos = int(y_pos)
    
    x_dir = 1 if 'right' in quadrant else -1
    y_dir = 1 if 'bottom' in quadrant else -1
    
    symbol_size = 50     
    
    if digit == 1:
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

        vert_end_y = int(y_pos - y_dir * symbol_size)
        cv2.line(img, (horiz_end_x, y_pos), (horiz_end_x, vert_end_y), 0, thickness)
    
    
    elif digit == 8:
        
        horiz_end_x = int(center_x + x_dir * symbol_size)
        vert_end_y = int(y_pos - y_dir * symbol_size)
        
        cv2.line(img, (center_x, vert_end_y), (horiz_end_x, vert_end_y), 0, thickness)    
        cv2.line(img, (horiz_end_x, y_pos), (horiz_end_x, vert_end_y), 0, thickness)
    
    elif digit == 9:
        horiz_end_x = int(center_x + x_dir * symbol_size)
        vert_end_y = int(y_pos - y_dir * symbol_size)
        
        cv2.line(img, (center_x, y_pos), (horiz_end_x, y_pos), 0, thickness)  # Bottom
        cv2.line(img, (horiz_end_x, y_pos), (horiz_end_x, vert_end_y), 0, thickness)  # Right
        cv2.line(img, (horiz_end_x, vert_end_y), (center_x, vert_end_y), 0, thickness)  # Top
        
def save_image_to_file(img, filename):
    os.makedirs('created_numbers', exist_ok=True)
    cv2.imwrite('created_numbers/' + filename, img)

def number_to_cistercian_image(number):
    if number < 0 or number > 9999:
        raise ValueError("Number must be between 0 and 9999")
    
    img = create_blank_image(300, 400)
    
    img = draw_cistercian_symbol(img, number)
    
    save_image_to_file(img, f'cistercian_{number}.png')
    
    return encode_image_to_base64(img)

def generate_all_cistercian_templates():
    output_dir = 'created_numbers'
    os.makedirs(output_dir, exist_ok=True)
    
    for number in range(1, 10000):
        img = create_blank_image(300, 400)
        img = draw_cistercian_symbol(img, number)
        filename = f'{output_dir}/cistercian_{number}.png'
        cv2.imwrite(filename, img)
        if number % 1000 == 0:
            print(f'{number} imagens geradas...')
    
    print("✅ Todas as 9999 imagens foram geradas com sucesso na pasta 'created_numbers/'")


