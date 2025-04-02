import os
import logging
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import io
import base64

from cistercian_utils import number_to_cistercian_image, decode_base64_image
from ocr import recognize_cistercian_numeral

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key_for_development")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure upload folder
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload size

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert-to-cistercian', methods=['POST'])
def convert_to_cistercian():
    try:
        data = request.json
        number = int(data.get('number', 0))
        
        # Validate input
        if number < 0 or number > 9999:
            return jsonify({'error': 'Number must be between 0 and 9999'}), 400
        
        # Generate Cistercian numeral image
        img_base64 = number_to_cistercian_image(number)
        
        return jsonify({'image': img_base64, 'number': number})
    except ValueError:
        return jsonify({'error': 'Invalid number format'}), 400
    except Exception as e:
        logger.error(f"Error converting to Cistercian: {str(e)}")
        return jsonify({'error': 'An error occurred during conversion'}), 500

@app.route('/recognize-cistercian', methods=['POST'])
def recognize_cistercian():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files and 'imageData' not in request.form:
            return jsonify({'error': 'No file or image data provided'}), 400
        
        image = None
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and allowed_file(file.filename):
                # Read the file
                in_memory_file = io.BytesIO()
                file.save(in_memory_file)
                data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        
        elif 'imageData' in request.form:
            # Handle base64 image data
            image_data = request.form['imageData']
            image = decode_base64_image(image_data)
        
        if image is None:
            return jsonify({'error': 'Could not process the image'}), 400
        
        # Recognize the Cistercian numeral
        number = recognize_cistercian_numeral(image)
        
        return jsonify({'number': number})
    except Exception as e:
        logger.error(f"Error recognizing Cistercian: {str(e)}")
        return jsonify({'error': 'An error occurred during recognition'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
