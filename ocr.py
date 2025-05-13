import os
import cv2
import numpy as np
import logging
import hashlib
import pickle

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def image_hash(img):
    return hashlib.sha256(img.tobytes()).hexdigest()

def load_templates(template_folder="created_numbers"):
    templates = {}
    hashes = {}
    for filename in os.listdir(template_folder):
        if filename.endswith(".png"):
            path = os.path.join(template_folder, filename)
            name = os.path.splitext(filename)[0]
            value = int(name.split('_')[1])
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            pre = preprocess_image_from_array(img)
            h = image_hash(pre)
            templates[value] = pre
            hashes[h] = value
    return templates, hashes

STANDARD_SIZE = (128, 256)

def preprocess_image_from_array(image_array):
    image_array = cv2.resize(image_array, STANDARD_SIZE)
    _, thresh = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thresh

def match_template(input_img, templates):
    best_match = None
    max_score = -1

    for value, template in templates.items():
        resized = cv2.resize(input_img, (template.shape[1], template.shape[0]))
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > max_score:
            max_score = score
            best_match = value

    return best_match, max_score

def recognize_cistercian_numeral(image_array, hashes=None):
    img = preprocess_image_from_array(image_array)
    h = image_hash(img)
    value = hashes.get(h)
    logger.info(f"Reconhecimento: {value} (hash match)")
    return value

def save_hash_cache(templates, hashes, filename="template_cache.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((templates, hashes), f)

def load_hash_cache(filename="template_cache.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None, None


