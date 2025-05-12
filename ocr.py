import cv2
import numpy as np
import logging
from collections import defaultdict

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CistercianRecognizer:
    def __init__(self):
        # Definir os padrões básicos para cada dígito em cada posição
        self.digit_patterns = {
            'units': {
                1: ['top_vertical'],
                2: ['top_horizontal_left'],
                3: ['top_vertical', 'top_horizontal_left'],
                4: ['full_vertical'],
                5: ['top_vertical', 'top_horizontal_right'],
                6: ['top_horizontal', 'mid_vertical_down'],
                7: ['top_vertical', 'top_horizontal', 'mid_vertical_down'],
                8: ['full_vertical', 'mid_horizontal'],
                9: ['full_vertical', 'mid_horizontal', 'top_right_diagonal']
            },
            'tens': {
                1: ['bottom_vertical'],
                2: ['bottom_horizontal_left'],
                3: ['bottom_vertical', 'bottom_horizontal_left'],
                4: ['full_vertical'],
                5: ['bottom_vertical', 'bottom_horizontal_right'],
                6: ['bottom_horizontal', 'mid_vertical_up'],
                7: ['bottom_vertical', 'bottom_horizontal', 'mid_vertical_up'],
                8: ['full_vertical', 'mid_horizontal'],
                9: ['full_vertical', 'mid_horizontal', 'bottom_right_diagonal']
            },
            'hundreds': {
                1: ['left_vertical_top'],
                2: ['left_horizontal_top'],
                3: ['left_vertical_top', 'left_horizontal_top'],
                4: ['left_full_vertical'],
                5: ['left_vertical_top', 'left_horizontal_bottom'],
                6: ['left_horizontal', 'left_vertical_mid_right'],
                7: ['left_vertical_top', 'left_horizontal', 'left_vertical_mid_right'],
                8: ['left_full_vertical', 'left_horizontal'],
                9: ['left_full_vertical', 'left_horizontal', 'left_diagonal_bottom_right']
            },
            'thousands': {
                1: ['right_vertical_top'],
                2: ['right_horizontal_top'],
                3: ['right_vertical_top', 'right_horizontal_top'],
                4: ['right_full_vertical'],
                5: ['right_vertical_top', 'right_horizontal_bottom'],
                6: ['right_horizontal', 'right_vertical_mid_left'],
                7: ['right_vertical_top', 'right_horizontal', 'right_vertical_mid_left'],
                8: ['right_full_vertical', 'right_horizontal'],
                9: ['right_full_vertical', 'right_horizontal', 'right_diagonal_bottom_left']
            }
        }

    def preprocess_image(self, image):
        """Pré-processa a imagem para melhorar o reconhecimento"""
        try:
            # Converter para escala de cinza se necessário
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Aplicar limiarização adaptativa
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Reduzir ruído
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            return processed
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise

    def detect_staff(self, image):
        """Detecta o bastão vertical principal do numeral Cisterciense"""
        try:
            # Usar transformada de Hough para detectar linhas
            lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=50, 
                                   minLineLength=image.shape[0]//2, maxLineGap=10)
            
            if lines is None:
                return None
            
            # Encontrar a linha vertical mais central e longa
            vertical_lines = []
            center_x = image.shape[1] // 2
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 5:  # Linha quase vertical
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    # Priorizar linhas mais próximas do centro
                    distance_from_center = abs((x1 + x2)/2 - center_x)
                    vertical_lines.append((line[0], length, distance_from_center))
            
            if not vertical_lines:
                return None
            
            # Ordenar por comprimento e proximidade do centro
            vertical_lines.sort(key=lambda x: (-x[1], x[2]))
            staff = vertical_lines[0][0]
            
            return staff
        except Exception as e:
            logger.error(f"Error detecting staff: {str(e)}")
            raise

    def detect_segments(self, image, staff):
        """Detecta segmentos de linha na imagem em relação ao bastão principal"""
        try:
            height, width = image.shape
            staff_x = (staff[0] + staff[2]) // 2  # Posição x média do bastão
            staff_y1, staff_y2 = min(staff[1], staff[3]), max(staff[1], staff[3])
            
            # Detectar todas as linhas na imagem
            lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=30, 
                                   minLineLength=20, maxLineGap=10)
            
            if lines is None:
                return []
            
            segments = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Ignorar o próprio bastão principal
                if abs(x1 - staff_x) < 5 and abs(x2 - staff_x) < 5:
                    continue
                
                # Classificar o segmento baseado na posição relativa ao bastão
                segment_type = self.classify_segment(x1, y1, x2, y2, staff_x, staff_y1, staff_y2, height, width)
                if segment_type:
                    segments.append(segment_type)
            
            return segments
        except Exception as e:
            logger.error(f"Error detecting segments: {str(e)}")
            raise

    def classify_segment(self, x1, y1, x2, y2, staff_x, staff_y1, staff_y2, img_h, img_w):
        """Classifica um segmento de linha em relação ao bastão principal"""
        # Calcular ângulo do segmento (em graus)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Determinar se está à esquerda ou direita do bastão
        position = 'left' if mid_x < staff_x else 'right'
        
        # Classificar segmentos verticais (80-100 graus)
        if 80 <= angle <= 100:
            if position == 'left':
                if mid_y < staff_y1 + (staff_y2 - staff_y1)/3:
                    return 'left_vertical_top'
                elif mid_y > staff_y1 + 2*(staff_y2 - staff_y1)/3:
                    return 'left_vertical_bottom'
                else:
                    return 'left_vertical_mid'
            else:
                if mid_y < staff_y1 + (staff_y2 - staff_y1)/3:
                    return 'right_vertical_top'
                elif mid_y > staff_y1 + 2*(staff_y2 - staff_y1)/3:
                    return 'right_vertical_bottom'
                else:
                    return 'right_vertical_mid'
        
        # Classificar segmentos horizontais (0-10 ou 170-180 graus)
        elif angle <= 10 or angle >= 170:
            if position == 'left':
                if mid_y < staff_y1 + (staff_y2 - staff_y1)/3:
                    return 'left_horizontal_top'
                elif mid_y > staff_y1 + 2*(staff_y2 - staff_y1)/3:
                    return 'left_horizontal_bottom'
                else:
                    return 'left_horizontal_mid'
            else:
                if mid_y < staff_y1 + (staff_y2 - staff_y1)/3:
                    return 'right_horizontal_top'
                elif mid_y > staff_y1 + 2*(staff_y2 - staff_y1)/3:
                    return 'right_horizontal_bottom'
                else:
                    return 'right_horizontal_mid'
        
        # Classificar diagonais (40-50 ou 130-140 graus)
        elif 40 <= angle <= 50:
            return f'{position}_diagonal_down'
        elif 130 <= angle <= 140:
            return f'{position}_diagonal_up'
        
        return None

    def recognize_digit(self, segments, digit_type):
        """Reconhece um dígito específico baseado nos segmentos detectados"""
        best_match = 0
        best_score = 0
        
        for digit, patterns in self.digit_patterns[digit_type].items():
            score = 0
            required_segments = set(patterns)
            present_segments = set(segments)
            
            # Pontuar pela quantidade de segmentos correspondentes
            score = len(required_segments & present_segments)
            
            # Penalizar segmentos ausentes ou extras
            missing = len(required_segments - present_segments)
            extra = len(present_segments - required_segments)
            score -= (missing + extra) * 0.5
            
            if score > best_score:
                best_score = score
                best_match = digit
        
        return best_match if best_score > 0.5 else 0

    def recognize_cistercian_numeral(self, image):
        """Reconhece um numeral Cisterciense em uma imagem e retorna o número arábico"""
        try:
            # Pré-processar a imagem
            processed = self.preprocess_image(image)
            
            # Detectar o bastão principal
            staff = self.detect_staff(processed)
            if staff is None:
                logger.warning("No staff detected in the image")
                return 0
            
            # Detectar todos os segmentos
            segments = self.detect_segments(processed, staff)
            logger.debug(f"Detected segments: {segments}")
            
            # Reconhecer cada dígito
            units = self.recognize_digit([s for s in segments if 'top' in s or 'full' in s or 'mid_vertical_down' in s], 'units')
            tens = self.recognize_digit([s for s in segments if 'bottom' in s or 'full' in s or 'mid_vertical_up' in s], 'tens')
            hundreds = self.recognize_digit([s for s in segments if 'left' in s], 'hundreds')
            thousands = self.recognize_digit([s for s in segments if 'right' in s], 'thousands')
            
            # Calcular o número final
            number = thousands * 1000 + hundreds * 100 + tens * 10 + units
            
            logger.debug(f"Recognized number: {number} (U:{units}, T:{tens}, H:{hundreds}, Th:{thousands})")
            return number
        except Exception as e:
            logger.error(f"Error recognizing numeral: {str(e)}")
            return 0

# Função de interface para o Flask
def recognize_cistercian_numeral(image):
    recognizer = CistercianRecognizer()
    return recognizer.recognize_cistercian_numeral(image)