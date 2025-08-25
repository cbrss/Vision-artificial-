import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import math
import time

# Parámetros de visualización y ventana
MARGIN = 10  # Margen para los textos de las cajas
ROW_SIZE = 10  # Espaciado vertical para los textos de las cajas
FONT_SIZE = 2
FONT_THICKNESS = 2
TEXT_COLOR = (255, 0, 0)  # Rojo para las cajas y etiquetas
RESOLUTION = (1280, 720)  # Resolución de la ventana (ancho, alto)

# Parámetros del auto
CAR_SIZE = 40
CAR_SPEED = 2  # píxeles por frame
CAR_COLOR = (0, 255, 0)  # Verde para el auto
STOP_COLOR = (0, 0, 255)  # Rojo cuando está detenido

class Car:
    def __init__(self, canvas_width, canvas_height):
        self.x = 50  # Posición inicial: abajo izquierda
        self.y = canvas_height - 100
        self.width = canvas_width
        self.height = canvas_height
        self.speed = CAR_SPEED
        self.is_stopped = False
        self.stop_time = 0
        self.stop_duration = 3  # segundos que se detiene
        self.follow_cursor = False  # Modo de seguimiento del cursor
        self.target_x = self.x
        self.target_y = self.y
        
    def update(self, stop_detected=False, mouse_x=None, mouse_y=None):
        """Actualiza la posición del auto"""
        if stop_detected and not self.is_stopped:
            self.is_stopped = True
            self.stop_time = time.time()
            return
            
        if self.is_stopped:
            if time.time() - self.stop_time > self.stop_duration:
                self.is_stopped = False
            else:
                return  # El auto permanece detenido
        
        if self.follow_cursor and mouse_x is not None and mouse_y is not None:
            # Modo seguimiento del cursor
            self.target_x = mouse_x
            self.target_y = mouse_y
            
            # Calcular dirección hacia el cursor
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 5:  # Solo mover si está a más de 5 píxeles del objetivo
                # Normalizar y aplicar velocidad
                dx = (dx / distance) * self.speed
                dy = (dy / distance) * self.speed
                self.x += dx
                self.y += dy
        else:
            # Modo patrón rectangular original
            # Movimiento del auto en patrón rectangular
            if self.x < self.target_x:  # Moviéndose hacia la derecha
                self.x += self.speed
                if self.x >= self.target_x:
                    self.x = self.target_x
                    self.target_y = 100
            elif self.y > self.target_y:  # Moviéndose hacia arriba
                self.y -= self.speed
                if self.y <= self.target_y:
                    self.y = self.target_y
                    self.target_x = 50
            elif self.x > self.target_x:  # Moviéndose hacia la izquierda
                self.x -= self.speed
                if self.x <= self.target_x:
                    self.x = self.target_x
                    self.target_y = self.height - 100
            elif self.y < self.target_y:  # Moviéndose hacia abajo
                self.y += self.speed
                if self.y >= self.target_y:
                    self.y = self.target_y
                    self.target_x = self.width - 100
    
    def draw(self, canvas):
        """Dibuja el auto en el canvas"""
        color = STOP_COLOR if self.is_stopped else CAR_COLOR
        
        # Cuerpo del auto (rectángulo)
        cv2.rectangle(canvas, 
                     (int(self.x - CAR_SIZE//2), int(self.y - CAR_SIZE//4)), 
                     (int(self.x + CAR_SIZE//2), int(self.y + CAR_SIZE//4)), 
                     color, -1)
        
        # Ruedas (círculos)
        wheel_radius = CAR_SIZE // 6
        cv2.circle(canvas, (int(self.x - CAR_SIZE//3), int(self.y + CAR_SIZE//4)), wheel_radius, (0, 0, 0), -1)
        cv2.circle(canvas, (int(self.x + CAR_SIZE//3), int(self.y + CAR_SIZE//4)), wheel_radius, (0, 0, 0), -1)
        cv2.circle(canvas, (int(self.x - CAR_SIZE//3), int(self.y - CAR_SIZE//4)), wheel_radius, (0, 0, 0), -1)
        cv2.circle(canvas, (int(self.x + CAR_SIZE//3), int(self.y - CAR_SIZE//4)), wheel_radius, (0, 0, 0), -1)
        
        # Faros (círculos pequeños)
        cv2.circle(canvas, (int(self.x - CAR_SIZE//3), int(self.y)), 3, (255, 255, 255), -1)
        cv2.circle(canvas, (int(self.x + CAR_SIZE//3), int(self.y)), 3, (255, 255, 255), -1)
        
        # Estado del auto
        if self.is_stopped:
            status_text = "AUTO DETENIDO"
        elif self.follow_cursor:
            status_text = "AUTO SIGUIENDO CURSOR"
        else:
            status_text = "AUTO EN MOVIMIENTO"
        cv2.putText(canvas, status_text, (10, RESOLUTION[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def center_on_canvas(image, resolution):
    """
    Centra una imagen en un lienzo del tamaño especificado.
    Si la imagen es más grande, se recorta; si es más pequeña, se rellena con negro.
    """
    h_img, w_img = image.shape[:2]
    w_canvas, h_canvas = resolution
    canvas = np.zeros((h_canvas, w_canvas, 3), dtype=np.uint8)
    x0 = max((w_canvas - w_img) // 2, 0)
    y0 = max((h_canvas - h_img) // 2, 0)
    x1 = x0 + min(w_img, w_canvas)
    y1 = y0 + min(h_img, h_canvas)
    img_x0 = 0 if w_img <= w_canvas else (w_img - w_canvas) // 2
    img_y0 = 0 if h_img <= h_canvas else (h_img - h_canvas) // 2
    img_x1 = img_x0 + min(w_img, w_canvas)
    img_y1 = img_y0 + min(h_img, h_canvas)
    canvas[y0:y1, x0:x1] = image[img_y0:img_y1, img_x0:img_x1]
    return canvas


def draw_detections(image, detection_result):
    """
    Dibuja las cajas y etiquetas de las detecciones sobre la imagen.
    """
    stop_detected = False
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f'{category_name} ({probability})'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
        
        # Verificar si se detectó una señal de stop
        if category_name.lower() in ['stop sign', 'stop', 'señal de stop']:
            stop_detected = True
            # Resaltar la señal de stop con un color especial
            cv2.rectangle(image, start_point, end_point, (0, 0, 255), 5)
            cv2.putText(image, "STOP SIGN", 
                       (bbox.origin_x, bbox.origin_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
    
    return image, stop_detected


def draw_help_text(canvas, help_text):
    """
    Dibuja los textos de ayuda (teclas y modo) en la esquina superior izquierda del lienzo.
    """
    if help_text:
        y0 = 30
        for i, line in enumerate(help_text.split('\n')):
            y = y0 + i * 25
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    return canvas

# Inicialización del modelo de detección
base_options = python.BaseOptions(model_asset_path='./model/efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Variables globales para el mouse
mouse_x = None
mouse_y = None

def mouse_callback(event, x, y, flags, param):
    """Callback para capturar la posición del mouse"""
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

# Inicialización del auto
car = Car(RESOLUTION[0], RESOLUTION[1])

# Carga de imágenes desde la carpeta ./img/
img_dir = './img/'
img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(img_exts)]
num_imgs = len(img_files)

mode = 'camera'  # Puede ser 'camera' o 'images'
img_idx = 0  # Índice de la imagen actual

# Inicialización de la cámara
cap = cv2.VideoCapture(0)
if cap is not None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
if not cap.isOpened():
    print("Could not open the camera.")
    cap = None

while True:
    if mode == 'camera' and cap is not None:
        # Modo camara
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        image_copy = np.copy(frame)
        annotated_image, stop_detected = draw_detections(image_copy, detection_result)
        canvas = center_on_canvas(annotated_image, RESOLUTION)
        
        # Actualizar y dibujar el auto
        car.update(stop_detected, mouse_x, mouse_y)
        car.draw(canvas)
        
        help_text = "Modo: Camara\n[c] Cambiar a imagenes\n[f] Seguir cursor\n[q] Salir"
        canvas = draw_help_text(canvas, help_text)
        cv2.imshow('Object Detection (Camera)', canvas)
        cv2.setMouseCallback('Object Detection (Camera)', mouse_callback)
        key = cv2.waitKey(1) & 0xFF
    elif mode == 'images' and num_imgs > 0:
        # Modo visualización de imágenes
        img_path = img_files[img_idx]
        frame = cv2.imread(img_path)
        if frame is None:
            print(f'Could not read image: {img_path}')
            img_idx = (img_idx + 1) % num_imgs
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        image_copy = np.copy(frame)
        annotated_image, stop_detected = draw_detections(image_copy, detection_result)
        canvas = center_on_canvas(annotated_image, RESOLUTION)
        
        # Actualizar y dibujar el auto
        car.update(stop_detected, mouse_x, mouse_y)
        car.draw(canvas)
        
        help_text = f"Modo: Imagenes ({img_idx+1}/{num_imgs})\n[c] Siguiente imagen\n[v] Volver a camara\n[f] Seguir cursor\n[q] Salir"
        canvas = draw_help_text(canvas, help_text)
        cv2.imshow('Object Detection (Images)', canvas)
        cv2.setMouseCallback('Object Detection (Images)', mouse_callback)
        key = cv2.waitKey(50) & 0xFF  # Cambiado de 0 a 50ms para permitir movimiento continuo
    else:
        # no hay imagenes, entonces lo mando a modo camara
        mode = 'camera'
        continue

    # Manejo de teclas
    if key == ord('q'):
        break
    elif key == ord('c') and num_imgs > 0:
        if mode == 'camera':
            mode = 'images'
            img_idx = 0
            cv2.destroyAllWindows()
        else:
            img_idx = (img_idx + 1) % num_imgs
    elif key == ord('v'):
        if mode == 'images':
            mode = 'camera'
            cv2.destroyAllWindows()
    elif key == ord('f'):
        # Cambiar modo de seguimiento del cursor
        car.follow_cursor = not car.follow_cursor
        if car.follow_cursor:
            print("🚗 Modo seguimiento del cursor ACTIVADO")
        else:
            print("🚗 Modo patrón rectangular ACTIVADO")

# Liberar recursos
if cap is not None:
    cap.release()
cv2.destroyAllWindows()