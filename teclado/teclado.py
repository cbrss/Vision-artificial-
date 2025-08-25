import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time

# ===================== CONFIGURACIÓN =====================
model_path = './model/hand_landmarker.task'  # Ruta a tu modelo .task

# Clases de MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Opciones del detector
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== CONFIGURACIÓN DEL TECLADO =====================
# Notas musicales (frecuencias en Hz)
NOTAS = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
    'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
    'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
}

# Configuración del teclado
TECLAS_BLANCAS = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
TECLAS_NEGRAS = ['C#', 'D#', 'F#', 'G#', 'A#']

# Inicializar pygame para audio
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

# ===================== CLASE PARA GENERAR SONIDOS =====================
class GeneradorSonido:
    def __init__(self):
        self.sample_rate = 44100
        self.duration = 0.3  # duración de la nota en segundos
        
    def generar_tono(self, frecuencia):
        """Genera un tono sinusoidal de la frecuencia especificada"""
        samples = int(self.sample_rate * self.duration)
        arr = np.zeros((samples, 2))
        for i in range(samples):
            val = int(32767 * 0.3 * np.sin(2 * np.pi * frecuencia * i / self.sample_rate))
            arr[i] = [val, val]
        return arr.astype(np.int16)
    
    def tocar_nota(self, nota):
        """Toca una nota musical"""
        if nota in NOTAS:
            frecuencia = NOTAS[nota]
            tono = self.generar_tono(frecuencia)
            sonido = pygame.sndarray.make_sound(tono)
            sonido.play()

# ===================== CLASE DEL TECLADO VIRTUAL =====================
class TecladoVirtual:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.teclas = {}
        self.teclas_activas = set()
        self.generador = GeneradorSonido()
        self.crear_teclado()
        
    def crear_teclado(self):
        """Crea las posiciones de las teclas del piano"""
        # Configuración del teclado
        teclado_y = int(self.frame_height * 0.7)  # Posición Y del teclado
        teclado_ancho = int(self.frame_width * 0.8)  # Ancho del teclado
        tecla_blanca_ancho = teclado_ancho // len(TECLAS_BLANCAS)
        tecla_altura = 80
        
        # Crear teclas blancas
        for i, nota in enumerate(TECLAS_BLANCAS):
            x = int(self.frame_width * 0.1) + i * tecla_blanca_ancho
            self.teclas[nota] = {
                'x': x, 'y': teclado_y,
                'ancho': tecla_blanca_ancho,
                'alto': tecla_altura,
                'color': (255, 255, 255),  # Blanco
                'color_activo': (200, 200, 200),  # Gris claro
                'es_negra': False
            }
        
        # Crear teclas negras
        for i, nota in enumerate(TECLAS_NEGRAS):
            # Posicionar teclas negras entre las blancas
            if nota == 'C#':
                x = int(self.frame_width * 0.1 + tecla_blanca_ancho * 0.7)
            elif nota == 'D#':
                x = int(self.frame_width * 0.1 + tecla_blanca_ancho * 1.7)
            elif nota == 'F#':
                x = int(self.frame_width * 0.1 + tecla_blanca_ancho * 3.7)
            elif nota == 'G#':
                x = int(self.frame_width * 0.1 + tecla_blanca_ancho * 4.7)
            elif nota == 'A#':
                x = int(self.frame_width * 0.1 + tecla_blanca_ancho * 5.7)
                
            self.teclas[nota] = {
                'x': x, 'y': teclado_y,
                'ancho': int(tecla_blanca_ancho * 0.6),
                'alto': int(tecla_altura * 0.6),
                'color': (0, 0, 0),  # Negro
                'color_activo': (100, 100, 100),  # Gris oscuro
                'es_negra': True
            }
    
    def dibujar_teclado(self, frame):
        """Dibuja el teclado en el frame"""
        for nota, tecla in self.teclas.items():
            color = tecla['color_activo'] if nota in self.teclas_activas else tecla['color']
            cv2.rectangle(frame, 
                         (tecla['x'], tecla['y']), 
                         (tecla['x'] + tecla['ancho'], tecla['y'] + tecla['alto']), 
                         color, -1)
            
            # Borde de la tecla
            borde_color = (0, 0, 0) if not tecla['es_negra'] else (255, 255, 255)
            cv2.rectangle(frame, 
                         (tecla['x'], tecla['y']), 
                         (tecla['x'] + tecla['ancho'], tecla['y'] + tecla['alto']), 
                         borde_color, 2)
            
            # Texto de la nota
            texto_color = (0, 0, 0) if not tecla['es_negra'] else (255, 255, 255)
            cv2.putText(frame, nota, 
                       (tecla['x'] + 5, tecla['y'] + tecla['alto'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, texto_color, 1)
    
    def detectar_toque(self, landmarks):
        """Detecta si algún dedo está tocando una tecla"""
        if not landmarks:
            return
        
        # Usar el dedo índice (landmark 8) para detectar toques
        for hand_landmarks in landmarks:
            h, w = self.frame_height, self.frame_width
            
            # Dedo índice (landmark 8)
            dedo_x = int(hand_landmarks[8].x * w)
            dedo_y = int(hand_landmarks[8].y * h)
            
            # Verificar si está tocando alguna tecla
            for nota, tecla in self.teclas.items():
                if (tecla['x'] <= dedo_x <= tecla['x'] + tecla['ancho'] and
                    tecla['y'] <= dedo_y <= tecla['y'] + tecla['alto']):
                    
                    if nota not in self.teclas_activas:
                        self.teclas_activas.add(nota)
                        # Tocar la nota en un hilo separado
                        threading.Thread(target=self.generador.tocar_nota, args=(nota,)).start()
                    break
            else:
                # Si no está tocando ninguna tecla, limpiar teclas activas
                self.teclas_activas.clear()

# ===================== FUNCIÓN PARA DIBUJAR LANDMARKS =====================
def draw_landmarks(frame, detection_result):
    if not detection_result.hand_landmarks:
        return frame
    
    for landmarks in detection_result.hand_landmarks:
        for lm in landmarks:
            h, w, _ = frame.shape
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    return frame

# ===================== PROCESAMIENTO DE VIDEO =====================
cap = cv2.VideoCapture(0)  # 0 = cámara web

# Obtener dimensiones del frame
ret, frame = cap.read()
if ret:
    frame_height, frame_width = frame.shape[:2]
    teclado = TecladoVirtual(frame_width, frame_height)
else:
    print("Error al capturar video")
    exit()

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crear objeto MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Timestamp en ms (necesario en modo VIDEO)
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Detección
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Detectar toques en el teclado
        if detection_result.hand_landmarks:
            teclado.detectar_toque(detection_result.hand_landmarks)

        # Dibujar landmarks
        frame = draw_landmarks(frame, detection_result)
        
        # Dibujar teclado
        teclado.dibujar_teclado(frame)
        
        # Agregar instrucciones
        cv2.putText(frame, "Usa tu dedo indice para tocar las teclas", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Presiona ESC para salir", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mostrar resultado
        cv2.imshow("Piano Virtual", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
