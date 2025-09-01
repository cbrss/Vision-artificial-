import cv2
import numpy as np

# Diccionario con nombres y rutas de referencia
referencias = {
    "dest": "img/dest.jpg",
    "cel": "img/cel.jpg",
    "cua": "img/cua.jpg",
}

def resize_to_fit(img, max_width=900, max_height=700):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def preprocesar(img, umbral, auto, kernel_size):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar filtro gaussiano para reducir ruido
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    if auto:
        _, binaria = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binaria = cv2.threshold(img_gray, umbral, 255, cv2.THRESH_BINARY)
    k = max(1, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    return binaria

for nombre, ruta in referencias.items():
    img = cv2.imread(ruta)
    if img is None:
        print(f"No se pudo cargar la referencia {nombre}")
        continue
    # Ventana y barras
    win = f'Referencia: {nombre}'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Umbral', win, 127, 255, lambda x: None)
    cv2.createTrackbar('Auto', win, 0, 1, lambda x: None)
    cv2.createTrackbar('Kernel', win, 3, 20, lambda x: None)
    print(f"Mostrando referencia: {nombre}. Ajusta los parámetros y presiona cualquier tecla para continuar...")
    while True:
        umbral = cv2.getTrackbarPos('Umbral', win)
        auto = cv2.getTrackbarPos('Auto', win)
        kernel_size = cv2.getTrackbarPos('Kernel', win)
        binaria = preprocesar(img, umbral, auto, kernel_size)
        # Invertir la imagen binaria para detectar objetos negros sobre fondo blanco
        binaria_inv = cv2.bitwise_not(binaria)
        contornos, _ = cv2.findContours(binaria_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        salida = img.copy()
        if contornos:
            cv2.drawContours(salida, contornos, -1, (0, 255, 0), 2)
            for i, cnt in enumerate(contornos):
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.circle(salida, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(salida, f'C{i+1}', (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(salida, nombre, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(salida, 'Sin contorno', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(win, resize_to_fit(salida))
        cv2.imshow(f'Binaria_{nombre}', resize_to_fit(binaria))
        key = cv2.waitKey(30)
        if key != -1:
            break
    cv2.destroyAllWindows()
