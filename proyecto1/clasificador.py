import cv2
import numpy as np
import os
import glob

# area_min = 100
# kernel = 8 -> esto es la op morfologica de dilatacion

# preprocesamiento:
# 1. imagen a gris
# 2. filtro gaussiano -> reducir ruido
# 3. umbralizacion -> blanco y negro
# 4. morfologica (open y close)
# 5. escalado de areas

# filtrado:
# area de contornos (aca filtro los que tengan area menor a la marcada)

# Busco todas las carpetas que tengo en clases/
carpetas_referencia = {}
ruta_base = "clases"

if os.path.exists(ruta_base):
    # Reviso cada carpeta dentro de clases/
    for item in os.listdir(ruta_base):
        ruta_completa = os.path.join(ruta_base, item)
        if os.path.isdir(ruta_completa):
            # Busco imágenes en varios formatos
            formatos_imagen = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            imagenes = []
            for formato in formatos_imagen:
                patron_imagenes = os.path.join(ruta_completa, formato)
                imagenes.extend(glob.glob(patron_imagenes))
            if imagenes:  # Solo agrego si encontré imágenes
                carpetas_referencia[item] = ruta_completa
                print(f"Categoría detectada: {item} con {len(imagenes)} imágenes")
else:
    print(f"Error: No se encontró la carpeta {ruta_base}")
    # Creo carpetas de ejemplo si no existen
    carpetas_ejemplo = ["destornillador", "cuadrado", "circulo"]
    for carpeta in carpetas_ejemplo:
        ruta_carpeta = os.path.join(ruta_base, carpeta)
        if not os.path.exists(ruta_carpeta):
            os.makedirs(ruta_carpeta)
            print(f"Carpeta creada: {ruta_carpeta}")
        carpetas_referencia[carpeta] = ruta_carpeta

print(f"Total de categorías detectadas: {len(carpetas_referencia)}")

# Me aseguro de que encontré algo
if not carpetas_referencia:
    print("Error: No se encontraron carpetas con imágenes de referencia.")
    print("Asegúrate de tener carpetas con imágenes en la carpeta 'clases/'")
    print("Ejemplo: clases/destornillador/, clases/cuadrado/, clases/circulo/")
    exit(1)

contornos_ref = {}

# Cargo todas las imágenes de cada carpeta
for nombre_categoria, ruta_carpeta in carpetas_referencia.items():
    contornos_ref[nombre_categoria] = []
    
    # Busco imágenes en varios formatos
    formatos_imagen = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    imagenes = []
    for formato in formatos_imagen:
        patron_imagenes = os.path.join(ruta_carpeta, formato)
        imagenes.extend(glob.glob(patron_imagenes))
    
    if not imagenes:
        print(f"No se encontraron imágenes en la carpeta {ruta_carpeta}")
        continue
    
    print(f"Cargando {len(imagenes)} imágenes para {nombre_categoria}...")
    
    for ruta_imagen in imagenes:
        img_ref = cv2.imread(ruta_imagen, 0)
        if img_ref is None:
            print(f"No se pudo cargar la imagen {ruta_imagen}")
            continue
        
        _, ref_bin = cv2.threshold(img_ref, 127, 255, cv2.THRESH_BINARY)
        # Invierto para tener objetos negros sobre fondo blanco
        ref_bin_inv = cv2.bitwise_not(ref_bin)
        conts, _ = cv2.findContours(ref_bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if conts:
            # Me quedo con el contorno más grande
            areas = [cv2.contourArea(c) for c in conts]
            idx_max = int(np.argmax(areas))
            contornos_ref[nombre_categoria].append(conts[idx_max])
        else:
            print(f"No se encontró contorno en {ruta_imagen}")
    
    print(f"Se cargaron {len(contornos_ref[nombre_categoria])} contornos para {nombre_categoria}")

# Función de procesamiento

def escalar_contorno(contorno, area_objetivo):
    area_actual = cv2.contourArea(contorno)
    if area_actual == 0:
        return contorno.copy()
    factor = (area_objetivo / area_actual) ** 0.5
    M = cv2.moments(contorno)
    if M['m00'] == 0:
        return contorno.copy()
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    contorno_centrado = contorno - [cx, cy]
    contorno_escalado = (contorno_centrado * factor).astype(np.int32) + [cx, cy]
    return contorno_escalado

def procesar(frame):
    # ROI: 3/4 inferiores de la imagen
    h, w = frame.shape[:2]
    y0 = h // 4
    roi = frame[y0:h, :]

    # Leo los valores de los controles
    umbral = cv2.getTrackbarPos('Umbral', 'Ajustes')
    auto = cv2.getTrackbarPos('Auto', 'Ajustes')
    kernel_size = cv2.getTrackbarPos('Kernel', 'Ajustes')
    area_min = cv2.getTrackbarPos('Area min', 'Ajustes') 
    match_umbral = cv2.getTrackbarPos('Match max', 'Ajustes') / 1000.0

    # Convierto a escala de grises
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Aplico filtro gaussiano para quitar ruido
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Umbralizo la imagen
    if auto:
        _, binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binaria = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)

    # Hago operaciones morfológicas
    k = max(1, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

    # Invierto para tener objetos negros sobre fondo blanco
    binaria_inv = cv2.bitwise_not(binaria)

    # Busco contornos en el ROI
    contornos, _ = cv2.findContours(binaria_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Proceso cada contorno encontrado
    salida = frame.copy()
    # Marco el área donde busco
    cv2.rectangle(salida, (0, y0), (w, h), (255, 0, 255), 2)
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < area_min:
            continue  # Me salto los mas pequenos
        # Busco qué objeto se parece más
        area = cv2.contourArea(cnt)
        mejor_nombre = "desconocido"
        mejor_dist = float('inf')
        
        for nombre_categoria, lista_contornos in contornos_ref.items():
            # Comparo con todas las referencias de esta categoría
            for ref in lista_contornos:
                ref_escalado = escalar_contorno(ref, area)
                dist = cv2.matchShapes(cnt, ref_escalado, cv2.CONTOURS_MATCH_I1, 0.0)
                if dist < mejor_dist:
                    mejor_dist = dist
                    mejor_nombre = nombre_categoria
        if match_umbral == 0:
            continue  # No muestro nada si el umbral es 0
        if mejor_dist > match_umbral:
            color = (0, 0, 255)  # Rojo si no lo reconozco
            texto = f"desconocido A:{int(area)}"
        else:
            color = (0, 255, 0)  # Verde si lo reconozco
            texto = f"{mejor_nombre} D:{mejor_dist:.2f} A:{int(area)}"
        # Ajusto las coordenadas al frame completo
        cnt_shifted = cnt + np.array([0, y0])
        cv2.drawContours(salida, [cnt_shifted], -1, color, 2)
        # Muestro el texto con la información
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00']) + y0
            cv2.putText(salida, texto, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return binaria, salida

def preprocesar_referencia(img, umbral, auto, kernel_size):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplico el mismo filtro gaussiano
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    if auto:
        _, binaria = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binaria = cv2.threshold(img_gray, umbral, 255, cv2.THRESH_BINARY)
    k = max(1, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    binaria_inv = cv2.bitwise_not(binaria)
    contornos, _ = cv2.findContours(binaria_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return binaria, contornos

# Cargo una imagen de cada categoría para mostrar
imagenes_ref_color = {}
for nombre_categoria, ruta_carpeta in carpetas_referencia.items():
    # Busco imágenes en varios formatos
    formatos_imagen = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    imagenes = []
    for formato in formatos_imagen:
        patron_imagenes = os.path.join(ruta_carpeta, formato)
        imagenes.extend(glob.glob(patron_imagenes))
    if imagenes:
        img_color = cv2.imread(imagenes[0])  # Tomo la primera como ejemplo
        if img_color is not None:
            imagenes_ref_color[nombre_categoria] = img_color

# Ventanas y barras
cv2.namedWindow('Ajustes')
cv2.createTrackbar('Umbral', 'Ajustes', 127, 255, lambda x: None)
cv2.createTrackbar('Auto', 'Ajustes', 0, 1, lambda x: None)
cv2.createTrackbar('Kernel', 'Ajustes', 3, 20, lambda x: None)
cv2.createTrackbar('Area min', 'Ajustes', 100, 10000, lambda x: None)  # Eliminar esta línea
cv2.createTrackbar('Match max', 'Ajustes', 20, 200, lambda x: None)  # 0.02 por defecto, máximo 0.20

# Eliminar la sección que muestra las imágenes de referencia con sus contornos

# Captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception('No se pudo abrir la cámara')

print('Presiona ESC para salir')
while True:
    ret, frame = cap.read()
    if not ret:
        print('No se pudo leer el frame de la cámara')
        break
    # Leo los valores de los controles
    umbral = cv2.getTrackbarPos('Umbral', 'Ajustes')
    auto = cv2.getTrackbarPos('Auto', 'Ajustes')
    kernel_size = cv2.getTrackbarPos('Kernel', 'Ajustes')
    # Proceso el video
    binaria, salida = procesar(frame)
    cv2.imshow('Binaria', binaria)
    cv2.imshow('Salida', salida)
    # Muestro las imágenes de referencia con sus contornos
    for nombre_categoria, img_color in imagenes_ref_color.items():
        binaria_ref, contornos_ref_img = preprocesar_referencia(img_color, umbral, auto, kernel_size)
        ref_img_contornos = img_color.copy()
        area_texto = ''
        if contornos_ref_img:
            cv2.drawContours(ref_img_contornos, contornos_ref_img, -1, (0, 255, 0), 2)
            # Calculo el área del contorno más grande
            areas = [cv2.contourArea(c) for c in contornos_ref_img]
            idx_max = int(np.argmax(areas))
            area_max = areas[idx_max]
            M = cv2.moments(contornos_ref_img[idx_max])
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.putText(ref_img_contornos, f'Area: {int(area_max)}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            area_texto = f'Area: {int(area_max)}'
        cv2.putText(ref_img_contornos, nombre_categoria, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(f'Ref: {nombre_categoria}', ref_img_contornos)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
