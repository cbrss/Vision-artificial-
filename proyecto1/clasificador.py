import os
import cv2
import numpy as np
from joblib import load


def calcular_hu(cnt) -> list:
    m = cv2.moments(cnt)
    hu = cv2.HuMoments(m).flatten()
    return [float(v) for v in hu]


def procesar(frame):
    h, w = frame.shape[:2]
    y0 = h // 4
    roi = frame[y0:h, :]

    umbral = cv2.getTrackbarPos('Umbral', 'Ajustes')
    auto = cv2.getTrackbarPos('Auto', 'Ajustes')
    kernel_size = cv2.getTrackbarPos('Kernel', 'Ajustes')
    area_min = cv2.getTrackbarPos('Area min', 'Ajustes')

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if auto:
        _, binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binaria = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)

    k = max(1, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_filtrados = [c for c in contornos if cv2.contourArea(c) >= area_min]

    salida = frame.copy()
    cv2.rectangle(salida, (0, y0), (w, h), (255, 0, 255), 2)
    return binaria, salida, contornos_filtrados, y0


def main():
    base_dir = os.path.dirname(__file__)
    modelo_path = os.path.join(base_dir, 'modelo_hu.joblib')
    if not os.path.exists(modelo_path):
        raise FileNotFoundError('No se encontró el modelo. Entrénalo con entrenador.py')

    artifact = load(modelo_path)
    clf = artifact['model']
    classes = artifact['classes']

    cv2.namedWindow('Ajustes')
    cv2.createTrackbar('Umbral', 'Ajustes', 127, 255, lambda x: None)
    cv2.createTrackbar('Auto', 'Ajustes', 0, 1, lambda x: None)
    cv2.createTrackbar('Kernel', 'Ajustes', 3, 20, lambda x: None)
    cv2.createTrackbar('Area min', 'Ajustes', 100, 10000, lambda x: None)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception('No se pudo abrir la cámara')

    print('ESC para salir')
    while True:
        ok, frame = cap.read()
        if not ok:
            print('No se pudo leer el frame de la cámara')
            break

        binaria, salida, contornos_filtrados, y0 = procesar(frame)

        for cnt in contornos_filtrados:
            hu = calcular_hu(cnt)
            # Probabilidad y clase
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba([hu])[0]
                pred_idx = int(np.argmax(proba))
                conf = float(proba[pred_idx])
            else:
                pred_idx = int(clf.predict([hu])[0])
                conf = 1.0
            texto = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
            etiqueta = f"{texto} {conf*100:.0f}%"

            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00']) + y0
                cv2.putText(salida, etiqueta, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cnt_shifted = cnt + np.array([0, y0])
            cv2.drawContours(salida, [cnt_shifted], -1, (0, 255, 0), 2)

        cv2.imshow('Salida', salida)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


