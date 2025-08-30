import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from recognizer import create_gesture_recognizer

if __name__ == '__main__':
    recognizer = create_gesture_recognizer()

    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir BGR > RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crear objeto MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Realizar la detección de gestos
        recognition_result = recognizer.recognize(mp_image)

        top_gesture = recognition_result.gestures[0][0] if recognition_result.gestures else None

        if top_gesture:
            print(f"Top gesture: {top_gesture.category_name} ({top_gesture.score})")

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
