import time

import cv2
import mediapipe as mp
import recognizer
import windows as win

TIME_BETWEEN_GESTURES = 1

if __name__ == '__main__':
    recognizer = recognizer.create_gesture_recognizer()
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    timer = time.time()

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

        if top_gesture and time.time() - timer > TIME_BETWEEN_GESTURES:
            print(f"Top gesture: {top_gesture.category_name} ({top_gesture.score})")
            if (top_gesture.category_name == 'Open_Palm'):
                win.maximizeWindow()
            elif (top_gesture.category_name == 'Closed_Fist'):
                win.minimizeWindow()
            elif (top_gesture.category_name == 'Victory'):
                win.screenshotFull()
            elif (top_gesture.category_name == 'Pointing_Up'):
                win.taskView()
            elif (top_gesture.category_name == 'ILoveYou'):
                win.nextTask()
            timer = time.time()

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
