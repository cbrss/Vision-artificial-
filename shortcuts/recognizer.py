from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def create_gesture_recognizer():
    base_options = python.BaseOptions(
        model_asset_path='model/gesture_recognizer.task.tflite'
    )
    options = vision.GestureRecognizerOptions(base_options=base_options)
    return vision.GestureRecognizer.create_from_options(options)
