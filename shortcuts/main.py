import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IMAGE_FILENAMES = ['thumbs_down.jpg', 'victory.jpg',
                   'thumbs_up.jpg', 'pointing_up.jpg']

# STEP 1: Create an GestureRecognizer object.
base_options = python.BaseOptions(
    model_asset_path='model/gesture_recognizer.task.tflite'
)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

for image_file_name in IMAGE_FILENAMES:
    # STEP 2: Load the input image.
    image = mp.Image.create_from_file(f"test/{image_file_name}")

    # STEP 3: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(image)

    # STEP 4: Process the result. In this case, print it.
    top_gesture = recognition_result.gestures[0][0]

    print(f"Input image: {image_file_name} Top gesture: {top_gesture.category_name} ({top_gesture.score})")
