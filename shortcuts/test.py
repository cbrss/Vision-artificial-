import mediapipe as mp
import recognizer

IMAGE_FILENAMES = ['thumbs_down.jpg', 'victory.jpg',
                   'thumbs_up.jpg', 'pointing_up.jpg']

# STEP 1: Create an GestureRecognizer object.
recognizer = recognizer.create_gesture_recognizer()

for image_file_name in IMAGE_FILENAMES:
    # STEP 2: Load the input image.
    image = mp.Image.create_from_file(f"testing_assets/{image_file_name}")

    # STEP 3: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(image)

    # STEP 4: Process the result. In this case, print it.
    top_gesture = recognition_result.gestures[0][0]

    print(f"Input image: {image_file_name} Top gesture: {top_gesture.category_name} ({top_gesture.score})")
