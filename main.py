import cv2
import numpy as np
import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
import threading

model_path = "Hand Tracking/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
annotated_image = None
lock = threading.Lock() 

def result_callback(result, output_image, timestamp_ms):
    global annotated_image
    print("------------------------------------------------")
    print("result: {}".format(result))
    print("------------------------------------------------")

    with lock:
        # Convert Mediapipe Image to NumPy array
        annotated_image = np.asarray(output_image.numpy_view())

        if result.hand_world_landmarks:
            for hand_landmark in result.hand_world_landmarks:
                try:
                    print(result.hand_world_landmarks)

                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmark,
                        mp_hands.HAND_CONNECTIONS
                    )
                except Exception as e:
                    print(f"Error drawing landmarks: {e}")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands = 2,
    result_callback=result_callback)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

with HandLandmarker.create_from_options(options) as landmarker:
    def capture_and_detect(): 
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to capture image.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            landmarker.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

            if annotated_image is not None:
                with lock:
                    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Hand Tracking", annotated_image_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    capture_thread = threading.Thread(target=capture_and_detect)
    capture_thread.start()

    capture_thread.join()