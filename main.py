import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading

model_path = "Hand Tracking/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result, output_image, timestamp_ms):
    print('hand landmarker result: {}'.format(result))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with HandLandmarker.create_from_options(options) as landmarker:
    
    # Initialize the video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not found or accessible.")
        exit()

    def capture_and_detect():
        frame_timestamp_ms = 0  # Initialize timestamp variable
        while True:
            # Capture frame from webcam
            success, frame = cap.read()
            if not success:
                print("Error: Failed to capture image.")
                break

            # Convert BGR frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a MediaPipe Image object (Ensure proper image handling)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Send live image data asynchronously for hand landmarks detection
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 1

            # Break the loop on pressing 'q'
            cv2.imshow("Test Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    # Start the capture thread
    capture_thread = threading.Thread(target=capture_and_detect)
    capture_thread.start()

    # Wait for the thread to finish
    capture_thread.join()