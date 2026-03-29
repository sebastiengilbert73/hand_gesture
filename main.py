import cv2
import mediapipe as mp
import subprocess
import sys
import os
import urllib.request

def get_camera_names():
    """Attempt to use PowerShell to get webcam names on Windows."""
    if sys.platform != 'win32':
        return []
    try:
        cmd = ['powershell', '-Command', "Get-PnpDevice -PresentOnly | Where-Object { $_.PNPClass -in 'Camera','Image' } | Select-Object -ExpandProperty FriendlyName"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        names = result.stdout.strip().split('\n')
        names = [n.strip() for n in names if n.strip()]
        return names
    except Exception as e:
        print(f"Could not retrieve hardware names: {e}")
        return []

def main():
    print("Detecting webcams...")
    camera_names = get_camera_names()
    
    # We will probe OpenCV to see which indices actually open
    available_indices = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # Fast initialization on Windows
        if cap.isOpened():
            available_indices.append(i)
            cap.release()
            
    if not available_indices:
        print("No webcams found!")
        return
        
    print("\nAvailable cameras:")
    for i, idx in enumerate(available_indices):
        name = camera_names[i] if i < len(camera_names) else f"Generic Camera Index {idx}"
        print(f"[{idx}] {name}")
        
    if len(available_indices) == 1:
        cam_idx = available_indices[0]
        print(f"\nAutomatically selecting the only available camera: [{cam_idx}]")
    else:
        while True:
            try:
                cam_idx = int(input("\nEnter the index of the camera you want to use: "))
                if cam_idx in available_indices:
                    break
                else:
                    print("Invalid index. Please choose one from the list above.")
            except ValueError:
                print("Please enter a valid number.")

    print(f"Opening camera {cam_idx}...")
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    
    mp_hands = None # Removed solutions API dependency
    
    # Download the hand landmarker model if it doesn't exist
    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        print("Downloading MediaPipe hand landmarker model...")
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, model_path)
    
    # Initialize MediaPipe Tasks Vision HandLandmarker
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE)
    
    detector = vision.HandLandmarker.create_from_options(options)
    
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), # thumb
        (0, 5), (5, 6), (6, 7), (7, 8), # index finger
        (5, 9), (9, 10), (10, 11), (11, 12), # middle finger
        (9, 13), (13, 14), (14, 15), (15, 16), # ring finger
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # pinky finger
    ]
    
    print("Camera opened. Position your hand in the frame. Press 'q' on the window to quit.")
    with detector:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
                
            # Convert the BGR image to RGB format for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detect hand landmarks
            detection_result = detector.detect(mp_image)
            
            # Draw the hand annotations on the image
            if detection_result.hand_landmarks:
                h, w, c = image.shape
                for hand_landmarks in detection_result.hand_landmarks:
                    # Draw connections
                    for connection in HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        start = hand_landmarks[start_idx]
                        end = hand_landmarks[end_idx]
                        x1, y1 = int(start.x * w), int(start.y * h)
                        x2, y2 = int(end.x * w), int(end.y * h)
                        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    
                    # Draw joints
                    for landmark in hand_landmarks:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            
            # Flip the image horizontally for a selfie-view display
            cv2.imshow('Hand Gesture Detection', cv2.flip(image, 1))
            
            # Press 'q' to exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
