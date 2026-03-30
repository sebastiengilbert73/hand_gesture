import cv2
import mediapipe as mp
import subprocess
import sys
import os
import urllib.request
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

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

class HandTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Dashboard")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 1. Download/Initialize Model
        self.init_mediapipe()

        # 2. UI Layout
        self.top_frame = ttk.Frame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Task Selection
        ttk.Label(self.top_frame, text="Task:").pack(side=tk.LEFT, padx=(0, 5))
        self.task_var = tk.StringVar()
        self.task_cb = ttk.Combobox(self.top_frame, textvariable=self.task_var, state="readonly")
        self.task_cb['values'] = ("Track hands", "Magic wand", "Facial landmarks")
        self.task_cb.current(0)
        self.task_cb.pack(side=tk.LEFT, padx=5)

        # Camera Selection
        ttk.Label(self.top_frame, text="Camera:").pack(side=tk.LEFT, padx=(20, 5))
        self.camera_var = tk.StringVar()
        self.camera_cb = ttk.Combobox(self.top_frame, textvariable=self.camera_var, state="readonly", width=30)
        self.camera_cb.pack(side=tk.LEFT, padx=5)
        self.camera_cb.bind("<<ComboboxSelected>>", self.on_camera_select)
        
        # Quit Button
        self.quit_btn = ttk.Button(self.top_frame, text="Quit", command=self.on_close)
        self.quit_btn.pack(side=tk.RIGHT, padx=5)

        # Video Canvas
        self.canvas = tk.Canvas(root, width=640, height=480, bg="black")
        self.canvas.pack(padx=10, pady=(0, 10))

        # 3. Detect Cameras
        self.detect_cameras()

        self.cap = None
        self.is_running = True
        
        # Hand Connections constant
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4), # thumb 
            (0, 5), (5, 6), (6, 7), (7, 8), # index finger
            (5, 9), (9, 10), (10, 11), (11, 12), # middle finger
            (9, 13), (13, 14), (14, 15), (15, 16), # ring finger
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # pinky finger
        ]

        # Open the initial camera if available
        if self.available_indices:
            self.camera_cb.current(0)
            self.open_camera(self.available_indices[0])

        # Start Video loop
        self.delay = 15 # ms
        self.update_frame()

    def init_mediapipe(self):
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading MediaPipe hand landmarker model...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
        
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Download Face Landmarker Model
        face_model_path = 'face_landmarker.task'
        if not os.path.exists(face_model_path):
            print("Downloading MediaPipe face landmarker model...")
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            urllib.request.urlretrieve(url, face_model_path)
            
        face_base_options = python.BaseOptions(model_asset_path=face_model_path)
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            running_mode=vision.RunningMode.IMAGE)
        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)

    def detect_cameras(self):
        print("Detecting webcams...")
        camera_names = get_camera_names()
        self.available_indices = []
        cb_values = []

        for i in range(3): # Check first 3 indices to speed up load time
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.available_indices.append(i)
                name = camera_names[i] if i < len(camera_names) else f"Generic Interface [{i}]"
                cb_values.append(f"[{i}] {name}")
                cap.release()

        if not self.available_indices:
            print("No webcams found!")
            cb_values = ["No Cameras Found"]
        
        self.camera_cb['values'] = cb_values

    def open_camera(self, idx):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

    def on_camera_select(self, event):
        selection_idx = self.camera_cb.current()
        if selection_idx >= 0 and selection_idx < len(self.available_indices):
            self.open_camera(self.available_indices[selection_idx])

    def update_frame(self):
        if self.is_running and self.cap is not None and self.cap.isOpened():
            success, image = self.cap.read()
            if success:
                # Get selected task in case we branch later
                current_task = self.task_var.get()

                # Convert the BGR image to RGB format
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                texts_to_draw = []
                
                if current_task == "Track hands":
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    detection_result = self.detector.detect(mp_image)
                    
                    if detection_result.hand_landmarks:
                        h, w, c = image.shape
                        for hand_landmarks in detection_result.hand_landmarks:
                            # Draw connections
                            for connection in self.HAND_CONNECTIONS:
                                start_idx, end_idx = connection
                                start = hand_landmarks[start_idx]
                                end = hand_landmarks[end_idx]
                                x1, y1 = int(start.x * w), int(start.y * h)
                                x2, y2 = int(end.x * w), int(end.y * h)
                                cv2.line(image_rgb, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            
                            # Draw joints
                            for landmark in hand_landmarks:
                                x, y = int(landmark.x * w), int(landmark.y * h)
                                cv2.circle(image_rgb, (x, y), 5, (255, 0, 0), -1) # Red in RGB
                
                elif current_task == "Magic wand":
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    detection_result = self.detector.detect(mp_image)
                    
                    if detection_result.hand_landmarks:
                        h, w, c = image.shape
                        
                        # Hand tracking is required for the magic wand: draw hand structure
                        for hand_landmarks in detection_result.hand_landmarks:
                            # Draw connections
                            for connection in self.HAND_CONNECTIONS:
                                start_idx, end_idx = connection
                                start = hand_landmarks[start_idx]
                                end = hand_landmarks[end_idx]
                                x1, y1 = int(start.x * w), int(start.y * h)
                                x2, y2 = int(end.x * w), int(end.y * h)
                                cv2.line(image_rgb, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            # Draw joints
                            for landmark in hand_landmarks:
                                x, y = int(landmark.x * w), int(landmark.y * h)
                                cv2.circle(image_rgb, (x, y), 5, (255, 0, 0), -1)
                        
                        # We use the depth (z) of the first hand's palm (landmark 9) as the wand's z-depth
                        base_z = detection_result.hand_landmarks[0][9].z
                        
                        # Create a spatial mask strictly constraining detection to areas physically near the hands
                        hand_mask = np.zeros((h, w), dtype=np.uint8)
                        for hand_landmarks in detection_result.hand_landmarks:
                            pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks], np.int32)
                            hx, hy, hw, hh = cv2.boundingRect(pts)
                            # Pad the bounding box heavily to catch the length of a wand sticking out
                            padding = 150
                            cv2.rectangle(hand_mask, 
                                          (max(0, hx - padding), max(0, hy - padding)), 
                                          (min(w, hx + hw + padding), min(h, hy + hh + padding)), 
                                          255, -1)
                        
                        # Find the intensely colored "blue pen" acting as the wand via HSV space
                        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                        
                        # Generalized Blue bounds in OpenCV hue space [100-140]
                        lower_blue = np.array([100, 150, 50])
                        upper_blue = np.array([140, 255, 255])
                        
                        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
                        # We still use the hand_mask to loosely constrain to proximity 
                        constrained_mask = cv2.bitwise_and(blue_mask, hand_mask)
                        
                        contours, _ = cv2.findContours(constrained_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            best_contour = None
                            min_dist = float('inf')
                            
                            # We want to find the contour closest to the thumb tip (must be in contact)
                            # Get thumb of the primary hand (landmark 4 = tip, landmark 2 = mcp)
                            primary_hand = detection_result.hand_landmarks[0]
                            thumb_tip = primary_hand[4]
                            thumb_mcp = primary_hand[2]
                            thumb_pt = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                            
                            # Calculate the thumb overall pointing direction vector
                            tx_dir = thumb_tip.x - thumb_mcp.x
                            ty_dir = thumb_tip.y - thumb_mcp.y
                            thumb_vec = np.array([tx_dir * w, ty_dir * h], dtype=np.float64)
                            thumb_norm = np.linalg.norm(thumb_vec)
                            if thumb_norm > 0:
                                thumb_vec = thumb_vec / thumb_norm
                            
                            for c_idx in contours:
                                if cv2.contourArea(c_idx) < 100:
                                    continue
                                
                                # Distance from thumb tip to contour (returns positive if inside, negative if outside)
                                dist = cv2.pointPolygonTest(c_idx, thumb_pt, True)
                                # We want distance to be minimal if outside (pointPolygonTest returns distance value)
                                abs_dist = abs(dist) if dist < 0 else 0 
                                
                                # Must be touching or extremely close to the thumb (within ~60 pixels)
                                if abs_dist < 60 and abs_dist < min_dist:
                                    best_contour = c_idx
                                    min_dist = abs_dist
                                    
                            if best_contour is not None:
                                # Calculate strict minimum area rotated rectangle guaranteeing edge alignment
                                rect = cv2.minAreaRect(best_contour)
                                box = cv2.boxPoints(rect)
                                box = np.int32(box)
                                
                                cv2.drawContours(image_rgb, [box], 0, (0, 255, 0), 2)
                                
                                # Geometry to measure the ends - checking side distances to find the "short" side midpoints
                                dist_01 = np.linalg.norm(box[0] - box[1])
                                dist_12 = np.linalg.norm(box[1] - box[2])
                                
                                if dist_01 > dist_12:
                                    end1 = (box[1] + box[2]) // 2
                                    end2 = (box[0] + box[3]) // 2
                                else:
                                    end1 = (box[0] + box[1]) // 2
                                    end2 = (box[2] + box[3]) // 2
                                    
                                # Determine which end points in the direction of the thumb
                                wand_vec = np.array([end2[0] - end1[0], end2[1] - end1[1]], dtype=np.float64)
                                norm1 = np.linalg.norm(wand_vec)
                                if norm1 > 0:
                                    wand_vec = wand_vec / norm1
                                # dot product measures alignment
                                dot_alignment = np.dot(wand_vec, thumb_vec)
                                
                                # Depending on alignment, label one as the "Tip" and one as the "Base"
                                if dot_alignment < 0:
                                    # swap so end2 is always the one the thumb is pointing toward
                                    end1, end2 = end2, end1
                                
                                # Render the explicit endpoints cleanly
                                cv2.circle(image_rgb, tuple(end1), 6, (0, 0, 255), -1) # Base is Red
                                cv2.circle(image_rgb, tuple(end2), 6, (0, 255, 255), -1) # Tip is Cyan
                                
                                # Export endpoints to global 3D coordinates
                                e1_x, e1_y, e1_z = end1[0] / w, end1[1] / h, base_z
                                e2_x, e2_y, e2_z = end2[0] / w, end2[1] / h, base_z
                                
                                # Prepare endpoint coordinates to be drawn AFTER flipping rendering
                                t1 = f"Base: ({e1_x:.2f}, {e1_y:.2f}, {e1_z:.2f})"
                                t2 = f"Tip: ({e2_x:.2f}, {e2_y:.2f}, {e2_z:.2f})"
                                texts_to_draw.append((t1, (w - int(end1[0]) + 10, int(end1[1]) - 15), (255, 100, 100)))
                                texts_to_draw.append((t2, (w - int(end2[0]) + 10, int(end2[1]) - 15), (0, 255, 255)))
                
                elif current_task == "Facial landmarks":
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    detection_result = self.face_detector.detect(mp_image)
                    
                    if detection_result.face_landmarks:
                        from mediapipe.tasks.python import vision
                        tesselation = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
                        h, w, c = image_rgb.shape
                        for face_landmark in detection_result.face_landmarks:
                            # Parse out 3D coordinates locally
                            # First, physically map mesh connection layout
                            if tesselation is not None:
                                for connection in tesselation:
                                    start_idx = connection.start if hasattr(connection, 'start') else connection[0]
                                    end_idx = connection.end if hasattr(connection, 'end') else connection[1]
                                    
                                    start_pt = face_landmark[start_idx]
                                    end_pt = face_landmark[end_idx]
                                    
                                    x1, y1 = int(start_pt.x * w), int(start_pt.y * h)
                                    x2, y2 = int(end_pt.x * w), int(end_pt.y * h)
                                    # Very thin lightweight wireframe line
                                    cv2.line(image_rgb, (x1, y1), (x2, y2), (255, 255, 255), 1)
                            
                            # Next drop minimal nodes mapping intersection points accurately
                            for lm in face_landmark:
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(image_rgb, (cx, cy), 1, (0, 255, 0), -1)

                # Flip image horizontally for a selfie-view display
                image_rgb = cv2.flip(image_rgb, 1)

                # Draw any text overlays seamlessly after the flip onto accurate layout coordinates
                for txt, pos, color in texts_to_draw:
                    cv2.putText(image_rgb, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Convert to PIL Image and then ImageTk to display on Canvas
                pil_image = Image.fromarray(image_rgb)
                self.photo = ImageTk.PhotoImage(image=pil_image)
                
                # Update canvas safely
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        if self.is_running:
            self.root.after(self.delay, self.update_frame)

    def on_close(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        self.detector.close()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = HandTrackerApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
