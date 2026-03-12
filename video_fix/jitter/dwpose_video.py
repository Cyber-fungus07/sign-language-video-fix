import cv2
import numpy as np
from dwpose import Wholebody, DwposeDetector

VIDEO_PATH = "/Users/ayushmishra06/Desktop/sign-language-video-fix/data/ai_generated/ai_sign2.mp4"
OUTPUT_PATH = "/Users/ayushmishra06/Desktop/sign-language-video-fix/data/ai_generated/smoothed_tracking2.mp4"

print("Loading DWPose models...")
pose_model = Wholebody(
    det_model_path="/Users/ayushmishra06/Desktop/sign-language-video-fix/models/yolox_l.onnx",
    pose_model_path="/Users/ayushmishra06/Desktop/sign-language-video-fix/models/dw-ll_ucoco_384.onnx"
)
detector = DwposeDetector(pose_model)


def create_kalman(init_x, init_y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

    kf.statePre = np.array([[init_x], [init_y], [0], [0]], dtype=np.float32)
    kf.statePost = np.array([[init_x], [init_y], [0], [0]], dtype=np.float32)
    return kf

def extract_all_keypoints(pose_result):
    points = []

    # 1. Body & Face
    for part in ['body', 'face']:
        if hasattr(pose_result, part) and getattr(pose_result, part) is not None:
            data = getattr(pose_result, part)
            if hasattr(data, 'keypoints'):
                points.extend(data.keypoints)
            else:
                points.extend(data)

    # 2. Hands (DWPose usually returns shape (2, 21, 2) for L/R hands)
    if hasattr(pose_result, 'hands') and pose_result.hands is not None:
        hands_data = pose_result.hands
        if isinstance(hands_data, np.ndarray):
            # Flatten the 3D array into a flat list of (x,y) coordinates
            hands_flat = hands_data.reshape(-1, hands_data.shape[-1])
            points.extend(hands_flat.tolist())

    # 3. Clean up invalid points (replace [0,0] with None so Kalman ignores them)
    valid_points = []
    for p in points:
        if p is not None and len(p) >= 2 and (p[0] != 0 or p[1] != 0):
            valid_points.append(p)
        else:
            valid_points.append(None)

    return valid_points

cap = cv2.VideoCapture(VIDEO_PATH)

# Get original FPS
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (512, 288))

frame_id = 0
kalman_filters = []
latest_keypoints = []

print("Processing video... This will run in the background. Please wait.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (512, 288))
    canvas = frame.copy()

    new_measurement = False

    if frame_id % 2 == 0:
        poses = detector.detect_poses(frame)
        if poses is not None and len(poses) > 0:
            latest_keypoints = extract_all_keypoints(poses[0])
            new_measurement = True

    frame_id += 1

    if not latest_keypoints:
        out.write(canvas)
        continue

    while len(kalman_filters) < len(latest_keypoints):
        kalman_filters.append(None)  # Add an empty slot

    # Update and predict
    for i, kp in enumerate(latest_keypoints):
        if kp is None or len(kp) < 2:
            continue

        if kalman_filters[i] is None:
            kalman_filters[i] = create_kalman(kp[0], kp[1])

        kf = kalman_filters[i]

        if new_measurement:
            x, y = kp[:2]
            pred_x, pred_y = kf.statePre[0, 0], kf.statePre[1, 0]
            dist = np.sqrt((x - pred_x) ** 2 + (y - pred_y) ** 2)

            # Anti-Floating fix (Face & Hands)
            if dist < 40.0:
                measurement = np.array([[np.float32(x)], [np.float32(y)]])
                kf.correct(measurement)

        # Predict smooth motion
        prediction = kf.predict()
        px, py = int(prediction[0, 0]), int(prediction[1, 0])

        # Draw the tracking points (Green dots)
        cv2.circle(canvas, (px, py), 2, (0, 255, 0), -1)

    out.write(canvas)

cap.release()
out.release()
print(f"Done! Smooth tracking video saved to:\n{OUTPUT_PATH}")