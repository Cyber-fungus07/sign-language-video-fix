import cv2
import numpy as np
from tqdm import tqdm
from dwpose import Wholebody, DwposeDetector
from gfpgan import GFPGANer


VIDEO_PATH = "/Users/ayushmishra06/Desktop/sign-language-video-fix/data/ai_generated/ai_sign1.mp4"
OUTPUT_PATH = "/Users/ayushmishra06/Desktop/sign-language-video-fix/data/outputs/fully_restored1.mp4"
MODEL_PATH = "/Users/ayushmishra06/Desktop/sign-language-video-fix/models/GFPGANv1.4.pth"

print("Loading Models (DWPose & GFPGAN)...")
pose_model = Wholebody(
    det_model_path="/Users/ayushmishra06/Desktop/sign-language-video-fix/models/yolox_l.onnx",
    pose_model_path="/Users/ayushmishra06/Desktop/sign-language-video-fix/models/dw-ll_ucoco_384.onnx"
)
detector = DwposeDetector(pose_model)

restorer = GFPGANer(
    model_path=MODEL_PATH,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)


# --- Helper Functions ---
def create_kalman(init_x, init_y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]
    ], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.statePre = np.array([[init_x], [init_y], [0], [0]], dtype=np.float32)
    kf.statePost = np.array([[init_x], [init_y], [0], [0]], dtype=np.float32)
    return kf


def extract_structured_keypoints(pose_result):
    data = {'body': [], 'face': [], 'hands': []}
    if hasattr(pose_result, 'body') and pose_result.body is not None:
        pts = pose_result.body.keypoints if hasattr(pose_result.body, 'keypoints') else pose_result.body
        data['body'] = pts.tolist() if isinstance(pts, np.ndarray) else pts
    if hasattr(pose_result, 'face') and pose_result.face is not None:
        pts = pose_result.face.keypoints if hasattr(pose_result.face, 'keypoints') else pose_result.face
        data['face'] = pts.tolist() if isinstance(pts, np.ndarray) else pts
    if hasattr(pose_result, 'hands') and pose_result.hands is not None:
        pts = pose_result.hands
        if isinstance(pts, np.ndarray):
            data['hands'] = pts.reshape(-1, pts.shape[-1]).tolist()

    for key in data:
        data[key] = [p if (p is not None and len(p) >= 2 and (p[0] != 0 or p[1] != 0)) else None for p in data[key]]
    return data


def get_bbox_from_points(points, frame_w, frame_h, padding=40):
    valid_points = [p for p in points if p is not None]
    if not valid_points: return None
    xs, ys = [p[0] for p in valid_points], [p[1] for p in valid_points]
    x1, y1 = max(0, int(min(xs) - padding)), max(0, int(min(ys) - padding))
    x2, y2 = min(frame_w, int(max(xs) + padding)), min(frame_h, int(max(ys) + padding))
    return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None


# --- Video Setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (512, 288))

frame_id = 0
kalman_filters = {'body': [], 'face': [], 'hands': []}
latest_keypoints = {'body': [], 'face': [], 'hands': []}

for _ in tqdm(range(total_frames), desc="Stabilizing & Restoring"):
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (512, 288))
    canvas = frame.copy()
    new_measurement = False

    if frame_id % 2 == 0:
        poses = detector.detect_poses(frame)
        if poses is not None and len(poses) > 0:
            latest_keypoints = extract_structured_keypoints(poses[0])
            new_measurement = True

    frame_id += 1
    smoothed_current_frame = {'body': [], 'face': [], 'hands': []}

    # Kalman Smoothing
    for part in ['body', 'face', 'hands']:
        pts = latest_keypoints.get(part, [])
        while len(kalman_filters[part]) < len(pts): kalman_filters[part].append(None)

        for i, kp in enumerate(pts):
            if kp is None or len(kp) < 2:
                smoothed_current_frame[part].append(None)
                continue

            if kalman_filters[part][i] is None:
                kalman_filters[part][i] = create_kalman(kp[0], kp[1])

            kf = kalman_filters[part][i]

            if new_measurement:
                x, y = kp[:2]
                pred_x, pred_y = kf.statePre[0, 0], kf.statePre[1, 0]
                if np.sqrt((x - pred_x) ** 2 + (y - pred_y) ** 2) < 40.0:
                    kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))

            prediction = kf.predict()
            smoothed_current_frame[part].append((int(prediction[0, 0]), int(prediction[1, 0])))


    face_bbox = get_bbox_from_points(smoothed_current_frame['face'], 512, 288, padding=40)

    if face_bbox:
        fx1, fy1, fx2, fy2 = face_bbox
        face_crop = canvas[fy1:fy2, fx1:fx2].copy()

        _, _, restored_crop = restorer.enhance(
            face_crop,
            has_aligned=False,
            only_center_face=True,
            paste_back=True
        )

        if restored_crop is not None:
            canvas[fy1:fy2, fx1:fx2] = restored_crop

    out.write(canvas)

cap.release()
out.release()
print(f"Done! Fully stabilized and restored video saved to:\n{OUTPUT_PATH}")