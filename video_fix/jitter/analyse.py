import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

video_path = BASE / "data" / "ai_generated" / "ai_sign3.mp4"

cap = cv2.VideoCapture(str(video_path))
#cap = cv2.VideoCapture('/Users/ayushmishra06/Desktop/sign-language-video-fix/data/ai_generated/ai_sign3.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"fps: {fps} , width: {w} , height: {h} , count: {total}")

# analyzing the jitter by looking at frame to frame optical flow

frames = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(min(30,total)):
    ret,frame = cap.read()
    if ret:
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))

motion = []
for i in range(1,len(frames)):
    flow = cv2.calcOpticalFlowFarneback(
        frames[i-1],
        frames[i],
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )
    mag ,_ = cv2.cartToPolar(flow[...,0],flow[...,1])
    motion.append(mag.mean())

print(f"Avg motion: {np.mean(motion):.3f}  , Std : {np.std(motion):.3f}")
cap.release()
