from tqdm import tqdm
import cv2
from gfpgan import GFPGANer

VIDEO_PATH = "/Users/ayushmishra06/Desktop/sign-language-video-fix/data/ai_generated/ai_sign3.mp4"
OUTPUT_PATH = "/Users/ayushmishra06/Desktop/sign-language-video-fix/data/outputs/restored_face3.mp4"
MODEL_PATH = "/Users/ayushmishra06/Desktop/sign-language-video-fix/models/GFPGANv1.4.pth"

print("Loading GFPGAN model...")
restorer = GFPGANer(
    model_path=MODEL_PATH,
    upscale=1,           # Keep the video at its original resolution
    arch='clean',        # The v1.4 architecture
    channel_multiplier=2,
    bg_upsampler=None    # Set to None because we ONLY want to process the face
)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"Restoring faces across {total_frames} frames... This will take a moment.")

for _ in tqdm(range(total_frames), desc="Processing Video"):
    ret, frame = cap.read()
    if not ret:
        break

    # The enhance function detects the face, restores it, and pastes it back seamlessly
    _, _, restored_frame = restorer.enhance(
        frame,
        has_aligned=False,
        only_center_face=True,
        paste_back=True
    )

    # # If GFPGAN fails to find a face, it just returns the original frame
    if restored_frame is not None:
        restored_frame = cv2.resize(restored_frame, (width, height))
        out.write(restored_frame)
    else:
        out.write(frame)

cap.release()
out.release()
print(f"Done! Face restored video saved to:\n{OUTPUT_PATH}")