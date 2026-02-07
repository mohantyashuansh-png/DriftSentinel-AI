import cv2
import numpy as np

# CONFIG - USE RAW STRINGS
INPUT_VIDEO = r"C:\Users\Ashutosh\Downloads\WhatsApp Video 2026-01-27 at 20.33.34.mp4" 
OUTPUT_VIDEO = "data/coal_mine_severe.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing severe drift video from: {INPUT_VIDEO}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. HEAVY FOG (Whiteout)
    overlay = np.ones(frame.shape, dtype=np.uint8) * 255
    frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0) # 60% White
    
    # 2. HEAVY BLUR (Focus Loss)
    frame = cv2.GaussianBlur(frame, (21, 21), 0)

    # 3. SENSOR NOISE (The VAE Killer)
    # VAEs hate random noise. This will spike the error.
    noise = np.random.normal(0, 50, frame.shape).astype(np.uint8)
    frame = cv2.add(frame, noise)

    # 4. LENS OBSTRUCTION (Mud Splash)
    # Draws a black circle to simulate dirt on the lens
    h, w, _ = frame.shape
    cv2.circle(frame, (int(w*0.8), int(h*0.8)), 80, (0,0,0), -1)

    out.write(frame)

cap.release()
out.release()
print(f"✅ Done! Saved to {OUTPUT_VIDEO}")