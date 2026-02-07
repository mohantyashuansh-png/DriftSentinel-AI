import cv2
import numpy as np
import os
from pathlib import Path

# CONFIGURATION
INPUT_DIR = "data/normal"
OUTPUT_DIR = "data/drifted"
FOG_INTENSITY = 0.5  # 0.0 is clear, 1.0 is pure white wall
BLUR_LEVEL = 15      # Higher means more blurry (must be odd number)

def add_fog(image):
    # Create a white overlay same size as image
    fog_overlay = np.ones(image.shape, dtype=np.uint8) * 255
    
    # Blend the image with the white overlay (Simulates fog)
    # The 'beta' parameter controls how much white is added
    drifted_image = cv2.addWeighted(image, 1 - FOG_INTENSITY, fog_overlay, FOG_INTENSITY, 0)
    
    # Add blur (Simulates focus loss or dust on lens)
    drifted_image = cv2.GaussianBlur(drifted_image, (BLUR_LEVEL, BLUR_LEVEL), 0)
    
    return drifted_image

def main():
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(Path(INPUT_DIR).glob(ext))
        
    print(f"Found {len(image_files)} images. Generating drift...")

    count = 0
    for img_path in image_files:
        # Read Image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Apply the "Drift" (Fog + Blur)
        drifted_img = add_fog(img)
        
        # Save to output folder
        filename = img_path.name
        save_path = os.path.join(OUTPUT_DIR, f"drifted_{filename}")
        cv2.imwrite(save_path, drifted_img)
        count += 1

    print(f"✅ Success! Generated {count} drifted images in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()