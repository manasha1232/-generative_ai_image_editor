"""
Generate Synthetic Test Scene & ROI Mask for Day 30 Generative AI Image Editing Studio
Day 30 - 30-Day Computer Vision Challenge
"""

import os
import cv2
import numpy as np

def create_synthetic_editor_demo(output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "demo_input_scene.jpg")
    mask_path = os.path.join(output_dir, "demo_roi_mask.png")

    # Create 512x512 RGB studio background scene image
    h, w = 512, 512
    scene_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Gradient background
    for y in range(h):
        val = int(30 + 40 * (y / h))
        scene_img[y, :] = (val, val + 5, val + 15)

    # Draw tabletop surface
    cv2.rectangle(scene_img, (0, 340), (w, h), (80, 70, 60), -1)
    cv2.line(scene_img, (0, 340), (w, 340), (120, 110, 100), 3)

    # Original object to be edited/inpainted (A plain wooden box on tabletop: 180, 220 -> 330, 360)
    cv2.rectangle(scene_img, (180, 220), (330, 360), (40, 80, 140), -1)
    cv2.putText(scene_img, "OBJECT", (210, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(img_path, scene_img)
    print(f"[SUCCESS] Synthetic editor input scene saved: {img_path} ({w}x{h})")

    # Create Binary ROI Mask image (White circle/box where editing will happen)
    mask_img = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask_img, (160, 200), (350, 380), 255, -1) # Expanded ROI mask

    cv2.imwrite(mask_path, mask_img)
    print(f"[SUCCESS] Synthetic ROI mask saved: {mask_path} ({w}x{h})")

    return img_path, mask_path

if __name__ == "__main__":
    create_synthetic_editor_demo()
