"""
Task 7 — Depth Approximation from Two Images
=============================================
Member 4

Estimates relative depth from a stereo image pair using disparity.

Usage:
    1. Place stereo pairs in images/task7/ (e.g., left.jpg, right.jpg)
    2. Run: python tasks/task7_depth_approximation.py
    3. Results saved to outputs/task7/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import load_image, save_image, show_images_row, to_grayscale


def compute_disparity_bm(left_gray, right_gray, num_disparities=64, block_size=15):
    """
    Compute disparity map using StereoBM (Block Matching).
    """
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(left_gray, right_gray)
    return disparity


def compute_disparity_sgbm(left_gray, right_gray, num_disparities=64, block_size=7):
    """
    Compute disparity map using StereoSGBM (Semi-Global Block Matching).
    More accurate but slower than StereoBM.
    """
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(left_gray, right_gray)
    return disparity


def normalize_disparity(disparity):
    """Normalize disparity map to 0-255 uint8 for visualization."""
    disp_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    return disp_normalized.astype(np.uint8)


def create_depth_map(disparity, smooth=True):
    """
    Create a depth map from disparity.
    Brighter = closer, Darker = farther.
    """
    depth = normalize_disparity(disparity)

    if smooth:
        depth = cv2.medianBlur(depth, 5)
        depth = cv2.bilateralFilter(depth, 9, 75, 75)

    return depth


def create_colored_depth(depth_gray):
    """Apply a color map to the depth for better visualization."""
    return cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)


def main():
    input_dir, output_dir = os.path.join("images", "task7"), os.path.join("outputs", "task7")

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"⚠️  No images in {input_dir}/")
        print("   Add stereo image pairs:")
        print("   - Take two photos of the same scene, shifting camera slightly sideways")
        print("   - Or download from Middlebury stereo datasets")
        print("   Name them: left.jpg, right.jpg (or left1.jpg, right1.jpg, etc.)")
        return

    files = sorted([f for f in os.listdir(input_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    if len(files) < 2:
        print("  ❌ Need at least 2 images (left + right)")
        return

    for i in range(0, len(files) - 1, 2):
        f_left, f_right = files[i], files[i + 1]
        print(f"\n{'='*50}\nStereo pair: {f_left} + {f_right}\n{'='*50}")

        left = load_image(os.path.join(input_dir, f_left))
        right = load_image(os.path.join(input_dir, f_right))

        # Ensure same size
        if left.shape != right.shape:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))

        left_gray = to_grayscale(left)
        right_gray = to_grayscale(right)

        # Method 1: StereoBM
        print("  Computing disparity (StereoBM)...")
        disp_bm = compute_disparity_bm(left_gray, right_gray)
        depth_bm = create_depth_map(disp_bm)
        depth_bm_color = create_colored_depth(depth_bm)

        # Method 2: StereoSGBM
        print("  Computing disparity (StereoSGBM)...")
        disp_sgbm = compute_disparity_sgbm(left_gray, right_gray)
        depth_sgbm = create_depth_map(disp_sgbm)
        depth_sgbm_color = create_colored_depth(depth_sgbm)

        # Display results
        show_images_row(
            [left, right, depth_bm_color, depth_sgbm_color],
            ["Left", "Right", "Depth (BM)", "Depth (SGBM)"],
            figsize=(22, 5)
        )

        # Save outputs
        pair_name = f"pair{i//2 + 1}"
        save_image(depth_bm, os.path.join(output_dir, f"{pair_name}_depth_bm.png"))
        save_image(depth_bm_color, os.path.join(output_dir, f"{pair_name}_depth_bm_color.png"))
        save_image(depth_sgbm, os.path.join(output_dir, f"{pair_name}_depth_sgbm.png"))
        save_image(depth_sgbm_color, os.path.join(output_dir, f"{pair_name}_depth_sgbm_color.png"))

    print("\n✅ Task 7 complete!")


if __name__ == "__main__":
    main()
