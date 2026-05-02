"""
Task 1 — Selective Object Enhancement (Color-Based Editing Tool)
================================================================
Member 1

Enhances a specific object by color while applying different effects
to the rest of the image.

Example: Sharpen red roses while blurring yellow flowers.

Usage:
    1. Place input images in images/task1/
    2. Run: python tasks/task1_selective_enhancement.py
    3. Results saved to outputs/task1/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from cv_utils import (
    load_image, save_image, show_image, show_images_row,
    show_before_after, print_analysis
)


# ─────────────────────────────────────────────
#  Core Functions
# ─────────────────────────────────────────────

def create_color_mask(img, lower_hsv, upper_hsv):
    """
    Create a binary mask for pixels within a given HSV color range.

    Parameters
    ----------
    img : numpy.ndarray
        Input BGR image.
    lower_hsv : tuple
        Lower bound (H, S, V).
    upper_hsv : tuple
        Upper bound (H, S, V).

    Returns
    -------
    numpy.ndarray
        Binary mask (255 for matching pixels, 0 otherwise).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def create_red_mask(img):
    """
    Create a mask specifically for red objects.
    Red wraps around in HSV, so we need two ranges.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red has two ranges in HSV (wraps around 180)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def sharpen_image(img, strength=1.5):
    """
    Apply unsharp mask sharpening.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    strength : float
        Sharpening strength (1.0 = mild, 2.0 = strong).

    Returns
    -------
    numpy.ndarray
        Sharpened image.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def selective_enhance(img, mask, enhance_sharp=True, blur_ksize=21):
    """
    Enhance (sharpen) pixels inside the mask and blur the rest.

    Parameters
    ----------
    img : numpy.ndarray
        Input BGR image.
    mask : numpy.ndarray
        Binary mask (255 = region to sharpen).
    enhance_sharp : bool
        If True, sharpen the masked region. If False, sharpen the rest.
    blur_ksize : int
        Kernel size for Gaussian blur on background.

    Returns
    -------
    numpy.ndarray
        Selectively enhanced image.
    """
    # Sharpen the target region
    sharpened = sharpen_image(img, strength=1.5)

    # Blur the background
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    # Create 3-channel mask for blending
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

    if enhance_sharp:
        # Sharpen inside mask, blur outside
        result = (sharpened * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
    else:
        # Blur inside mask, sharpen outside
        result = (blurred * mask_3ch + sharpened * (1 - mask_3ch)).astype(np.uint8)

    return result


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    input_dir = os.path.join("images", "task1")
    output_dir = os.path.join("outputs", "task1")

    # Check for input images
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"⚠️  No images found in {input_dir}/")
        print("   Please add images (e.g., flowers with red and yellow colors)")
        print("   Example sources:")
        print("   - Search 'red roses yellow flowers' on Unsplash/Pexels")
        print("   - Any image with distinct colored objects")
        return

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue

        filepath = os.path.join(input_dir, filename)
        print(f"\n{'='*50}")
        print(f"Processing: {filename}")
        print(f"{'='*50}")

        img = load_image(filepath)
        print_analysis(img, filename)

        # --- Example: Enhance red objects, blur the rest ---
        red_mask = create_red_mask(img)
        result_red = selective_enhance(img, red_mask, enhance_sharp=True, blur_ksize=21)

        # --- Example: Enhance yellow objects instead ---
        yellow_mask = create_color_mask(img, (20, 70, 50), (35, 255, 255))
        result_yellow = selective_enhance(img, yellow_mask, enhance_sharp=True, blur_ksize=21)

        # Display results
        show_images_row(
            [img, red_mask, result_red],
            ["Original", "Red Mask", "Red Enhanced / Rest Blurred"],
            figsize=(18, 6)
        )

        show_images_row(
            [img, yellow_mask, result_yellow],
            ["Original", "Yellow Mask", "Yellow Enhanced / Rest Blurred"],
            figsize=(18, 6)
        )

        # Save outputs
        name = os.path.splitext(filename)[0]
        save_image(red_mask, os.path.join(output_dir, f"{name}_red_mask.png"))
        save_image(result_red, os.path.join(output_dir, f"{name}_red_enhanced.png"))
        save_image(result_yellow, os.path.join(output_dir, f"{name}_yellow_enhanced.png"))

    print("\n✅ Task 1 complete!")


if __name__ == "__main__":
    main()
