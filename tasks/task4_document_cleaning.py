"""
Task 4 — Document Cleaning System
==================================
Member 1

Cleans scanned documents affected by noise, shadows, uneven lighting.
Reference: https://medium.com/data-science/denoising-noisy-documents-6807c34730c4

Usage:
    1. Place scanned document images in images/task4/
    2. Run: python tasks/task4_document_cleaning.py
    3. Results saved to outputs/task4/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from cv_utils import load_image, save_image, show_images_row, print_analysis, to_grayscale


def remove_shadows(gray):
    """Remove shadows by dividing by a blurred background estimate."""
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    # Normalize: divide original by background, scale to [0, 255]
    shadow_free = cv2.divide(gray, bg, scale=255)
    return shadow_free


def adaptive_binarize(gray, method="gaussian", block_size=15, C=10):
    """Apply adaptive thresholding for binarization."""
    if method == "gaussian":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
    else:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, C
        )


def morphological_cleanup(binary, kernel_size=2):
    """Clean noise using morphological opening then closing."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # remove small noise
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)  # fill small gaps
    return cleaned


def denoise(gray, strength=10):
    """Apply non-local means denoising."""
    return cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def clean_document(img):
    """
    Full document cleaning pipeline.
    Returns intermediate and final results for visualization.
    """
    gray = to_grayscale(img)
    
    # Step 1: Denoise
    denoised = denoise(gray, strength=10)
    
    # Step 2: Shadow removal
    shadow_free = remove_shadows(denoised)
    
    # Step 3: Adaptive binarization
    binary = adaptive_binarize(shadow_free, method="gaussian", block_size=15, C=10)
    
    # Step 4: Morphological cleanup
    cleaned = morphological_cleanup(binary, kernel_size=2)
    
    return {
        "gray": gray,
        "denoised": denoised,
        "shadow_free": shadow_free,
        "binary": binary,
        "cleaned": cleaned
    }


def main():
    input_dir, output_dir = os.path.join("images", "task4"), os.path.join("outputs", "task4")

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"⚠️  No images in {input_dir}/")
        print("   Download from the Medium article reference:")
        print("   https://medium.com/data-science/denoising-noisy-documents-6807c34730c4")
        return

    for fn in sorted(os.listdir(input_dir)):
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue
        print(f"\n{'='*50}\nProcessing: {fn}\n{'='*50}")
        img = load_image(os.path.join(input_dir, fn))
        print_analysis(img, fn)

        steps = clean_document(img)

        # Show pipeline steps
        show_images_row(
            [steps["gray"], steps["denoised"], steps["shadow_free"], steps["binary"], steps["cleaned"]],
            ["Grayscale", "Denoised", "Shadow Removed", "Binarized", "Final Cleaned"],
            figsize=(25, 5)
        )

        # Save results
        name = os.path.splitext(fn)[0]
        for key, result in steps.items():
            save_image(result, os.path.join(output_dir, f"{name}_{key}.png"))

    print("\n✅ Task 4 complete!")


if __name__ == "__main__":
    main()
