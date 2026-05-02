"""
Task 2 — Comparative Study of Sharpening Pipelines for X-Ray Images
====================================================================
Member 2

Implements three different enhancement pipelines for X-ray images
and compares their results.

Usage:
    1. Place X-ray images in images/task2/
    2. Run: python tasks/task2_xray_sharpening.py
    3. Results saved to outputs/task2/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import (
    load_image, save_image, show_images_row,
    print_analysis, compute_psnr, to_grayscale, plot_histogram
)


# ─────────────────────────────────────────────
#  Pipeline A: Histogram Equalization → Unsharp Mask → Median Filter
# ─────────────────────────────────────────────

def pipeline_a(img_gray):
    """
    Pipeline A: Global histogram equalization + Unsharp masking + Median noise reduction.
    """
    # Step 1: Histogram Equalization
    equalized = cv2.equalizeHist(img_gray)

    # Step 2: Unsharp Masking (sharpen)
    blurred = cv2.GaussianBlur(equalized, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(equalized, 1.5, blurred, -0.5, 0)

    # Step 3: Median filter for noise reduction
    result = cv2.medianBlur(sharpened, 3)

    return result


# ─────────────────────────────────────────────
#  Pipeline B: CLAHE → Laplacian Sharpening → Median Filter
# ─────────────────────────────────────────────

def pipeline_b(img_gray):
    """
    Pipeline B: CLAHE + Laplacian-based sharpening + Median noise reduction.
    """
    # Step 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    # Step 2: Laplacian sharpening
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(enhanced.astype(np.float64) - laplacian)

    # Step 3: Median filter
    result = cv2.medianBlur(sharpened, 3)

    return result


# ─────────────────────────────────────────────
#  Pipeline C: Negative → Gamma Correction → High-Boost Filter
# ─────────────────────────────────────────────

def pipeline_c(img_gray):
    """
    Pipeline C: Image negative + Gamma correction + High-boost filtering.
    """
    # Step 1: Negative transform
    negative = 255 - img_gray

    # Step 2: Gamma correction (gamma < 1 to brighten dark areas)
    gamma = 0.7
    normalized = negative / 255.0
    gamma_corrected = np.power(normalized, gamma) * 255
    gamma_corrected = gamma_corrected.astype(np.uint8)

    # Step 3: High-boost filter (identity + amplified high-pass)
    blurred = cv2.GaussianBlur(gamma_corrected, (5, 5), 0)
    high_pass = cv2.subtract(gamma_corrected, blurred)
    A = 2.0  # Amplification factor
    result = cv2.addWeighted(gamma_corrected, 1, high_pass, A, 0)

    # Invert back to positive
    result = 255 - result

    return result


# ─────────────────────────────────────────────
#  Comparison & Metrics
# ─────────────────────────────────────────────

def compute_contrast(img):
    """Compute contrast as standard deviation of pixel values."""
    return np.std(img.astype(np.float64))


def compare_pipelines(original, results, names, output_dir, base_name):
    """
    Compare pipeline results visually and with metrics.
    """
    # --- Visual comparison ---
    all_images = [original] + results
    all_titles = ["Original"] + names

    show_images_row(all_images, all_titles, figsize=(20, 5))

    # --- Metrics comparison ---
    print(f"\n{'Pipeline':<15} {'Contrast (σ)':<15} {'Mean Brightness':<18} {'PSNR vs Original':<18}")
    print("-" * 66)

    for name, result in zip(names, results):
        contrast = compute_contrast(result)
        mean_val = np.mean(result)
        psnr = compute_psnr(original, result)
        print(f"{name:<15} {contrast:<15.2f} {mean_val:<18.2f} {psnr:<18.2f} dB")

    # --- Save comparison figure ---
    fig, axes = plt.subplots(2, len(all_images), figsize=(20, 8))

    for i, (img, title) in enumerate(zip(all_images, all_titles)):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(title, fontsize=11)
        axes[0, i].axis('off')

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        axes[1, i].plot(hist, color='black')
        axes[1, i].set_title(f"{title} Histogram")
        axes[1, i].set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_comparison.png"), dpi=150)
    plt.show()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    input_dir = os.path.join("images", "task2")
    output_dir = os.path.join("outputs", "task2")

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"⚠️  No images found in {input_dir}/")
        print("   Please add X-ray images. Sources:")
        print("   - Kaggle: 'Chest X-Ray Images (Pneumonia)'")
        print("   - NIH Chest X-ray dataset")
        print("   - Search 'X-ray image dataset' online")
        return

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue

        filepath = os.path.join(input_dir, filename)
        print(f"\n{'='*50}")
        print(f"Processing: {filename}")
        print(f"{'='*50}")

        img = load_image(filepath)
        gray = to_grayscale(img)
        print_analysis(gray, filename)

        # Run all three pipelines
        result_a = pipeline_a(gray)
        result_b = pipeline_b(gray)
        result_c = pipeline_c(gray)

        results = [result_a, result_b, result_c]
        names = ["Pipeline A", "Pipeline B", "Pipeline C"]

        # Compare
        base_name = os.path.splitext(filename)[0]
        compare_pipelines(gray, results, names, output_dir, base_name)

        # Save individual results
        for name_short, result in zip(["pipeA", "pipeB", "pipeC"], results):
            save_image(result, os.path.join(output_dir, f"{base_name}_{name_short}.png"))

    print("\n✅ Task 2 complete!")


if __name__ == "__main__":
    main()
