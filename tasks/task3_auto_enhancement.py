"""
Task 3 — Intelligent Auto Image Enhancement System
===================================================
Member 2

Auto-analyzes an image (dark/bright/low-contrast) and applies
the best enhancement strategy.

Usage:
    1. Place images in images/task3/
    2. Run: python tasks/task3_auto_enhancement.py
    3. Results saved to outputs/task3/
"""

import cv2
import numpy as np


def analyze_quality(img):
    """Compute image quality metrics."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean = np.mean(gray)
    std = np.std(gray)
    dyn_range = gray.max() - gray.min()

    # sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return mean, std, dyn_range, sharpness


def score_image(img):
    """Compute overall quality score (higher = better)."""
    mean, std, dyn_range, sharpness = analyze_quality(img)

    score = (
        (std / 128) * 0.4 +
        (dyn_range / 255) * 0.3 +
        (sharpness / 1000) * 0.3
    )

    return score, (mean, std, dyn_range, sharpness)


def auto_gamma(img, mean):
    """Adaptive gamma correction."""
    gamma = 1.0

    if mean < 80:
        gamma = 0.5   # dark
    elif mean > 180:
        gamma = 2.0   # bright

    table = np.array([
        ((i / 255.0) ** gamma) * 255 for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(img, table)


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def enhance_image(img):
    """Smart enhancement pipeline."""
    score, metrics = score_image(img)
    mean, std, dyn_range, sharpness = metrics

    print(f"\n📊 Image Score: {score:.3f}")
    print(f"Mean: {mean:.1f}, Std: {std:.1f}, Range: {dyn_range}, Sharpness: {sharpness:.1f}")

    # Step 1: Gamma correction
    img = auto_gamma(img, mean)

    # Step 2: Contrast enhancement only if needed
    if std < 40 or dyn_range < 100:
        print("🔧 Applying CLAHE (low contrast detected)")
        img = apply_clahe(img)

    # Step 3: optional sharpening if blurry
    if sharpness < 100:
        print("🔧 Applying sharpening filter")
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

    return img


if __name__ == "__main__":
    path = "input.jpg"
    img = cv2.imread(path)

    enhanced = enhance_image(img)

    cv2.imshow("Original", img)
    cv2.imshow("Enhanced AI v2", enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()