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

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import (
    load_image, save_image, show_before_after, 
    print_analysis, analyze_image, to_grayscale
)


def classify_image(img):
    """Classify image as 'dark', 'bright', 'low_contrast', or 'normal'."""
    info = analyze_image(img)
    mean_b = info["mean_brightness"]
    std = info["std_dev"]
    dyn_range = info["max_val"] - info["min_val"]

    if mean_b < 80:
        return "dark"
    elif mean_b > 180:
        return "bright"
    elif std < 40 or dyn_range < 100:
        return "low_contrast"
    return "normal"


def enhance_dark(img):
    """Gamma correction (γ<1) + CLAHE for dark images."""
    corrected = (np.power(img / 255.0, 0.5) * 255).astype(np.uint8)
    if len(corrected.shape) == 3:
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.createCLAHE(3.0, (8, 8)).apply(corrected)


def enhance_bright(img):
    """Gamma correction (γ>1) + contrast stretching for bright images."""
    corrected = (np.power(img / 255.0, 2.0) * 255).astype(np.uint8)
    if len(corrected.shape) == 3:
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]
        mn, mx = l.min(), l.max()
        if mx > mn:
            lab[:, :, 0] = ((l - mn) / (mx - mn) * 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    mn, mx = corrected.min(), corrected.max()
    if mx > mn:
        return ((corrected - mn) / (mx - mn) * 255).astype(np.uint8)
    return corrected


def enhance_low_contrast(img):
    """CLAHE enhancement for low-contrast images."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(4.0, (8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.createCLAHE(4.0, (8, 8)).apply(img)


def auto_enhance(img):
    """Auto-detect and apply best enhancement. Returns (result, label)."""
    cls = classify_image(img)
    strategies = {
        "dark": enhance_dark,
        "bright": enhance_bright,
        "low_contrast": enhance_low_contrast,
        "normal": enhance_low_contrast,
    }
    print(f"  🔍 Detected: {cls.upper()} → applying {strategies[cls].__name__}")
    return strategies[cls](img), cls


def main():
    input_dir, output_dir = os.path.join("images", "task3"), os.path.join("outputs", "task3")

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"⚠️  No images in {input_dir}/. Add dark, bright, and low-contrast images.")
        return

    for fn in sorted(os.listdir(input_dir)):
        if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue
        print(f"\n{'='*50}\nProcessing: {fn}\n{'='*50}")
        img = load_image(os.path.join(input_dir, fn))
        print_analysis(img, fn)

        enhanced, cls = auto_enhance(img)
        show_before_after(img, enhanced, f"Original ({cls.upper()})", "Auto Enhanced")

        # Histogram comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].hist(to_grayscale(img).ravel(), 256, [0, 256], color='gray')
        axes[0].set_title(f"Original ({cls})")
        axes[1].hist(to_grayscale(enhanced).ravel(), 256, [0, 256], color='green')
        axes[1].set_title("Enhanced")
        plt.tight_layout()
        name = os.path.splitext(fn)[0]
        plt.savefig(os.path.join(output_dir, f"{name}_hist.png"), dpi=150)
        plt.show()

        save_image(enhanced, os.path.join(output_dir, f"{name}_{cls}_enhanced.png"))

    print("\n✅ Task 3 complete!")


if __name__ == "__main__":
    main()
