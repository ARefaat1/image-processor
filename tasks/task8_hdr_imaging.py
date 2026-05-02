"""
Task 8 — High Dynamic Range (HDR) Imaging
==========================================
Member 5

Combines multiple exposures into a single HDR image with tone mapping.

Usage:
    1. Place multi-exposure images in images/task8/
       Name them in order: exposure1.jpg, exposure2.jpg, exposure3.jpg
    2. Run: python tasks/task8_hdr_imaging.py
    3. Results saved to outputs/task8/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import load_image, save_image, show_images_row, show_image


def align_images(images):
    """Align multiple exposure images using MTB (Median Threshold Bitmap)."""
    align = cv2.createAlignMTB()
    align.process(images, images)
    return images


def estimate_response_curve(images, exposure_times):
    """
    Estimate camera response function using Debevec's method.
    """
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, exposure_times)
    return response


def merge_to_hdr(images, exposure_times, response=None):
    """
    Merge multiple exposures into an HDR radiance map.
    """
    if response is not None:
        merge = cv2.createMergeDebevec()
        hdr = merge.process(images, exposure_times, response)
    else:
        merge = cv2.createMergeMertens()
        hdr = merge.process(images)
    return hdr


def tonemap_reinhard(hdr, gamma=1.3, intensity=0.0, light_adapt=0.8, color_adapt=0.0):
    """Apply Reinhard tone mapping."""
    tonemap = cv2.createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)
    ldr = tonemap.process(hdr)
    ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
    return ldr


def tonemap_drago(hdr, gamma=1.0, saturation=1.0):
    """Apply Drago tone mapping."""
    tonemap = cv2.createTonemapDrago(gamma, saturation)
    ldr = tonemap.process(hdr)
    ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
    return ldr


def tonemap_simple(hdr):
    """Simple global tone mapping using log compression."""
    hdr_float = hdr.astype(np.float32)
    # Avoid log(0)
    hdr_float = np.maximum(hdr_float, 1e-6)
    # Log compression
    ldr = np.log1p(hdr_float)
    ldr = cv2.normalize(ldr, None, 0, 255, cv2.NORM_MINMAX)
    return ldr.astype(np.uint8)


def mertens_fusion(images):
    """
    Exposure fusion using Mertens' method (no exposure times needed).
    Good fallback when exposure times are unknown.
    """
    merge = cv2.createMergeMertens()
    fusion = merge.process(images)
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion


def main():
    input_dir, output_dir = os.path.join("images", "task8"), os.path.join("outputs", "task8")

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"⚠️  No images in {input_dir}/")
        print("   Add 3+ images of the same scene at different exposures:")
        print("   - Use bracketed exposure on your phone/camera")
        print("   - Or download from HDR image datasets online")
        print("   Name them: exposure1.jpg, exposure2.jpg, exposure3.jpg")
        return

    files = sorted([f for f in os.listdir(input_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])

    if len(files) < 2:
        print("  ❌ Need at least 2 exposure images")
        return

    print(f"\n{'='*50}")
    print(f"HDR Processing: {len(files)} exposures")
    print(f"{'='*50}")

    # Load all exposure images
    images = [load_image(os.path.join(input_dir, f)) for f in files]
    print(f"  Loaded {len(images)} images")

    # Align images
    print("  Aligning images...")
    images = align_images(images)

    # Show input exposures
    show_images_row(images, [f"Exposure {i+1}" for i in range(len(images))],
                    figsize=(5 * len(images), 5))

    # --- Method 1: Mertens Fusion (no exposure times needed) ---
    print("\n  Method 1: Mertens Exposure Fusion...")
    fusion = mertens_fusion(images)
    show_image(fusion, "Mertens Fusion", figsize=(10, 7))
    save_image(fusion, os.path.join(output_dir, "hdr_mertens.png"))

    # --- Method 2: Debevec HDR + Reinhard Tonemap ---
    # Estimate exposure times (if not known, use reasonable defaults)
    # These should be adjusted based on actual exposure settings
    n = len(images)
    exposure_times = np.array([1.0 / (2 ** i) for i in range(n)], dtype=np.float32)
    print(f"  Using estimated exposure times: {exposure_times}")

    print("\n  Method 2: Debevec HDR + Tone Mapping...")
    try:
        response = estimate_response_curve(images, exposure_times)
        hdr = merge_to_hdr(images, exposure_times, response)

        # Plot response curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(response[:, :, 0], 'b', label='Blue')
        ax.plot(response[:, :, 1], 'g', label='Green')
        ax.plot(response[:, :, 2], 'r', label='Red')
        ax.set_title("Camera Response Function")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Log Exposure")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "response_curve.png"), dpi=150)
        plt.show()

        # Reinhard tone mapping
        ldr_reinhard = tonemap_reinhard(hdr)
        show_image(ldr_reinhard, "Reinhard Tone Mapping", figsize=(10, 7))
        save_image(ldr_reinhard, os.path.join(output_dir, "hdr_reinhard.png"))

        # Drago tone mapping
        ldr_drago = tonemap_drago(hdr)
        show_image(ldr_drago, "Drago Tone Mapping", figsize=(10, 7))
        save_image(ldr_drago, os.path.join(output_dir, "hdr_drago.png"))

    except Exception as e:
        print(f"  ⚠️ Debevec method failed: {e}")
        print("  Falling back to Mertens fusion only.")

    # --- Final comparison ---
    print("\n  Generating comparison...")
    comparison_imgs = images + [fusion]
    comparison_titles = [f"Exp {i+1}" for i in range(len(images))] + ["HDR Result"]
    show_images_row(comparison_imgs, comparison_titles,
                    figsize=(5 * len(comparison_imgs), 5))

    print("\n✅ Task 8 complete!")


if __name__ == "__main__":
    main()
