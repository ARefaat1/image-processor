"""
Task 5 — Panorama Image Stitching
==================================
Member 3

Merges two overlapping images into a single seamless panorama.

Usage:
    1. Place overlapping image pairs in images/task5/ (name them like scene1_left.jpg, scene1_right.jpg)
    2. Run: python tasks/task5_panorama_stitching.py
    3. Results saved to outputs/task5/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import load_image, save_image, show_image, show_images_row


def detect_and_match(img1, img2, method="sift", ratio_thresh=0.75):
    """
    Detect keypoints, compute descriptors, and match features.
    
    Returns: keypoints1, keypoints2, good_matches, match_visualization
    """
    # Create detector
    if method == "sift":
        detector = cv2.SIFT_create()
    else:
        detector = cv2.ORB_create(nfeatures=3000)

    # Detect and compute
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    print(f"  Keypoints: img1={len(kp1)}, img2={len(kp2)}")

    if des1 is None or des2 is None:
        print("  ❌ No descriptors found!")
        return kp1, kp2, [], None

    # Match using BFMatcher + Lowe's ratio test
    if method == "sift":
        bf = cv2.BFMatcher(cv2.NORM_L2)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    print(f"  Good matches: {len(good)}")

    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good[:50], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return kp1, kp2, good, match_img


def compute_homography(kp1, kp2, good_matches, min_matches=10):
    """
    Compute homography matrix using RANSAC.
    
    Returns: H (3x3 homography matrix), mask
    """
    if len(good_matches) < min_matches:
        print(f"  ❌ Not enough matches ({len(good_matches)} < {min_matches})")
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = mask.ravel().sum()
    print(f"  Homography inliers: {inliers}/{len(good_matches)}")

    return H, mask


def warp_and_stitch(img1, img2, H):
    """
    Warp img1 using homography H and blend with img2.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Compute output canvas size
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners1_warped = cv2.perspectiveTransform(corners1, H)

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners1_warped, corners2], axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel()) - 10
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel()) + 10

    # Translation to handle negative coordinates
    translation = np.array([[1, 0, -x_min],
                             [0, 1, -y_min],
                             [0, 0, 1]], dtype=np.float64)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # Warp img1
    warped1 = cv2.warpPerspective(img1, translation @ H, (canvas_w, canvas_h))

    # Place img2 on canvas
    canvas = warped1.copy()
    canvas[-y_min:-y_min + h2, -x_min:-x_min + w2] = img2

    # Simple blending in overlap region
    mask1 = cv2.warpPerspective(
        np.ones((h1, w1), dtype=np.float32),
        translation @ H, (canvas_w, canvas_h)
    )
    mask2 = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    mask2[-y_min:-y_min + h2, -x_min:-x_min + w2] = 1.0

    overlap = (mask1 > 0) & (mask2 > 0)

    # Weighted blend in overlap
    if overlap.any():
        warped1_full = cv2.warpPerspective(img1, translation @ H, (canvas_w, canvas_h))
        canvas2 = np.zeros_like(canvas)
        canvas2[-y_min:-y_min + h2, -x_min:-x_min + w2] = img2

        alpha = 0.5
        canvas[overlap] = cv2.addWeighted(
            warped1_full, alpha, canvas2, 1 - alpha, 0
        )[overlap]

    return canvas


def crop_black_borders(img):
    """Remove black borders from the panorama."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return img[y:y+h, x:x+w]
    return img


def stitch_pair(img_left, img_right):
    """Full pipeline: detect → match → homography → warp → stitch."""
    kp1, kp2, good, match_img = detect_and_match(img_left, img_right)
    
    if len(good) < 10:
        print("  ❌ Not enough matches for stitching")
        return None, match_img

    H, mask = compute_homography(kp1, kp2, good)
    if H is None:
        return None, match_img

    panorama = warp_and_stitch(img_left, img_right, H)
    panorama = crop_black_borders(panorama)

    return panorama, match_img


def main():
    input_dir, output_dir = os.path.join("images", "task5"), os.path.join("outputs", "task5")

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"⚠️  No images in {input_dir}/")
        print("   Add overlapping image pairs named like:")
        print("   scene1_left.jpg, scene1_right.jpg")
        print("   Or: img1.jpg, img2.jpg")
        return

    # Try to find image pairs
    files = sorted([f for f in os.listdir(input_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    if len(files) < 2:
        print("  ❌ Need at least 2 images for stitching")
        return

    # Process pairs
    for i in range(0, len(files) - 1, 2):
        f1, f2 = files[i], files[i + 1]
        print(f"\n{'='*50}\nStitching: {f1} + {f2}\n{'='*50}")

        img1 = load_image(os.path.join(input_dir, f1))
        img2 = load_image(os.path.join(input_dir, f2))

        panorama, match_img = stitch_pair(img1, img2)

        if match_img is not None:
            show_image(match_img, "Feature Matches", figsize=(16, 6))
            save_image(match_img, os.path.join(output_dir, f"matches_{i//2+1}.png"))

        if panorama is not None:
            show_images_row([img1, img2, panorama],
                           ["Left", "Right", "Panorama"], figsize=(20, 6))
            save_image(panorama, os.path.join(output_dir, f"panorama_{i//2+1}.png"))
        else:
            print("  ❌ Stitching failed")

    print("\n✅ Task 5 complete!")


if __name__ == "__main__":
    main()
