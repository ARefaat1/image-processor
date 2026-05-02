"""
Task 6 — Transformed Object Recognition
========================================
Member 4

Identifies an object in a scene even if rotated, translated, or scaled.

Usage:
    1. Place images in images/task6/:
       - template.jpg (the object to find)
       - scene1.jpg, scene2.jpg, ... (scenes containing the object)
    2. Run: python tasks/task6_object_recognition.py
    3. Results saved to outputs/task6/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from cv_utils import load_image, save_image, show_image, show_images_row


def detect_features(img, method="sift"):
    """Detect keypoints and compute descriptors."""
    if method == "sift":
        detector = cv2.SIFT_create()
    else:
        detector = cv2.ORB_create(nfeatures=5000)
    return detector.detectAndCompute(img, None)


def match_features(des1, des2, method="sift", ratio=0.75):
    """Match descriptors using BFMatcher + Lowe's ratio test."""
    norm = cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def find_object(template, scene, method="sift", min_matches=10):
    """
    Find a template object in a scene image.
    
    Returns: result image with bounding box, match visualization, success flag
    """
    kp1, des1 = detect_features(template, method)
    kp2, des2 = detect_features(scene, method)

    print(f"  Template keypoints: {len(kp1)}")
    print(f"  Scene keypoints: {len(kp2)}")

    if des1 is None or des2 is None:
        print("  ❌ No descriptors found")
        return scene.copy(), None, False

    good = match_features(des1, des2, method)
    print(f"  Good matches: {len(good)}")

    # Draw match visualization
    match_img = cv2.drawMatches(template, kp1, scene, kp2, good[:50], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if len(good) < min_matches:
        print(f"  ❌ Not enough matches ({len(good)} < {min_matches})")
        return scene.copy(), match_img, False

    # Find homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = mask.ravel().sum() if mask is not None else 0
    print(f"  Inliers: {inliers}/{len(good)}")

    if H is None or inliers < min_matches // 2:
        print("  ❌ Homography estimation failed")
        return scene.copy(), match_img, False

    # Draw bounding box around detected object
    h, w = template.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, H)

    result = scene.copy()
    cv2.polylines(result, [np.int32(transformed)], True, (0, 255, 0), 3)
    
    # Add label
    cx, cy = int(transformed[:, 0, 0].mean()), int(transformed[:, 0, 1].mean())
    cv2.putText(result, "DETECTED", (cx - 50, cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print("  ✅ Object detected!")
    return result, match_img, True


def main():
    input_dir, output_dir = os.path.join("images", "task6"), os.path.join("outputs", "task6")

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"⚠️  No images in {input_dir}/")
        print("   Add a 'template.jpg' and 'scene1.jpg', 'scene2.jpg', etc.")
        print("   The scene images should contain the template object,")
        print("   possibly rotated, scaled, or translated.")
        return

    files = os.listdir(input_dir)
    
    # Find template
    template_file = None
    for f in files:
        if "template" in f.lower():
            template_file = f
            break
    
    if template_file is None:
        # Use first image as template
        template_file = sorted(files)[0]
        print(f"  No 'template' image found, using: {template_file}")

    template = load_image(os.path.join(input_dir, template_file))
    scene_files = [f for f in sorted(files) 
                   if f != template_file and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not scene_files:
        print("  ❌ No scene images found (need at least 1 besides template)")
        return

    for fn in scene_files:
        print(f"\n{'='*50}\nSearching for object in: {fn}\n{'='*50}")
        scene = load_image(os.path.join(input_dir, fn))

        result, match_img, found = find_object(template, scene)

        show_images_row([template, scene, result],
                       ["Template", "Scene", "Detection Result"], figsize=(20, 6))

        if match_img is not None:
            show_image(match_img, "Feature Matches", figsize=(16, 6))
            name = os.path.splitext(fn)[0]
            save_image(match_img, os.path.join(output_dir, f"{name}_matches.png"))

        name = os.path.splitext(fn)[0]
        save_image(result, os.path.join(output_dir, f"{name}_result.png"))

    print("\n✅ Task 6 complete!")


if __name__ == "__main__":
    main()
