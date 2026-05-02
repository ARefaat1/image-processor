"""
Generates all 8 test notebooks as .ipynb files.
Run once: python notebooks/generate_notebooks.py
"""
import json, os

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}
}

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.strip().splitlines(True),
            "execution_count": None, "outputs": []}

def make_nb(cells):
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": KERNEL, "cells": cells}

def save(nb, name):
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Created: {name}")

# ──────────────────────────────────────────────────────────────
#  NOTEBOOK 1
# ──────────────────────────────────────────────────────────────
save(make_nb([
    md("# Task 1 — Selective Object Enhancement\n\n**Member 1** | Sharpen a specific colored object while blurring the rest."),
    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import load_image, show_image, show_images_row, show_before_after, print_analysis
from tasks.task1_selective_enhancement import create_red_mask, create_color_mask, selective_enhance, sharpen_image

print("✅ Imports OK")"""),
    md("## 1. Load Test Image\n\nPlace a flower image (with red and/or yellow flowers) in `images/task1/`."),
    code("""# Load an image — update the filename to match your image
img = load_image('../images/task1/flowers.jpg')  # <-- CHANGE THIS
print_analysis(img, "Input Image")
show_image(img, "Original Image")"""),
    md("## 2. Create Color Masks"),
    code("""# Red mask (works for red roses, red objects, etc.)
red_mask = create_red_mask(img)
show_image(red_mask, "Red Color Mask", cmap='gray')

# Yellow mask
yellow_mask = create_color_mask(img, lower_hsv=(20, 70, 50), upper_hsv=(35, 255, 255))
show_image(yellow_mask, "Yellow Color Mask", cmap='gray')

# You can adjust HSV ranges for other colors:
# Blue:   (100, 70, 50) to (130, 255, 255)
# Green:  (35, 70, 50) to (85, 255, 255)
# Purple: (130, 70, 50) to (160, 255, 255)"""),
    md("## 3. Apply Selective Enhancement"),
    code("""# Sharpen red objects, blur everything else
result_red = selective_enhance(img, red_mask, enhance_sharp=True, blur_ksize=21)

show_images_row(
    [img, red_mask, result_red],
    ["Original", "Red Mask", "Red Sharpened / Rest Blurred"],
    figsize=(18, 6)
)"""),
    code("""# Sharpen yellow objects instead
result_yellow = selective_enhance(img, yellow_mask, enhance_sharp=True, blur_ksize=21)

show_images_row(
    [img, yellow_mask, result_yellow],
    ["Original", "Yellow Mask", "Yellow Sharpened / Rest Blurred"],
    figsize=(18, 6)
)"""),
    md("## 4. Save Results"),
    code("""os.makedirs('../outputs/task1', exist_ok=True)
cv2.imwrite('../outputs/task1/red_enhanced.png', result_red)
cv2.imwrite('../outputs/task1/yellow_enhanced.png', result_yellow)
cv2.imwrite('../outputs/task1/red_mask.png', red_mask)
print("✅ Saved to outputs/task1/")"""),
]), "test_task1.ipynb")

# ──────────────────────────────────────────────────────────────
#  NOTEBOOK 2
# ──────────────────────────────────────────────────────────────
save(make_nb([
    md("# Task 2 — Comparative Study of Sharpening Pipelines for X-Ray Images\n\n**Member 2** | Three different enhancement pipelines compared."),
    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import load_image, show_images_row, print_analysis, to_grayscale, compute_psnr
from tasks.task2_xray_sharpening import pipeline_a, pipeline_b, pipeline_c, compare_pipelines

print("✅ Imports OK")"""),
    md("## 1. Load X-Ray Image\n\nPlace X-ray images in `images/task2/`. Get them from Kaggle (Chest X-Ray dataset)."),
    code("""img = load_image('../images/task2/xray.jpg')  # <-- CHANGE THIS
gray = to_grayscale(img)
print_analysis(gray, "X-Ray Image")
show_images_row([gray], ["Original X-Ray"], figsize=(8, 6))"""),
    md("## 2. Run Three Pipelines"),
    code("""result_a = pipeline_a(gray)
result_b = pipeline_b(gray)
result_c = pipeline_c(gray)

show_images_row(
    [gray, result_a, result_b, result_c],
    ["Original", "Pipeline A\\n(HistEq+Unsharp+Median)",
     "Pipeline B\\n(CLAHE+Laplacian+Median)",
     "Pipeline C\\n(Negative+Gamma+HighBoost)"],
    figsize=(22, 6)
)"""),
    md("## 3. Compare Metrics"),
    code("""print(f"{'Pipeline':<15} {'Contrast (σ)':<15} {'Mean':<10} {'PSNR vs Original':<18}")
print("-" * 58)
for name, result in [("Pipeline A", result_a), ("Pipeline B", result_b), ("Pipeline C", result_c)]:
    contrast = np.std(result.astype(np.float64))
    mean_val = np.mean(result)
    psnr = compute_psnr(gray, result)
    print(f"{name:<15} {contrast:<15.2f} {mean_val:<10.2f} {psnr:<18.2f} dB")"""),
    md("## 4. Histogram Comparison"),
    code("""fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, img_data, title in zip(axes, [gray, result_a, result_b, result_c],
                                ["Original", "Pipeline A", "Pipeline B", "Pipeline C"]):
    ax.hist(img_data.ravel(), 256, [0, 256], color='black')
    ax.set_title(title)
    ax.set_xlim([0, 256])
plt.tight_layout()
plt.show()"""),
    md("## 5. Save Results"),
    code("""os.makedirs('../outputs/task2', exist_ok=True)
cv2.imwrite('../outputs/task2/pipeline_a.png', result_a)
cv2.imwrite('../outputs/task2/pipeline_b.png', result_b)
cv2.imwrite('../outputs/task2/pipeline_c.png', result_c)
print("✅ Saved to outputs/task2/")"""),
]), "test_task2.ipynb")

# ──────────────────────────────────────────────────────────────
#  NOTEBOOK 3
# ──────────────────────────────────────────────────────────────
save(make_nb([
    md("# Task 3 — Intelligent Auto Image Enhancement\n\n**Member 2** | Auto-detect dark/bright/low-contrast and enhance."),
    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import load_image, show_before_after, print_analysis, to_grayscale
from tasks.task3_auto_enhancement import classify_image, auto_enhance, enhance_dark, enhance_bright, enhance_low_contrast

print("✅ Imports OK")"""),
    md("## 1. Load Test Images\n\nPrepare 3 images in `images/task3/`: one dark, one bright, one low-contrast."),
    code("""# Test with a dark image
img_dark = load_image('../images/task3/dark.jpg')  # <-- CHANGE THIS
print(f"Classification: {classify_image(img_dark)}")
enhanced_dark, cls = auto_enhance(img_dark)
show_before_after(img_dark, enhanced_dark, f"Original ({cls})", "Auto Enhanced")"""),
    code("""# Test with a bright image
img_bright = load_image('../images/task3/bright.jpg')  # <-- CHANGE THIS
print(f"Classification: {classify_image(img_bright)}")
enhanced_bright, cls = auto_enhance(img_bright)
show_before_after(img_bright, enhanced_bright, f"Original ({cls})", "Auto Enhanced")"""),
    code("""# Test with a low contrast image
img_lowc = load_image('../images/task3/low_contrast.jpg')  # <-- CHANGE THIS
print(f"Classification: {classify_image(img_lowc)}")
enhanced_lowc, cls = auto_enhance(img_lowc)
show_before_after(img_lowc, enhanced_lowc, f"Original ({cls})", "Auto Enhanced")"""),
    md("## 2. Histogram Before/After"),
    code("""fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for i, (orig, enh, label) in enumerate([
    (img_dark, enhanced_dark, "Dark"),
    (img_bright, enhanced_bright, "Bright"),
    (img_lowc, enhanced_lowc, "Low Contrast")
]):
    axes[0, i].hist(to_grayscale(orig).ravel(), 256, [0, 256], color='gray')
    axes[0, i].set_title(f"{label} - Before")
    axes[1, i].hist(to_grayscale(enh).ravel(), 256, [0, 256], color='green')
    axes[1, i].set_title(f"{label} - After")
plt.tight_layout()
plt.show()"""),
    md("## 3. Save Results"),
    code("""os.makedirs('../outputs/task3', exist_ok=True)
cv2.imwrite('../outputs/task3/dark_enhanced.png', enhanced_dark)
cv2.imwrite('../outputs/task3/bright_enhanced.png', enhanced_bright)
cv2.imwrite('../outputs/task3/low_contrast_enhanced.png', enhanced_lowc)
print("✅ Saved to outputs/task3/")"""),
]), "test_task3.ipynb")

# ──────────────────────────────────────────────────────────────
#  NOTEBOOK 4
# ──────────────────────────────────────────────────────────────
save(make_nb([
    md("# Task 4 — Document Cleaning System\n\n**Member 1** | Clean noisy/shadowed scanned documents."),
    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import load_image, show_images_row, print_analysis, to_grayscale
from tasks.task4_document_cleaning import clean_document, remove_shadows, adaptive_binarize, morphological_cleanup, denoise

print("✅ Imports OK")"""),
    md("## 1. Load Scanned Document\n\nDownload noisy document images from: https://medium.com/data-science/denoising-noisy-documents-6807c34730c4"),
    code("""doc = load_image('../images/task4/noisy_doc.jpg')  # <-- CHANGE THIS
print_analysis(doc, "Scanned Document")
show_images_row([doc], ["Original Document"], figsize=(8, 10))"""),
    md("## 2. Run Full Cleaning Pipeline"),
    code("""steps = clean_document(doc)

show_images_row(
    [steps["gray"], steps["denoised"], steps["shadow_free"], steps["binary"], steps["cleaned"]],
    ["1. Grayscale", "2. Denoised", "3. Shadow Removed", "4. Binarized", "5. Final"],
    figsize=(25, 6)
)"""),
    md("## 3. Step-by-Step Inspection"),
    code("""# You can tweak individual steps:
gray = to_grayscale(doc)

# Try different adaptive threshold parameters
bin1 = adaptive_binarize(gray, method="gaussian", block_size=11, C=8)
bin2 = adaptive_binarize(gray, method="gaussian", block_size=15, C=10)
bin3 = adaptive_binarize(gray, method="mean", block_size=15, C=12)

show_images_row([bin1, bin2, bin3],
    ["Gaussian (11, C=8)", "Gaussian (15, C=10)", "Mean (15, C=12)"],
    figsize=(18, 6))"""),
    md("## 4. Save Results"),
    code("""os.makedirs('../outputs/task4', exist_ok=True)
cv2.imwrite('../outputs/task4/cleaned.png', steps["cleaned"])
cv2.imwrite('../outputs/task4/binarized.png', steps["binary"])
print("✅ Saved to outputs/task4/")"""),
]), "test_task4.ipynb")

# ──────────────────────────────────────────────────────────────
#  NOTEBOOK 5
# ──────────────────────────────────────────────────────────────
save(make_nb([
    md("# Task 5 — Panorama Image Stitching\n\n**Member 3** | Merge overlapping images into a panorama."),
    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import load_image, show_image, show_images_row
from tasks.task5_panorama_stitching import detect_and_match, compute_homography, warp_and_stitch, crop_black_borders, stitch_pair

print("✅ Imports OK")"""),
    md("## 1. Load Overlapping Image Pair\n\nTake two photos of the same scene with ~30-50% overlap."),
    code("""left = load_image('../images/task5/left.jpg')   # <-- CHANGE THIS
right = load_image('../images/task5/right.jpg')  # <-- CHANGE THIS

show_images_row([left, right], ["Left Image", "Right Image"], figsize=(16, 6))"""),
    md("## 2. Feature Detection & Matching"),
    code("""kp1, kp2, good_matches, match_img = detect_and_match(left, right, method="sift")

print(f"Keypoints: left={len(kp1)}, right={len(kp2)}")
print(f"Good matches: {len(good_matches)}")

show_image(match_img, "Feature Matches (top 50)", figsize=(18, 7))"""),
    md("## 3. Compute Homography"),
    code("""H, mask = compute_homography(kp1, kp2, good_matches)
if H is not None:
    print("Homography matrix:")
    print(H)
    inliers = mask.ravel().sum()
    print(f"\\nInliers: {inliers}/{len(good_matches)}")"""),
    md("## 4. Warp & Stitch"),
    code("""panorama, match_vis = stitch_pair(left, right)

if panorama is not None:
    show_images_row([left, right, panorama],
                   ["Left", "Right", "Panorama"],
                   figsize=(22, 6))"""),
    md("## 5. Save Results"),
    code("""os.makedirs('../outputs/task5', exist_ok=True)
if panorama is not None:
    cv2.imwrite('../outputs/task5/panorama.png', panorama)
if match_img is not None:
    cv2.imwrite('../outputs/task5/matches.png', match_img)
print("✅ Saved to outputs/task5/")"""),
]), "test_task5.ipynb")

# ──────────────────────────────────────────────────────────────
#  NOTEBOOK 6
# ──────────────────────────────────────────────────────────────
save(make_nb([
    md("# Task 6 — Transformed Object Recognition\n\n**Member 4** | Find an object even if rotated, scaled, or translated."),
    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import load_image, show_image, show_images_row
from tasks.task6_object_recognition import detect_features, match_features, find_object

print("✅ Imports OK")"""),
    md("## 1. Load Template & Scene\n\nPrepare:\n- `template.jpg` — the object to find\n- `scene1.jpg` — a scene with the object rotated/scaled"),
    code("""template = load_image('../images/task6/template.jpg')  # <-- CHANGE THIS
scene = load_image('../images/task6/scene1.jpg')        # <-- CHANGE THIS

show_images_row([template, scene], ["Template", "Scene"], figsize=(14, 6))"""),
    md("## 2. Detect & Match Features"),
    code("""kp1, des1 = detect_features(template, method="sift")
kp2, des2 = detect_features(scene, method="sift")

print(f"Template keypoints: {len(kp1)}")
print(f"Scene keypoints: {len(kp2)}")

# Draw keypoints
img_kp1 = cv2.drawKeypoints(template, kp1, None, color=(0, 255, 0))
img_kp2 = cv2.drawKeypoints(scene, kp2, None, color=(0, 255, 0))
show_images_row([img_kp1, img_kp2], ["Template Keypoints", "Scene Keypoints"], figsize=(16, 6))"""),
    md("## 3. Object Detection"),
    code("""result, match_img, found = find_object(template, scene, method="sift")

if match_img is not None:
    show_image(match_img, "Feature Matches", figsize=(18, 7))

show_images_row([template, scene, result],
               ["Template", "Scene", f"Detection: {'✅ FOUND' if found else '❌ NOT FOUND'}"],
               figsize=(20, 6))"""),
    md("## 4. Test with Transformed Versions"),
    code("""# Rotate the template and try again
h, w = template.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), 45, 0.8)  # 45° rotation, 80% scale
rotated = cv2.warpAffine(template, M, (w, h))

result2, _, found2 = find_object(rotated, scene, method="sift")
show_images_row([rotated, result2],
               ["Rotated Template", f"Detection: {'✅ FOUND' if found2 else '❌ NOT FOUND'}"],
               figsize=(14, 6))"""),
    md("## 5. Save Results"),
    code("""os.makedirs('../outputs/task6', exist_ok=True)
cv2.imwrite('../outputs/task6/detection_result.png', result)
if match_img is not None:
    cv2.imwrite('../outputs/task6/matches.png', match_img)
print("✅ Saved to outputs/task6/")"""),
]), "test_task6.ipynb")

# ──────────────────────────────────────────────────────────────
#  NOTEBOOK 7
# ──────────────────────────────────────────────────────────────
save(make_nb([
    md("# Task 7 — Depth Approximation from Two Images\n\n**Member 4** | Estimate depth using stereo disparity."),
    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import load_image, show_images_row, to_grayscale
from tasks.task7_depth_approximation import (
    compute_disparity_bm, compute_disparity_sgbm,
    normalize_disparity, create_depth_map, create_colored_depth
)

print("✅ Imports OK")"""),
    md("## 1. Load Stereo Pair\n\nTake two photos of the same scene, shifting the camera slightly to the right (~5-10cm)."),
    code("""left = load_image('../images/task7/left.jpg')   # <-- CHANGE THIS
right = load_image('../images/task7/right.jpg')  # <-- CHANGE THIS

show_images_row([left, right], ["Left View", "Right View"], figsize=(16, 6))"""),
    md("## 2. Compute Disparity Maps"),
    code("""left_gray = to_grayscale(left)
right_gray = to_grayscale(right)

# Method 1: StereoBM
disp_bm = compute_disparity_bm(left_gray, right_gray, num_disparities=64, block_size=15)
depth_bm = create_depth_map(disp_bm, smooth=True)

# Method 2: StereoSGBM (better quality)
disp_sgbm = compute_disparity_sgbm(left_gray, right_gray, num_disparities=64, block_size=7)
depth_sgbm = create_depth_map(disp_sgbm, smooth=True)

print("Disparity computed!")"""),
    md("## 3. Visualize Depth Maps"),
    code("""depth_bm_color = create_colored_depth(depth_bm)
depth_sgbm_color = create_colored_depth(depth_sgbm)

show_images_row(
    [left, depth_bm, depth_bm_color, depth_sgbm_color],
    ["Original", "Depth BM (gray)", "Depth BM (color)", "Depth SGBM (color)"],
    figsize=(24, 6)
)"""),
    md("## 4. Tune Parameters\n\nAdjust `numDisparities` and `blockSize` for best results."),
    code("""# Try different settings
for nd in [32, 64, 128]:
    for bs in [9, 15, 21]:
        disp = compute_disparity_bm(left_gray, right_gray, num_disparities=nd, block_size=bs)
        depth = create_depth_map(disp)
        plt.figure(figsize=(6, 4))
        plt.imshow(depth, cmap='jet')
        plt.title(f"numDisp={nd}, blockSize={bs}")
        plt.colorbar()
        plt.axis('off')
        plt.show()"""),
    md("## 5. Save Results"),
    code("""os.makedirs('../outputs/task7', exist_ok=True)
cv2.imwrite('../outputs/task7/depth_bm.png', depth_bm)
cv2.imwrite('../outputs/task7/depth_bm_color.png', depth_bm_color)
cv2.imwrite('../outputs/task7/depth_sgbm_color.png', depth_sgbm_color)
print("✅ Saved to outputs/task7/")"""),
]), "test_task7.ipynb")

# ──────────────────────────────────────────────────────────────
#  NOTEBOOK 8
# ──────────────────────────────────────────────────────────────
save(make_nb([
    md("# Task 8 — High Dynamic Range (HDR) Imaging\n\n**Member 5** | Merge multiple exposures into HDR."),
    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv_utils import load_image, show_image, show_images_row
from tasks.task8_hdr_imaging import (
    align_images, estimate_response_curve, merge_to_hdr,
    tonemap_reinhard, tonemap_drago, mertens_fusion
)

print("✅ Imports OK")"""),
    md("## 1. Load Multiple Exposures\n\nPlace 3+ images of the same scene at different exposures in `images/task8/`."),
    code("""# Load exposure bracket images — CHANGE THESE FILENAMES
files = ['exposure1.jpg', 'exposure2.jpg', 'exposure3.jpg']  # <-- CHANGE THIS
images = [load_image(f'../images/task8/{f}') for f in files]

print(f"Loaded {len(images)} exposures")
show_images_row(images, [f"Exposure {i+1}" for i in range(len(images))],
                figsize=(6*len(images), 5))"""),
    md("## 2. Align Images"),
    code("""images = align_images(images)
print("Images aligned!")"""),
    md("## 3. Method 1 — Mertens Exposure Fusion (No exposure times needed)"),
    code("""fusion = mertens_fusion(images)
show_image(fusion, "Mertens Exposure Fusion", figsize=(10, 7))"""),
    md("## 4. Method 2 — Debevec HDR + Tone Mapping"),
    code("""# Exposure times (adjust to match your actual camera settings)
exposure_times = np.array([1/30, 1/60, 1/125], dtype=np.float32)  # <-- ADJUST

try:
    response = estimate_response_curve(images, exposure_times)
    hdr = merge_to_hdr(images, exposure_times, response)

    # Plot camera response function
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(response[:, :, 0], 'b', label='Blue')
    ax.plot(response[:, :, 1], 'g', label='Green')
    ax.plot(response[:, :, 2], 'r', label='Red')
    ax.set_title("Camera Response Function")
    ax.legend()
    plt.show()

    # Tone mapping
    ldr_reinhard = tonemap_reinhard(hdr)
    ldr_drago = tonemap_drago(hdr)

    show_images_row([ldr_reinhard, ldr_drago, fusion],
                   ["Reinhard", "Drago", "Mertens"],
                   figsize=(20, 6))
except Exception as e:
    print(f"Debevec failed: {e}")
    print("Use Mertens fusion result instead.")"""),
    md("## 5. Final Comparison"),
    code("""show_images_row(
    images + [fusion],
    [f"Exp {i+1}" for i in range(len(images))] + ["HDR Result"],
    figsize=(5*len(images)+5, 5)
)"""),
    md("## 6. Save Results"),
    code("""os.makedirs('../outputs/task8', exist_ok=True)
cv2.imwrite('../outputs/task8/hdr_mertens.png', fusion)
try:
    cv2.imwrite('../outputs/task8/hdr_reinhard.png', ldr_reinhard)
    cv2.imwrite('../outputs/task8/hdr_drago.png', ldr_drago)
except:
    pass
print("✅ Saved to outputs/task8/")"""),
]), "test_task8.ipynb")

print("\n🎉 All 8 notebooks created successfully!")
