"""
CV Project — Shared Utility Functions
======================================
Common helpers for image I/O, display, comparison, and metrics.
Used across all 8 tasks.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────
#  Image I/O
# ─────────────────────────────────────────────

def load_image(path, color=True):
    """
    Load an image from disk.

    Parameters
    ----------
    path : str
        Path to the image file.
    color : bool
        If True, load as BGR (default). If False, load as grayscale.

    Returns
    -------
    numpy.ndarray
        The loaded image.
    """
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def save_image(img, path):
    """
    Save an image to disk. Creates parent directories if needed.

    Parameters
    ----------
    img : numpy.ndarray
        The image to save.
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
#  Display Helpers
# ─────────────────────────────────────────────

def bgr_to_rgb(img):
    """Convert BGR (OpenCV default) to RGB for matplotlib display."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_image(img, title="Image", cmap=None, figsize=(8, 6)):
    """
    Display a single image using matplotlib.

    Parameters
    ----------
    img : numpy.ndarray
        Image to display (BGR or grayscale).
    title : str
        Window title.
    cmap : str or None
        Colormap for grayscale images (e.g., 'gray').
    figsize : tuple
        Figure size.
    """
    plt.figure(figsize=figsize)
    if len(img.shape) == 3:
        plt.imshow(bgr_to_rgb(img))
    else:
        plt.imshow(img, cmap=cmap or "gray")
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_images_row(images, titles=None, figsize=(16, 5), cmap=None):
    """
    Display multiple images in a single row.

    Parameters
    ----------
    images : list of numpy.ndarray
        List of images to display.
    titles : list of str or None
        Titles for each image.
    figsize : tuple
        Figure size.
    cmap : str or None
        Colormap for grayscale images.
    """
    n = len(images)
    if titles is None:
        titles = [f"Image {i+1}" for i in range(n)]

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 3:
            ax.imshow(bgr_to_rgb(img))
        else:
            ax.imshow(img, cmap=cmap or "gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_before_after(before, after, title_before="Before", title_after="After", figsize=(14, 6)):
    """
    Display a before/after comparison side by side.

    Parameters
    ----------
    before : numpy.ndarray
        Original image.
    after : numpy.ndarray
        Processed image.
    title_before : str
        Title for the original.
    title_after : str
        Title for the processed.
    figsize : tuple
        Figure size.
    """
    show_images_row([before, after], [title_before, title_after], figsize=figsize)


# ─────────────────────────────────────────────
#  Histogram Helpers
# ─────────────────────────────────────────────

def plot_histogram(img, title="Histogram", figsize=(8, 4)):
    """
    Plot the histogram of an image (grayscale or color channels).

    Parameters
    ----------
    img : numpy.ndarray
        Input image (grayscale or BGR).
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    """
    plt.figure(figsize=figsize)
    if len(img.shape) == 3:
        colors = ('b', 'g', 'r')
        labels = ('Blue', 'Green', 'Red')
        for i, (color, label) in enumerate(zip(colors, labels)):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=label)
        plt.legend()
    else:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
    plt.title(title, fontsize=14)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  Image Analysis
# ─────────────────────────────────────────────

def analyze_image(img):
    """
    Analyze basic image properties: brightness, contrast, size.

    Parameters
    ----------
    img : numpy.ndarray
        Input image (BGR or grayscale).

    Returns
    -------
    dict
        Dictionary with keys: 'mean_brightness', 'std_dev', 'shape', 'dtype',
        'min_val', 'max_val'.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    mean, std = cv2.meanStdDev(gray)
    return {
        "mean_brightness": float(mean[0][0]),
        "std_dev": float(std[0][0]),
        "shape": img.shape,
        "dtype": str(img.dtype),
        "min_val": int(gray.min()),
        "max_val": int(gray.max()),
    }


def print_analysis(img, name="Image"):
    """Print a summary of image analysis."""
    info = analyze_image(img)
    print(f"\n{'='*40}")
    print(f"  Analysis: {name}")
    print(f"{'='*40}")
    print(f"  Shape:       {info['shape']}")
    print(f"  Dtype:       {info['dtype']}")
    print(f"  Brightness:  {info['mean_brightness']:.1f} / 255")
    print(f"  Contrast:    {info['std_dev']:.1f} (std dev)")
    print(f"  Range:       [{info['min_val']}, {info['max_val']}]")
    print(f"{'='*40}\n")
    return info


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────

def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    Parameters
    ----------
    img1, img2 : numpy.ndarray
        Images to compare (same size and type).

    Returns
    -------
    float
        PSNR value in dB. Higher is better.
    """
    return cv2.PSNR(img1, img2)


def compute_mse(img1, img2):
    """
    Compute Mean Squared Error between two images.

    Parameters
    ----------
    img1, img2 : numpy.ndarray
        Images to compare (same size and type).

    Returns
    -------
    float
        MSE value. Lower is better.
    """
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)


# ─────────────────────────────────────────────
#  Common Processing Helpers
# ─────────────────────────────────────────────

def resize_to_match(img, target):
    """Resize img to match target's dimensions."""
    h, w = target.shape[:2]
    return cv2.resize(img, (w, h))


def to_grayscale(img):
    """Convert to grayscale if needed."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def normalize_to_uint8(img):
    """Normalize a float image to uint8 [0, 255]."""
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img_normalized.astype(np.uint8)
