import time
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import fnblur
import matplotlib.pyplot as plt

IMAGE_PATH = "benchmarks/images/image_001.png"
OUTPUT_DIR = "benchmarks/results"


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def run_advanced_benchmark():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading {IMAGE_PATH}...")
    # Load as Alpha for fnblur
    original_cv2 = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if original_cv2 is None:
        raise FileNotFoundError(f"Could not load {IMAGE_PATH}")

    # Ensure RGBA for fair comparison
    if original_cv2.shape[2] == 3:
        img_bgra = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2BGRA)
        img_rgba = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2RGBA)
    else:
        img_bgra = original_cv2
        img_rgba = cv2.cvtColor(original_cv2, cv2.COLOR_BGRA2RGBA)

    # Use RGBA for PIL/fnblur, BGRA for OpenCV (native format)
    img_pil = Image.fromarray(img_rgba)

    # Settings
    iterations = 20
    kernel_size = 11
    sigma = 3  # kernel 11 ~ radius 5

    print(f"Image Size: {img_rgba.shape[:2]}")
    print(f"Running {iterations} iterations...")

    # --- fnblur ---
    # Working on a copy to avoid in-place modification affecting next runs
    fnblur_input = img_rgba.copy()
    start = time.time()
    # fnblur returns the result, it is NOT in-place
    fnblur_result = fnblur.gaussian(fnblur_input, iterations=iterations)
    fnblur_time = (time.time() - start) / (iterations)
    print(f"fnblur: {fnblur_time:.4f} s")

    # --- OpenCV ---
    cv2_input = img_bgra.copy()
    start = time.time()
    # To compare fairly with fnblur's "iterations" (which blurs the blurred result),
    # we must feed the output back as input.
    current_cv2_img = cv2_input
    for _ in range(iterations):
        current_cv2_img = cv2.GaussianBlur(
            current_cv2_img, (kernel_size, kernel_size), 0
        )
    opencv_time = (time.time() - start) / iterations
    # Convert back to RGBA for comparison
    opencv_result = cv2.cvtColor(current_cv2_img, cv2.COLOR_BGRA2RGBA)
    print(f"OpenCV: {opencv_time:.4f} s")

    # --- Pillow ---
    current_pil_img = img_pil
    start = time.time()
    for _ in range(iterations):
        current_pil_img = current_pil_img.filter(ImageFilter.GaussianBlur(sigma))
    pillow_time = (time.time() - start) / iterations
    pillow_result = np.array(current_pil_img)
    print(f"Pillow: {pillow_time:.4f} s")

    # --- Quality Check ---
    # We treat OpenCV as the "Ground Truth" for Gaussian Blur implementation
    psnr = calculate_psnr(opencv_result, fnblur_result)
    mse = calculate_mse(opencv_result, fnblur_result)
    print("\nQuality Metrics (fnblur vs OpenCV):")
    print(f"MSE: {mse:.2f} (Lower is better)")
    print(f"PSNR: {psnr:.2f} dB (Higher is better, >30 is good)")

    # --- Save Images ---
    cv2.imwrite(
        f"{OUTPUT_DIR}/output_fnblur.png",
        cv2.cvtColor(fnblur_result, cv2.COLOR_RGBA2BGRA),
    )
    cv2.imwrite(
        f"{OUTPUT_DIR}/output_opencv.png",
        cv2.cvtColor(opencv_result, cv2.COLOR_RGBA2BGRA),
    )
    cv2.imwrite(
        f"{OUTPUT_DIR}/output_pillow.png",
        cv2.cvtColor(pillow_result, cv2.COLOR_RGBA2BGRA),
    )

    # --- Plotting ---
    libs = ["fnblur", "OpenCV", "Pillow"]
    times = [fnblur_time, opencv_time, pillow_time]
    colors = ["#4CAF50", "#2196F3", "#FFC107"]  # Green, Blue, Amber

    plt.figure(figsize=(10, 6))
    bars = plt.bar(libs, times, color=colors)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}s",
            ha="center",
            va="bottom",
        )

    plt.title(f"Gaussian Blur Performance (Kernel {kernel_size})", fontsize=16)
    plt.ylabel("Time per Frame (seconds)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add speedup text
    speedup = opencv_time / fnblur_time
    plt.text(
        0.5,
        max(times) * 0.8,
        f"fnblur is {speedup:.1f}x faster than OpenCV",
        fontsize=14,
        ha="center",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.savefig(f"{OUTPUT_DIR}/benchmark_chart.png")
    print(f"\nSaved results to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_advanced_benchmark()
