import time
import cv2
import numpy as np
from PIL import Image, ImageFilter
import fnblur


def benchmark_blur(size=(2048, 2048), iterations=10):
    # Fixed Kernel Size for Fair Comparison
    # fnblur is optimized for a specific kernel (approx radius 5 / diam 11)
    kernel_size = 11
    radius = 5

    print(f"Benchmarking Image Size: {size}, Kernel Size: {kernel_size}")

    # Prepare Data
    img_np = np.random.randint(0, 255, size + (4,), dtype=np.uint8)
    img_pil = Image.fromarray(img_np, mode="RGBA")

    # Warmup
    fnblur.gaussian(img_np, iterations=1)
    cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
    img_pil.filter(ImageFilter.GaussianBlur(radius=radius))

    # Benchmark fnblur
    start = time.time()
    fnblur.gaussian(img_np, iterations=iterations)
    fnblur_time = (time.time() - start) / iterations
    print(f"fnblur: {fnblur_time:.4f} seconds")

    # Benchmark OpenCV
    start = time.time()
    for _ in range(iterations):
        cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
    opencv_time = (time.time() - start) / iterations
    print(f"OpenCV: {opencv_time:.4f} seconds")

    # Benchmark Pillow
    start = time.time()
    for _ in range(iterations):
        img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    pillow_time = (time.time() - start) / iterations
    print(f"Pillow: {pillow_time:.4f} seconds")

    print("-" * 30)


def benchmark_box(size=(2048, 2048), iterations=10):
    kernel_size = 5  # fnblur box is fixed 5x5
    radius = 2  # fnblur kernel 5 ~ radius 2

    print(f"Benchmarking Box Blur - Image Size: {size}, Kernel Size: {kernel_size}")

    # Prepare Data
    img_np = np.random.randint(0, 255, size + (4,), dtype=np.uint8)
    img_pil = Image.fromarray(img_np, mode="RGBA")

    # Warmup
    fnblur.box(img_np, iterations=1)
    cv2.blur(img_np, (kernel_size, kernel_size))
    img_pil.filter(ImageFilter.BoxBlur(radius))

    # Benchmark fnblur
    start = time.time()
    fnblur.box(img_np, iterations=iterations)
    fnblur_time = (time.time() - start) / iterations
    print(f"fnblur: {fnblur_time:.4f} seconds")

    # Benchmark OpenCV
    start = time.time()
    for _ in range(iterations):
        cv2.blur(img_np, (kernel_size, kernel_size))
    opencv_time = (time.time() - start) / iterations
    print(f"OpenCV: {opencv_time:.4f} seconds")

    # Benchmark Pillow
    start = time.time()
    for _ in range(iterations):
        img_pil.filter(ImageFilter.BoxBlur(radius))
    pillow_time = (time.time() - start) / iterations
    print(f"Pillow: {pillow_time:.4f} seconds")

    print("-" * 30)


if __name__ == "__main__":
    print("=== Gaussian Blur ===")
    benchmark_blur(size=(1024, 1024))
    benchmark_blur(size=(2048, 2048))
    benchmark_blur(size=(4096, 4096))

    print("\n=== Box Blur ===")
    benchmark_box(size=(1024, 1024))
    benchmark_box(size=(2048, 2048))
    benchmark_box(size=(4096, 4096))
