import time
import os
import glob
import cv2
import numpy as np
import fnblur
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter

IMAGES_DIR = "benchmarks/images"
OUTPUT_DIR = "benchmarks/results"

# Ensure output directories exist
for sub in ["raw_speed", "equivalent_blur"]:
    path = os.path.join(OUTPUT_DIR, sub, "outputs")
    if not os.path.exists(path):
        os.makedirs(path)

HAS_MPS = torch.backends.mps.is_available()
DEVICE = torch.device("mps") if HAS_MPS else torch.device("cpu")
print(f"PyTorch Device: {DEVICE}")


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


# ==================================================================================
# SUITE 1: RAW SPEED (Apple-to-Apples)
# Same Kernel Size (11), Same Iterations
# ==================================================================================
def run_raw_speed_test(image_path, filename):
    print(f"\n[Raw Speed] Processing {filename}...")

    # Load Image
    original_cv2 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original_cv2 is None:
        return None

    # Standardize to RGBA / BGRA
    if len(original_cv2.shape) == 2:
        img_bgra = cv2.cvtColor(original_cv2, cv2.COLOR_GRAY2BGRA)
        img_rgba = cv2.cvtColor(original_cv2, cv2.COLOR_GRAY2RGBA)
    elif original_cv2.shape[2] == 3:
        img_bgra = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2BGRA)
        img_rgba = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2RGBA)
    else:
        img_bgra = original_cv2
        img_rgba = cv2.cvtColor(original_cv2, cv2.COLOR_BGRA2RGBA)

    img_pil = Image.fromarray(img_rgba)
    h, w = img_rgba.shape[:2]
    megapixels = (w * h) / 1_000_000

    # --- Settings ---
    ITERATIONS = 20
    # Reduce iterations for huge images
    if megapixels > 20:
        ITERATIONS = 5
    KERNEL_SIZE = 11
    SIGMA = 2.0

    # 1. fnblur
    start = time.time()
    res_fnblur = fnblur.gaussian(img_rgba.copy(), iterations=ITERATIONS)
    t_fnblur = (time.time() - start) / ITERATIONS

    # 2. OpenCV
    start = time.time()
    tmp = img_bgra.copy()
    for _ in range(ITERATIONS):
        tmp = cv2.GaussianBlur(tmp, (KERNEL_SIZE, KERNEL_SIZE), 0)
    t_cv2 = (time.time() - start) / ITERATIONS
    res_cv2 = cv2.cvtColor(tmp, cv2.COLOR_BGRA2RGBA)

    # 3. Pillow
    if megapixels < 25:
        start = time.time()
        tmp_pil = img_pil
        for _ in range(ITERATIONS):
            tmp_pil = tmp_pil.filter(ImageFilter.GaussianBlur(radius=2.0))
        t_pil = (time.time() - start) / ITERATIONS
        res_pil = np.array(tmp_pil)
    else:
        t_pil = 0
        res_pil = None

    # 4. PyTorch
    pt_input = torch.from_numpy(img_rgba).permute(2, 0, 1).float() / 255.0
    if HAS_MPS:
        pt_input = pt_input.to(DEVICE)
        # Warmup
        _ = F.gaussian_blur(pt_input, [KERNEL_SIZE, KERNEL_SIZE], [SIGMA, SIGMA])
        torch.mps.synchronize()
        pt_input_cpu = pt_input.cpu()
    else:
        pt_input_cpu = pt_input

    start = time.time()
    t_tensor = pt_input_cpu.to(DEVICE)
    for _ in range(ITERATIONS):
        t_tensor = F.gaussian_blur(t_tensor, [KERNEL_SIZE, KERNEL_SIZE], [SIGMA, SIGMA])
    if HAS_MPS:
        torch.mps.synchronize()
    t_pt = (time.time() - start) / ITERATIONS

    # Save outputs
    base = os.path.splitext(filename)[0]
    out_path = os.path.join(OUTPUT_DIR, "raw_speed", "outputs", base)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    cv2.imwrite(f"{out_path}/fnblur.png", cv2.cvtColor(res_fnblur, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(f"{out_path}/opencv.png", cv2.cvtColor(res_cv2, cv2.COLOR_RGBA2BGRA))
    if res_pil is not None:
        cv2.imwrite(
            f"{out_path}/pillow.png", cv2.cvtColor(res_pil, cv2.COLOR_RGBA2BGRA)
        )

    pt_res_np = t_tensor.cpu().permute(1, 2, 0).numpy() * 255.0
    pt_res_np = np.clip(pt_res_np, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{out_path}/pytorch.png", cv2.cvtColor(pt_res_np, cv2.COLOR_RGBA2BGRA))

    # Metrics
    speedup_cv2 = t_cv2 / t_fnblur
    speedup_pt = t_pt / t_fnblur
    speedup_pil = t_pil / t_fnblur if t_pil > 0 else 0

    print(f"  fnblur:  {t_fnblur * 1000:.2f} ms")
    print(f"  OpenCV:  {t_cv2 * 1000:.2f} ms")
    print(f"  PyTorch: {t_pt * 1000:.2f} ms")
    if t_pil > 0:
        print(f"  Pillow:  {t_pil * 1000:.2f} ms")

    return {
        "name": filename,
        "mp": megapixels,
        "t_fnblur": t_fnblur,
        "t_cv2": t_cv2,
        "t_pil": t_pil,
        "t_pt": t_pt,
    }


# ==================================================================================
# SUITE 2: EQUIVALENT BLUR (Algorithmic Efficiency)
# fnblur (N iters) vs Single-Pass Large Sigma (OpenCV/PyTorch)
# ==================================================================================
def run_equivalent_blur_test(image_path, filename):
    print(f"\n[Equivalent Blur] Processing {filename}...")

    # Load Image
    original_cv2 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original_cv2 is None:
        return None
    if len(original_cv2.shape) == 2:
        img_bgra = cv2.cvtColor(original_cv2, cv2.COLOR_GRAY2BGRA)
        img_rgba = cv2.cvtColor(original_cv2, cv2.COLOR_GRAY2RGBA)
    elif original_cv2.shape[2] == 3:
        img_bgra = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2BGRA)
        img_rgba = cv2.cvtColor(original_cv2, cv2.COLOR_BGR2RGBA)
    else:
        img_bgra = original_cv2
        img_rgba = cv2.cvtColor(original_cv2, cv2.COLOR_BGRA2RGBA)

    h, w = img_rgba.shape[:2]
    megapixels = (w * h) / 1_000_000

    # --- Settings ---
    # User requested decreasing iterations to check performance/quality balance
    ITERATIONS = 12
    # For massive images (40MP+), reduce further to keep benchmark quick
    if megapixels > 30:
        ITERATIONS = 5

    TARGET_SIGMA = 2.0 * np.sqrt(ITERATIONS)

    # 1. fnblur (Multi-Pass)
    start = time.time()
    res_fnblur = fnblur.gaussian(img_rgba.copy(), iterations=ITERATIONS)
    t_fnblur = time.time() - start  # Total time

    # 2. OpenCV (Single-Pass Large Kernel)
    start = time.time()
    res_cv2_bgra = cv2.GaussianBlur(
        img_bgra.copy(), (0, 0), sigmaX=TARGET_SIGMA, sigmaY=TARGET_SIGMA
    )
    t_cv2 = time.time() - start
    res_cv2 = cv2.cvtColor(res_cv2_bgra, cv2.COLOR_BGRA2RGBA)

    # 3. PyTorch (Single-Pass Large Kernel)
    pt_input = torch.from_numpy(img_rgba).permute(2, 0, 1).float() / 255.0
    k_large = int(2 * np.ceil(3 * TARGET_SIGMA) + 1) | 1

    if HAS_MPS:
        pt_input = pt_input.to(DEVICE)
        _ = F.gaussian_blur(pt_input, [k_large, k_large], [TARGET_SIGMA, TARGET_SIGMA])
        torch.mps.synchronize()
        pt_input_cpu = pt_input.cpu()
    else:
        pt_input_cpu = pt_input

    start = time.time()
    t_tensor = pt_input_cpu.to(DEVICE)
    t_tensor = F.gaussian_blur(
        t_tensor, [k_large, k_large], [TARGET_SIGMA, TARGET_SIGMA]
    )
    if HAS_MPS:
        torch.mps.synchronize()
    pt_res_tensor = t_tensor.cpu().permute(1, 2, 0).numpy() * 255.0
    res_pt = np.clip(pt_res_tensor, 0, 255).astype(np.uint8)
    t_pt = time.time() - start

    # Metrics
    psnr = calculate_psnr(res_cv2, res_fnblur)
    speedup_cv2 = t_cv2 / t_fnblur
    speedup_pt = t_pt / t_fnblur

    print(f"  Target Sigma: {TARGET_SIGMA:.2f}")
    print(f"  fnblur ({ITERATIONS} passes): {t_fnblur:.4f}s")
    print(f"  OpenCV (1 pass):       {t_cv2:.4f}s (Speedup: {speedup_cv2:.2f}x)")
    print(f"  PyTorch (1 pass):      {t_pt:.4f}s (Speedup: {speedup_pt:.2f}x)")
    print(f"  Quality Verification:  {psnr:.2f} dB")

    # Save outputs
    base = os.path.splitext(filename)[0]
    out_path = os.path.join(OUTPUT_DIR, "equivalent_blur", "outputs", base)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    cv2.imwrite(f"{out_path}/fnblur.png", cv2.cvtColor(res_fnblur, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(f"{out_path}/opencv.png", cv2.cvtColor(res_cv2, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(f"{out_path}/pytorch.png", cv2.cvtColor(res_pt, cv2.COLOR_RGBA2BGRA))

    return {
        "name": filename,
        "mp": megapixels,
        "t_fnblur": t_fnblur,
        "t_cv2": t_cv2,
        "t_pt": t_pt,
        "speedup_cv2": speedup_cv2,
        "speedup_pt": speedup_pt,
    }


def print_raw_summary_table(results):
    results.sort(key=lambda x: x["mp"])
    print("\n" + "=" * 85)
    print(" SUITE 1 SUMMARY: RAW SPEED (Time per Iteration in ms)")
    print("-" * 85)
    print(
        f"{'Image':<20} | {'MP':<6} | {'fnblur':<10} | {'OpenCV':<10} | {'PyTorch':<10} | {'Pillow':<10}"
    )
    print("-" * 85)
    for r in results:
        t_fn = r["t_fnblur"] * 1000
        t_cv = r["t_cv2"] * 1000
        t_pt = r["t_pt"] * 1000
        t_pil = r["t_pil"] * 1000 if r["t_pil"] > 0 else 0
        pil_str = f"{t_pil:.2f}" if t_pil > 0 else "N/A"

        print(
            f"{r['name']:<20} | {r['mp']:<6.2f} | {t_fn:<10.2f} | {t_cv:<10.2f} | {t_pt:<10.2f} | {pil_str:<10}"
        )
    print("=" * 85)


def print_equiv_summary_table(results):
    results.sort(key=lambda x: x["mp"])
    print("\n" + "=" * 85)
    print(" SUITE 2 SUMMARY: EQUIVALENT BLUR (Total Time in Seconds)")
    print("-" * 85)
    print(
        f"{'Image':<20} | {'MP':<6} | {'fnblur':<12} | {'OpenCV Speedup':<16} | {'Torch Speedup':<16}"
    )
    print("-" * 85)
    for r in results:
        print(
            f"{r['name']:<20} | {r['mp']:<6.2f} | {r['t_fnblur']:<12.4f} | {r['speedup_cv2']:<16.2f}x | {r['speedup_pt']:<16.2f}x"
        )
    print("=" * 85)


def plot_raw_speed(results):
    results.sort(key=lambda x: x["mp"])
    names = [r["name"] for r in results]
    t_fnblur = [r["t_fnblur"] * 1000 for r in results]
    t_cv2 = [r["t_cv2"] * 1000 for r in results]
    t_pil = [r["t_pil"] * 1000 for r in results]
    t_pt = [r["t_pt"] * 1000 for r in results]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(
        x - 1.5 * width,
        t_fnblur,
        width,
        label="fnblur",
        color="#4CAF50",
        edgecolor="black",
    )
    ax.bar(
        x - 0.5 * width,
        t_cv2,
        width,
        label="OpenCV",
        color="#2196F3",
        edgecolor="black",
    )
    ax.bar(
        x + 0.5 * width,
        t_pt,
        width,
        label="PyTorch",
        color="#E91E63",
        edgecolor="black",
    )
    ax.bar(
        x + 1.5 * width,
        t_pil,
        width,
        label="Pillow",
        color="#FFC107",
        edgecolor="black",
    )

    ax.set_ylabel("Time per Iteration (ms)")
    ax.set_title("Raw Speed Test: Same Kernel (11), Same Iterations")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "raw_speed", "chart.png"), dpi=300)


def plot_equivalent_blur(results):
    results.sort(key=lambda x: x["mp"])
    names = [r["name"] for r in results]
    mps = [r["mp"] for r in results]
    speedup_cv2 = [r["speedup_cv2"] for r in results]
    speedup_pt = [r["speedup_pt"] for r in results]

    plt.figure(figsize=(12, 7))
    plt.plot(
        mps,
        speedup_cv2,
        marker="s",
        linewidth=3,
        color="#2196F3",
        label="vs OpenCV (1-Pass)",
    )
    plt.plot(
        mps,
        speedup_pt,
        marker="^",
        linewidth=3,
        color="#E91E63",
        label="vs PyTorch (1-Pass)",
    )
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Parity")

    plt.title(
        "Real-World Efficiency: fnblur Speedup vs Single-Pass Baselines", fontsize=16
    )
    plt.xlabel("Resolution (Megapixels)")
    plt.ylabel("Speedup (x times faster than Baseline)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "equivalent_blur", "chart.png"), dpi=300)


def run_suite():
    types = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in types:
        files.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    if not files:
        return

    # Suite 1
    print("\n" + "=" * 50)
    print(" SUITE 1: RAW SPEED (Apples-to-Apples)")
    print(" Same Kernel (11), Same Iterations (20)")
    print("=" * 50)
    raw_results = []
    for f in files:
        res = run_raw_speed_test(f, os.path.basename(f))
        if res:
            raw_results.append(res)
    print_raw_summary_table(raw_results)
    plot_raw_speed(raw_results)

    # Suite 2
    print("\n" + "=" * 50)
    print(" SUITE 2: EQUIVALENT BLUR (Algorithmic)")
    print(" fnblur (20 passes) vs Single-Pass Large Sigma")
    print("=" * 50)
    equiv_results = []
    for f in files:
        res = run_equivalent_blur_test(f, os.path.basename(f))
        if res:
            equiv_results.append(res)
    print_equiv_summary_table(equiv_results)
    plot_equivalent_blur(equiv_results)

    print("\nAll Benchmarks Completed.")
    print(
        f"Charts saved to {OUTPUT_DIR}/raw_speed/chart.png and {OUTPUT_DIR}/equivalent_blur/chart.png"
    )


if __name__ == "__main__":
    run_suite()
