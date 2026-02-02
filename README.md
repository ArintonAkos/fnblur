# fnblur

**fnblur** is an ultra-fast, lightweight Gaussian Blur library optimized for ARMv8 NEON architectures (Apple Silicon, Raspberry Pi 4/5, Jetson). It exposes a high-performance C++ implementation to Python via Pybind11.

## Features

- **Blazing Fast**: Hand-tuned ARM NEON intrinsics for maximum throughput.
- **Pythonic**: Seamless integration with NumPy arrays.
- **Lightweight**: minimal dependencies, just needs a C++ compiler.

## Installation

### Prerequisites

- Python 3.8+
- C++17 compliant compiler
- macOS (Apple Silicon recommended) or Linux ARM64

### From Source

```bash
pip install .
```

## Usage

```python
import cv2
import fnblur

# Load an image as a NumPy array (H, W, 3)
img = cv2.imread("image.jpg")

# Apply Gaussian Blur
# Optimized for Kernel Size 11 (Radius 5)
result = fnblur.gaussian(img, iterations=1)

cv2.imwrite("blurred.jpg", result)
```

## Benchmarks

_Hardware: Apple M1 Pro_

| Image Size | Kernel Size | fnblur (s) | OpenCV (s) | Pillow (s) | Speedup vs OpenCV |
| ---------- | ----------- | ---------- | ---------- | ---------- | ----------------- |
| 1024x1024  | 11          | 0.0005     | 0.0010     | 0.0062     | **2.0x**          |
| 2048x2048  | 11          | 0.0010     | 0.0031     | 0.0268     | **3.1x**          |
| 4096x4096  | 11          | 0.0049     | 0.0130     | 0.1819     | **2.6x**          |

_(Run `python benchmarks/benchmark.py` to generate your own results)_

## Architecture

See [docs/architecture.md](docs/architecture.md) for details on the internal SIMD implementation.

## License

MIT
