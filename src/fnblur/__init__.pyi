import numpy as np
from numpy.typing import NDArray

def gaussian(
    image: NDArray[np.uint8],
    iterations: int = 1,
    threads: int = -1,
    max_threads: int = -1,
) -> NDArray[np.uint8]:
    """
    Apply Gaussian Blur (11-tap kernel). Highly optimized for ARMv8 NEON.

    Args:
        image (NDArray[np.uint8]): Input image (Height, Width, 4). Must be RGBA.
        iterations (int): Number of blur passes. Defaults to 1.
        threads (int): Number of threads to use. -1 for auto-detection.
        max_threads (int): Maximum number of threads to use. -1 for auto-detection.

    Returns:
        NDArray[np.uint8]: Blurred image (same shape as input).
    """
    ...

def masked(
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    iterations: int = 1,
    threads: int = -1,
    max_threads: int = -1,
) -> NDArray[np.uint8]:
    """
    Apply Gaussian Blur only where the mask is white (255).

    Args:
        image (NDArray[np.uint8]): Input image (Height, Width, 4). Must be RGBA.
        mask (NDArray[np.uint8]): Blending mask (Height, Width). Grayscale.
        iterations (int): Blur intensity.
        threads (int): -1 for auto. Uses all available cores.
        max_threads (int): -1 for auto. Maximum number of threads to use.

    Returns:
        NDArray[np.uint8]: Selective blurred image.
    """
    ...

def alpha(
    image: NDArray[np.uint8],
    iterations: int = 1,
    threads: int = -1,
    max_threads: int = -1,
) -> NDArray[np.uint8]:
    """
    Smart Blur: Uses the image's own Alpha channel as the mask.
    Alpha 0 = Sharp (Original).
    Alpha 255 = Blurred.

    Args:
        image (NDArray[np.uint8]): Input image (H, W, 4). Alpha is treated as mask.
        iterations (int): Blur intensity.
        threads (int): -1 for auto. Uses all available cores.
        max_threads (int): -1 for auto. Maximum number of threads to use.

    Returns:
        NDArray[np.uint8]: Blended image.
    """
    ...

def box(
    image: NDArray[np.uint8],
    iterations: int = 1,
    threads: int = -1,
    max_threads: int = -1,
) -> NDArray[np.uint8]:
    """
    Apply Box Blur (5x5 kernel). Simpler look than Gaussian.

    Args:
        image (NDArray[np.uint8]): Input image (H, W, 4).
        iterations (int): Blur intensity.
        threads (int): -1 for auto. Uses all available cores.
        max_threads (int): -1 for auto. Maximum number of threads to use.

    Returns:
        NDArray[np.uint8]: Blurred image.
    """
    ...
