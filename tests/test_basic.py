import pytest
import numpy as np
import fnblur


def test_gaussian_blur_basic():
    # Create a random image (Height, Width, Channels) - Must be RGBA (4 channels)
    img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)

    # Simple call to ensure no crash
    result = fnblur.gaussian(img, 5)

    assert result.shape == img.shape
    assert result.dtype == img.dtype


def test_gaussian_blur_properties():
    # A solid color image strings should remain solid after blur (ignoring boundary effects)
    img = np.full((100, 100, 4), 128, dtype=np.uint8)
    result = fnblur.gaussian(img, 3)

    # Check center pixel
    center_pixel = result[50, 50]
    assert np.all(center_pixel == 128)
