import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random

def apply_brightness(image):
    """Apply random brightness augmentation."""
    brightness = ImageEnhance.Brightness(image)
    brightness_factor = np.random.uniform(0.5, 1.5)  # Random factor between 0.5 and 1.5
    return brightness.enhance(brightness_factor)

def add_noise(image):
    """Add Gaussian noise to the image without clipping negative values prematurely."""
    # Convert image to float32 so that noise can be added correctly.
    np_image = np.array(image, dtype=np.float32)
    # Generate noise from a normal distribution (mean=0, std=15)
    noise = np.random.normal(0, 15, np_image.shape)
    # Add noise, clip to valid [0, 255] range, and convert back to uint8
    noisy_image = Image.fromarray(np.clip(np_image + noise, 0, 255).astype(np.uint8))
    return noisy_image

def add_perspective_change(image):
    """Apply a random perspective change to the image."""
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        A = np.array(matrix, dtype=np.float32)
        B = np.array(pb).reshape(8)
        # Solve for the transformation coefficients
        res = np.linalg.lstsq(A, B, rcond=None)[0]
        return res

    width, height = image.size
    # Use a smaller factor to reduce extreme warping
    factor = np.random.uniform(0.005, 0.03)

    # Randomly decide which perspective transformation to apply.
    if random.choice([True, False]):
        coeffs = find_coeffs(
            [(0, 0), (width, 0), (width, height), (0, height)],
            [(0, 0), (width - width * factor, height * factor),
             (width, height - height * factor), (width * factor, height)]
        )
    else:
        coeffs = find_coeffs(
            [(0, 0), (width, 0), (width, height), (0, height)],
            [(width * factor, 0), (width, height * factor),
             (width - width * factor, height), (0, height - height * factor)]
        )
    perspective_image = image.transform(
        (width, height), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC
    )
    return perspective_image

def add_sharpening(image):
    """Apply an unsharp mask to sharpen the image."""
    return image.filter(ImageFilter.UnsharpMask())

def add_gaussian_blur(image):
    """Apply a Gaussian blur with a milder fixed radius."""
    return image.filter(ImageFilter.GaussianBlur(radius=1))

def add_contrast(image):
    """Randomly change the image contrast."""
    contrast = ImageEnhance.Contrast(image)
    factor = np.random.uniform(0.5, 2)  # Random contrast factor between 0.5 and 2
    return contrast.enhance(factor)

def add_saturation(image):
    """Randomly adjust the image saturation."""
    saturation = ImageEnhance.Color(image)
    factor = np.random.uniform(0.5, 2)  # Random saturation factor between 0.5 and 2
    return saturation.enhance(factor)

def apply_augmentations(image, augmentation_chance=0.05):
    """
    Randomly apply each augmentation independently with a probability of augmentation_chance.
    Adjust augmentation_chance to reduce the frequency of augmentations if needed.
    """
    if np.random.rand() < augmentation_chance:
        image = apply_brightness(image)
    if np.random.rand() < augmentation_chance:
        image = add_noise(image)
    if np.random.rand() < augmentation_chance:
        image = add_perspective_change(image)
    if np.random.rand() < augmentation_chance:
        image = add_sharpening(image)
    if np.random.rand() < augmentation_chance:
        image = add_gaussian_blur(image)
    if np.random.rand() < augmentation_chance:
        image = add_contrast(image)
    if np.random.rand() < augmentation_chance:
        image = add_saturation(image)
    return image