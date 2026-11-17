import cv2
import numpy as np


def normalise(image):
    image = image.astype(np.float32)
    low, high = np.percentile(image, (2, 98))

    if high <= low:
        return np.clip(image, 0, 255).astype(np.uint8)

    image = (image - low) * (255.0 / (high - low))
    return np.clip(image, 0, 255).astype(np.uint8)


def ssr(image, sigma=80):
    image = image.astype(np.float32) + 1.0

    illumination = cv2.GaussianBlur(
        image, (0, 0), sigmaX=float(sigma)
    )

    reflectance = np.log(image) - np.log(illumination + 1.0)

    return reflectance, illumination


def msr(image):
    image = image.astype(np.float32) + 1.0

    scales = (15, 80, 200)
    result = np.zeros_like(image, dtype=np.float32)

    illumination = np.zeros_like(image, dtype=np.float32)

    for sigma in scales:
        blurred = cv2.GaussianBlur(
            image, (0, 0), sigmaX=float(sigma)
        )

        result += np.log(image) - np.log(blurred + 1.0)
        illumination += blurred / len(scales)

    result /= len(scales)

    return result, illumination


def metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return {
        "brightness": round(float(np.mean(gray)), 2),
        "contrast": round(float(np.std(gray)), 2)
    }


def enhance(image, method="msr", sigma=80, clip=1.0):
    # Process only luminance to preserve natural colours
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    y_float = y.astype(np.float32)

    if method.lower() == "ssr":
        retinex, illumination = ssr(y_float, sigma)
    else:
        retinex, illumination = msr(y_float)

    # Normalise Retinex result
    retinex = normalise(retinex)

    # Blend with original brightness for a natural result
    strength = float(np.clip(clip, 0.2, 1.0))
    enhanced_y = (
        y.astype(np.float32) * (1.0 - 0.55 * strength)
        + retinex.astype(np.float32) * (0.55 * strength)
    )

    enhanced_y = np.clip(enhanced_y, 0, 255).astype(np.uint8)

    # Gentle noise reduction
    enhanced_y = cv2.bilateralFilter(
        enhanced_y, 5, 25, 25
    )

    # Restore colour
    enhanced_image = cv2.merge((enhanced_y, cr, cb))
    enhanced_image = cv2.cvtColor(
        enhanced_image, cv2.COLOR_YCrCb2BGR
    )

    # Create display maps
    illumination_map = normalise(np.log1p(illumination))
    reflectance_map = normalise(retinex)

    illumination_map = cv2.cvtColor(
        illumination_map, cv2.COLOR_GRAY2BGR
    )

    reflectance_map = cv2.cvtColor(
        reflectance_map, cv2.COLOR_GRAY2BGR
    )

    return (
        enhanced_image,
        illumination_map,
        reflectance_map
    )
