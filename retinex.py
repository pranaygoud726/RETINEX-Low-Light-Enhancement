import cv2
import numpy as np
import time


def normalise(image, clip=1.0):
    clip = float(np.clip(clip, 0, 5))
    low, high = np.percentile(image, [clip, 100 - clip])

    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    result = (image - low) / (high - low)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def ssr(image, sigma=80):
    image = image.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(image, (0, 0), sigma) + 1.0
    return np.log(image) - np.log(blur)


def msr(image):
    result = np.zeros_like(image, dtype=np.float32)

    for sigma in (15, 80, 250):
        result += ssr(image, sigma) / 3.0

    return result


def metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist(
        [gray], [0], None, [256], [0, 256]
    ).flatten()

    p = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(p * np.log2(p + 1e-10))

    return {
        "brightness": round(float(np.mean(gray)), 2),
        "contrast": round(float(np.std(gray)), 2),
        "entropy": round(float(entropy), 2)
    }


def enhance(img_bgr, method="msr", sigma=80,
            clip=1.0, denoise=True):

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Invalid image")

    start = time.time()

    original = img_bgr.copy()

    # Keep processing fast on Render
    h, w = original.shape[:2]
    scale = min(1.0, 1200 / max(h, w))

    if scale < 1.0:
        original = cv2.resize(
            original, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA
        )

    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    luminance, a, b = cv2.split(lab)

    light = luminance.astype(np.float32) + 1.0

    if method == "ssr":
        retinex = ssr(light, sigma)
    else:
        retinex = msr(light)

    enhanced_l = normalise(retinex, clip)

    # Natural brightness and colour preservation
    enhanced_l = cv2.addWeighted(
        luminance, 0.75,
        enhanced_l, 0.25,
        0
    )

    # Lightweight noise reduction
    if denoise:
        enhanced_l = cv2.bilateralFilter(
            enhanced_l, 5, 20, 20
        )

    enhanced_lab = cv2.merge([enhanced_l, a, b])
    enhanced = cv2.cvtColor(
        enhanced_lab, cv2.COLOR_LAB2BGR
    )

    illumination = cv2.GaussianBlur(
        luminance, (0, 0), 80
    )

    reflectance = light / (
        illumination.astype(np.float32) + 1.0
    )

    reflectance = normalise(
        np.log(reflectance + 1.0), clip
    )

    return {
        "enhanced": enhanced,
        "illumination": illumination,
        "reflectance": reflectance,
        "processing_time": round(time.time() - start, 3)
    }
