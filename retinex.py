import cv2
import numpy as np
import time


# ---------------------------------
# 1. Illumination estimation
# ---------------------------------

def _illumination(channel, sigma):
    sigma = max(1, int(sigma))

    return cv2.GaussianBlur(
        channel,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )


# ---------------------------------
# 2. Single Scale Retinex
# ---------------------------------

def single_scale_retinex(img, sigma=80):
    img = img.astype(np.float32) + 1.0

    blur = _illumination(img, sigma) + 1.0

    return np.log(img) - np.log(blur)


# ---------------------------------
# 3. Multi Scale Retinex
# ---------------------------------

def multi_scale_retinex(
    img,
    sigmas=(15, 80, 250),
    weights=None
):
    if weights is None:
        weights = [1 / len(sigmas)] * len(sigmas)

    result = np.zeros_like(
        img,
        dtype=np.float32
    )

    for sigma, weight in zip(sigmas, weights):
        result += weight * single_scale_retinex(
            img, sigma
        )

    return result


# ---------------------------------
# 4. Safe normalization
# ---------------------------------

def _normalise(log_img, clip=1.0):
    clip = float(np.clip(clip, 0, 5))

    low = np.percentile(log_img, clip)
    high = np.percentile(log_img, 100 - clip)

    if high <= low:
        return np.zeros(log_img.shape, dtype=np.uint8)

    result = np.clip(
        (log_img - low) / (high - low),
        0,
        1
    )

    return (result * 255).astype(np.uint8)


# ---------------------------------
# 5. Noise reduction
# ---------------------------------

def reduce_noise(img, strength=5):
    strength = int(np.clip(strength, 1, 15))

    return cv2.fastNlMeansDenoising(
        img,
        None,
        strength,
        7,
        21
    )


# ---------------------------------
# 6. Main enhancement function
# ---------------------------------

def enhance(
    img_bgr,
    method="msr",
    sigma=80,
    sigmas=(15, 80, 250),
    clip=1.0,
    denoise=True,
    denoise_strength=5
):
    if img_bgr is None:
        raise ValueError("Input image is empty")

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image")

    start_time = time.time()

    original = np.clip(
        img_bgr, 0, 255
    ).astype(np.uint8)

    # Convert to LAB for colour preservation
    lab = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2LAB
    )

    luminance, a, b = cv2.split(lab)

    luminance_float = luminance.astype(np.float32) + 1.0

    # Select Retinex method
    method = method.lower()

    if method == "ssr":
        retinex = single_scale_retinex(
            luminance_float, sigma
        )

    elif method == "msr":
        retinex = multi_scale_retinex(
            luminance_float, sigmas
        )

    else:
        raise ValueError("Method must be ssr or msr")

    # Normalize Retinex result
    retinex_normalized = _normalise(
        retinex, clip
    )

    # ---------------------------------
    # Control excessive brightness
    # ---------------------------------

    # Blend original luminance and Retinex
    # Higher original weight = more natural brightness

    enhanced_l = cv2.addWeighted(
        luminance,
        0.55,
        retinex_normalized,
        0.45,
        0
    )

    # Slightly reduce over-bright appearance
    enhanced_l = np.clip(
        enhanced_l.astype(np.float32) * 0.95,
        0,
        255
    ).astype(np.uint8)

    # ---------------------------------
    # Noise reduction
    # ---------------------------------

    if denoise:
        enhanced_l = reduce_noise(
            enhanced_l,
            denoise_strength
        )

    # ---------------------------------
    # Preserve natural colours
    # ---------------------------------

    enhanced_lab = cv2.merge([
        enhanced_l,
        a,
        b
    ])

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # ---------------------------------
    # Illumination and reflectance
    # ---------------------------------

    illumination = _illumination(
        luminance,
        sigma
    )

    reflectance = luminance.astype(
        np.float32
    ) / (
        illumination.astype(np.float32) + 1.0
    )

    reflectance = _normalise(
        np.log(reflectance + 1.0),
        clip
    )

    processing_time = time.time() - start_time

    return {
        "enhanced": enhanced,
        "illumination": illumination,
        "reflectance": reflectance,
        "processing_time": processing_time
    }


# ---------------------------------
# 7. Image quality metrics
# ---------------------------------

def metrics(img_bgr):
    if img_bgr is None:
        raise ValueError("Input image is empty")

    gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    entropy = float(
        -np.sum(
            np.histogram(
                gray,
                bins=256,
                range=(0, 256),
                density=True
            )[0]
            * np.log2(
                np.histogram(
                    gray,
                    bins=256,
                    range=(0, 256),
                    density=True
                )[0] + 1e-10
            )
        )
    )

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "entropy": round(entropy, 2)
    }