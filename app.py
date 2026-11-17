import cv2
import base64
import time
import numpy as np

from flask import Flask, render_template, request, jsonify
from retinex import enhance, metrics

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def image_to_base64(image):
    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 80]
    )

    if not success:
        raise ValueError("Image encoding failed")

    encoded = base64.b64encode(buffer).decode("utf-8")
    return "data:image/jpeg;base64," + encoded


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/enhance", methods=["POST"])
def enhance_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Please upload an image"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if original is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Resize large images for faster processing
        height, width = original.shape[:2]
        max_size = 800

        scale = min(1.0, max_size / max(height, width))

        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)

            original = cv2.resize(
                original,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )

        method = request.form.get("method", "msr").lower()

        if method not in ("ssr", "msr"):
            method = "msr"

        try:
            sigma = float(request.form.get("sigma", 80))
        except ValueError:
            sigma = 80

        try:
            clip = float(request.form.get("clip", 1.0))
        except ValueError:
            clip = 1.0

        sigma = float(np.clip(sigma, 10, 200))
        clip = float(np.clip(clip, 0.2, 1.0))

        start_time = time.time()

        enhanced, illumination, reflectance = enhance(
            original,
            method=method,
            sigma=sigma,
            clip=clip
        )

        processing_time = round(time.time() - start_time, 2)

        original_data = image_to_base64(original)
        enhanced_data = image_to_base64(enhanced)

        return jsonify({
            "success": True,
            "original": original_data,
            "input": original_data,
            "enhanced": enhanced_data,
            "illumination": image_to_base64(illumination),
            "reflectance": image_to_base64(reflectance),
            "metrics": {
                "before": metrics(original),
                "after": metrics(enhanced)
            },
            "processing_time": processing_time,
            "method": method.upper()
        })

    except Exception as error:
        app.logger.exception("Image enhancement failed")
        return jsonify({"error": str(error)}), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image is too large. Please upload an image under 10 MB."
    }), 413


if __name__ == "__main__":
    app.run(debug=True)
