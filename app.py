from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

from retinex import enhance, metrics

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def image_url(image):
    success, encoded = cv2.imencode(
    ".jpg", image,
    [cv2.IMWRITE_JPEG_QUALITY, 80]
)

    if not success:
        raise ValueError("Image encoding failed")

    data = base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")

    return "data:image/jpeg;base64," + data


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/enhance", methods=["POST"])
def process_image():
    try:
        uploaded = request.files.get("image")

        if not uploaded:
            return jsonify({"error": "Please upload an image"}), 400

        raw = np.frombuffer(uploaded.read(), np.uint8)
        original = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if original is None:
            return jsonify({"error": "Invalid image"}), 400

        # Limit image size before processing
        h, w = original.shape[:2]
        scale = min(1.0, 1200 / max(h, w))

        if scale < 1.0:
            original = cv2.resize(
                original, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA
            )

        method = request.form.get("method", "msr").lower()

        if method not in ("ssr", "msr"):
            return jsonify({"error": "Invalid method"}), 400

        sigma = max(5, min(int(request.form.get("sigma", 80)), 300))
        clip = max(0, min(float(request.form.get("clip", 1)), 5))

        result = enhance(
            original,
            method=method,
            sigma=sigma,
            clip=clip
        )

        enhanced = result["enhanced"]

        return jsonify({
            "original": image_url(original),
            "input": image_url(original),
            "enhanced": image_url(enhanced),
            "illumination": image_url(result["illumination"]),
            "reflectance": image_url(result["reflectance"]),
            "metrics": {
                "original": metrics(original),
                "enhanced": metrics(enhanced)
            },
            "processing_time": result["processing_time"]
        })

    except Exception as error:
        app.logger.exception("Image processing failed")
        return jsonify({"error": str(error)}), 500


@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "error": "Image is too large. Maximum size is 10 MB."
    }), 413


if __name__ == "__main__":
    app.run()
