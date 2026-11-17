import base64
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from retinex import enhance, metrics

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
MAX_SIDE = 1280  # downscale big images to keep processing fast


def to_b64(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/enhance", methods=["POST"])
def enhance_route():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="No image uploaded."), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="File is not a valid image (use JPG, PNG or BMP)."), 400

    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    method = request.form.get("method", "msr")
    try:
        sigma = float(request.form.get("sigma", 80))
        clip = float(request.form.get("clip", 1.0))
    except ValueError:
        return jsonify(error="Invalid parameter values."), 400
    sigma = max(1, min(sigma, 300))
    clip = max(0, min(clip, 10))

    out = enhance(img, method=method, sigma=sigma, clip=clip)
    return jsonify(
        original=to_b64(img),
        enhanced=to_b64(out["enhanced"]),
        illumination=to_b64(out["illumination"]),
        before=metrics(img),
        after=metrics(out["enhanced"]),
        method=method.upper(),
    )


if __name__ == "__main__":
    app.run(debug=True)
