# RETINEX – Low-Light Image Enhancement

Flask web app that enhances dark images with Single-Scale (SSR) and Multi-Scale (MSR) Retinex.

## Run
```
python -m venv venv && venv\Scripts\activate    (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Structure
- `retinex.py` – SSR, MSR, log transform, illumination-reflectance decomposition, metrics
- `app.py` – Flask routes (`/` page, `/enhance` API)
- `templates/index.html`, `static/style.css`, `static/script.js` – front end

## Algorithm
I = R · L  →  log I = log R + log L  →  log R = log I − log(G_σ * I)
SSR uses one σ; MSR averages SSR over σ = 15, 80, 250. Output is percentile-clipped and stretched to 0–255.
