from flask import Flask, request, render_template
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image uploaded. Please choose an image file."
        else:
            file = request.files["image"]
            try:
                image = Image.open(io.BytesIO(file.read())).convert("RGB")
                image_np = np.array(image)

                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detections = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 100:
                        x, y, w, h = cv2.boundingRect(cnt)
                        detections.append({
                            "class": "potential_debris",
                            "confidence": 0.75,
                            "bbox": [x, y, x + w, y + h],
                        })

                results = {
                    "total_detections": len(detections),
                    "detections": detections,
                    "map": "Detected debris clusters in the uploaded image area.",
                    "actionable_report": f"Detected {len(detections)} potential debris items. Recommend cleanup deployment in this region.",
                }
            except Exception as exc:
                error = f"Unable to process the image: {exc}"

    return render_template("index.html", results=results, error=error)

if __name__ == "__main__":
    app.run(debug=True)
