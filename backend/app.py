from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import io
import json
from io import BytesIO

# Try to import optional packages
try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import geojson
    GEOJSON_AVAILABLE = True
except ImportError:
    GEOJSON_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None
    heatmap_html = None
    priority_zones = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image uploaded. Please choose an image file."
        else:
            file = request.files["image"]
            # Get geographic coordinates from form
            lat = float(request.form.get("latitude", 0))
            lng = float(request.form.get("longitude", 0))
            zoom = int(request.form.get("zoom", 15))

            try:
                image = Image.open(io.BytesIO(file.read())).convert("RGB")
                image_np = np.array(image)

                # Image processing for debris detection
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detections = []
                detection_coords = []
                debris_type_counts = {}

                height, width = image_np.shape[:2]

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 100:
                        x, y, w, h = cv2.boundingRect(cnt)

                        # Convert pixel coordinates to geographic coordinates
                        # This is a simplified conversion - in reality, you'd use proper georeferencing
                        pixel_lat = lat - (y / height) * 0.01  # Rough approximation
                        pixel_lng = lng + (x / width) * 0.01   # Rough approximation

                        debris_type = classify_debris(area, w, h)
                        debris_type_counts[debris_type] = debris_type_counts.get(debris_type, 0) + 1

                        detections.append({
                            "class": "potential_debris",
                            "type": debris_type,
                            "confidence": 0.75,
                            "bbox": [x, y, x + w, y + h],
                            "latitude": pixel_lat,
                            "longitude": pixel_lng,
                            "area": area
                        })

                        detection_coords.append([pixel_lat, pixel_lng])

                # Generate heatmap
                if detection_coords and FOLIUM_AVAILABLE:
                    heatmap_html = generate_heatmap(detection_coords, lat, lng, zoom)
                else:
                    heatmap_html = None

                # Generate priority zones
                if detection_coords and SKLEARN_AVAILABLE:
                    priority_zones = generate_priority_zones(detection_coords)
                else:
                    priority_zones = []

                results = {
                    "total_detections": len(detections),
                    "detections": detections,
                    "types_found": debris_type_counts,
                    "map": f"Detected {len(detections)} debris items in area centered at {lat}, {lng}",
                    "actionable_report": f"Detected {len(detections)} potential debris items. {len(priority_zones) if priority_zones else 0} priority zones identified for cleanup.",
                    "heatmap_available": heatmap_html is not None,
                    "geojson_available": True
                }
            except Exception as exc:
                error = f"Unable to process the image: {exc}"

    return render_template("index.html", results=results, error=error, heatmap_html=heatmap_html, priority_zones=priority_zones)

def classify_debris(area, width, height):
    """Classify debris into a rough type based on size and shape."""
    aspect_ratio = width / height if height > 0 else 1
    if area > 2000 and aspect_ratio > 1.5:
        return "plastic_container"
    if area > 1500 and aspect_ratio < 0.6:
        return "fishing_net"
    if area > 800:
        return "wood_fragment"
    if area > 300:
        return "plastic_fragment"
    return "general_debris"


def generate_heatmap(detection_coords, center_lat, center_lng, zoom):
    """Generate a folium heatmap with debris concentration overlay"""
    if not FOLIUM_AVAILABLE:
        return None

    # Create base map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom)

    # Add heatmap layer
    HeatMap(detection_coords).add_to(m)

    # Add markers for individual detections
    for coord in detection_coords:
        folium.CircleMarker(
            location=coord,
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup='Potential Debris'
        ).add_to(m)

    # Save map to HTML string
    return m._repr_html_()

def generate_priority_zones(detection_coords):
    """Generate priority zones based on debris density clustering"""
    if not SKLEARN_AVAILABLE:
        return []

    if len(detection_coords) < 2:
        return []

    # Convert coordinates to numpy array
    coords = np.array(detection_coords)

    # Normalize coordinates for clustering
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # Perform density-based clustering
    db = DBSCAN(eps=0.3, min_samples=2).fit(coords_scaled)

    # Calculate zone priorities
    zones = []
    unique_labels = set(db.labels_)
    for label in unique_labels:
        if label == -1:  # Noise points
            continue

        mask = db.labels_ == label
        zone_coords = coords[mask]
        density = len(zone_coords)

        # Calculate zone center
        center_lat = np.mean(zone_coords[:, 0])
        center_lng = np.mean(zone_coords[:, 1])

        # Calculate zone bounds
        lat_min, lat_max = np.min(zone_coords[:, 0]), np.max(zone_coords[:, 0])
        lng_min, lng_max = np.min(zone_coords[:, 1]), np.max(zone_coords[:, 1])

        zones.append({
            "zone_id": f"Zone_{label + 1}",
            "center_lat": center_lat,
            "center_lng": center_lng,
            "density": density,
            "bounds": [lat_min, lng_min, lat_max, lng_max],
            "priority_score": density * 1.5  # Simple priority calculation
        })

    # Sort by priority score (highest first)
    zones.sort(key=lambda x: x["priority_score"], reverse=True)

    return zones

@app.route("/export_geojson", methods=["POST"])
def export_geojson():
    """Export detection results as GeoJSON for fleet coordination"""
    if not GEOJSON_AVAILABLE:
        return jsonify({"error": "GeoJSON package not available"}), 500

    data = request.get_json()

    if not data or "detections" not in data:
        return jsonify({"error": "No detection data provided"}), 400

    detections = data["detections"]

    # Create GeoJSON features
    features = []
    for detection in detections:
        if "latitude" in detection and "longitude" in detection:
            feature = geojson.Feature(
                geometry=geojson.Point((detection["longitude"], detection["latitude"])),
                properties={
                    "class": detection.get("class", "debris"),
                    "type": detection.get("type", "unknown"),
                    "confidence": detection.get("confidence", 0.0),
                    "area": detection.get("area", 0),
                    "bbox": detection.get("bbox", [])
                }
            )
            features.append(feature)

    # Create feature collection
    feature_collection = geojson.FeatureCollection(features)

    # Return as downloadable file
    geojson_str = geojson.dumps(feature_collection, indent=2)
    buffer = BytesIO()
    buffer.write(geojson_str.encode('utf-8'))
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/json',
        as_attachment=True,
        download_name='marine_debris_detections.geojson'
    )

if __name__ == "__main__":
    app.run(debug=True)
