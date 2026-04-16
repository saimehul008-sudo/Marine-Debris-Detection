from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
from PIL import Image, ExifTags
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
            zoom = int(request.form.get("zoom", 15))

            try:
                image = Image.open(io.BytesIO(file.read()))
                gps_coords = get_exif_location(image)
                lat_str = request.form.get("latitude", "").strip()
                lng_str = request.form.get("longitude", "").strip()

                if gps_coords:
                    lat, lng = gps_coords
                    coordinate_source = "image EXIF"
                elif lat_str and lng_str:
                    lat = float(lat_str)
                    lng = float(lng_str)
                    coordinate_source = "manual input"
                else:
                    raise ValueError("Image has no EXIF GPS data. Provide latitude and longitude or upload a GPS-tagged image.")

                image = image.convert("RGB")
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
                        debris_removal = get_removal_guidance(debris_type)

                        detections.append({
                            "class": "potential_debris",
                            "type": debris_type,
                            "confidence": 0.75,
                            "bbox": [x, y, x + w, y + h],
                            "latitude": pixel_lat,
                            "longitude": pixel_lng,
                            "area": area,
                            "removal": debris_removal
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
                    "removal_guidance": {debris_type: get_removal_guidance(debris_type) for debris_type in debris_type_counts},
                    "coordinate_source": coordinate_source,
                    "coordinates_used": {"latitude": lat, "longitude": lng},
                    "map": f"Detected {len(detections)} debris items in area centered at {lat}, {lng}",
                    "actionable_report": f"Detected {len(detections)} potential debris items. {len(priority_zones) if priority_zones else 0} priority zones identified for cleanup.",
                    "heatmap_available": heatmap_html is not None,
                    "geojson_available": True
                }
            except Exception as exc:
                error = f"Unable to process the image: {exc}"

    return render_template("index.html", results=results, error=error, heatmap_html=heatmap_html, priority_zones=priority_zones)

def classify_debris(area, width, height):
    """Classify debris into more specific types based on area and shape."""
    area = float(area)
    width = float(width)
    height = float(height)
    aspect_ratio = width / height if height > 0 else 1.0

    if area > 2500 and 0.8 <= aspect_ratio <= 1.5:
        return "plastic_container"
    if area > 1800 and aspect_ratio > 1.8:
        return "fishing_net_or_rope"
    if area > 1500 and aspect_ratio < 0.7:
        return "fishing_net_or_rope"
    if area > 1200 and 0.9 <= aspect_ratio <= 1.5:
        return "wood_fragment"
    if area > 700 and aspect_ratio > 1.3:
        return "foam_piece"
    if area > 600 and 0.7 <= aspect_ratio <= 1.2:
        return "buoy_or_rubber"
    if area > 400:
        return "plastic_fragment"
    return "general_debris"


def get_decimal_from_dms(dms, ref):
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1]
    seconds = dms[2][0] / dms[2][1] if len(dms) > 2 else 0
    coord = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ["S", "W"]:
        coord = -coord
    return float(coord)


def get_exif_location(image):
    """Extract latitude and longitude from image EXIF GPS metadata."""
    exif = getattr(image, "_getexif", None)
    if not exif:
        return None

    exif_data = exif()
    if not exif_data:
        return None

    gps_info = None
    for tag, value in exif_data.items():
        decoded = ExifTags.TAGS.get(tag)
        if decoded == "GPSInfo":
            gps_info = value
            break

    if not gps_info:
        return None

    gps_lat = gps_info.get(2)
    gps_lat_ref = gps_info.get(1)
    gps_lng = gps_info.get(4)
    gps_lng_ref = gps_info.get(3)

    if gps_lat and gps_lat_ref and gps_lng and gps_lng_ref:
        lat = get_decimal_from_dms(gps_lat, gps_lat_ref)
        lng = get_decimal_from_dms(gps_lng, gps_lng_ref)
        return lat, lng

    return None


def get_removal_guidance(debris_type):
    """Return a removal recommendation for each debris type."""
    guidance = {
        "plastic_container": "Pick up rigid plastic containers with a net or gloved hand, bag separately, and recycle if possible.",
        "plastic_fragment": "Collect plastic fragments manually with protective gloves and bag small pieces to prevent spread.",
        "fishing_net_or_rope": "Use strong gloves and avoid entanglement; remove nets and ropes in sections and dispose or recycle separately.",
        "wood_fragment": "Gather wood debris carefully, remove any metal hardware, and reuse or dispose according to local rules.",
        "foam_piece": "Skim foam pieces from the surface with nets or baskets and bag them to stop dispersion.",
        "buoy_or_rubber": "Retrieve buoyant rubber debris by hand or hook, keep intact, and transport to proper waste handling.",
        "metal_scrap": "Handle sharp metal debris with thick gloves, secure edges, and send to metal recycling.",
        "general_debris": "Collect debris manually with protective equipment; prioritize items that threaten wildlife or navigation."
    }
    return guidance.get(debris_type, guidance["general_debris"])


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
