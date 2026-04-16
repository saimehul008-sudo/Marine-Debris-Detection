from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
from PIL import Image, ExifTags
import io
import json
from io import BytesIO
import base64

# Try to import optional packages
# Try to import optional packages
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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

            try:
                image = Image.open(io.BytesIO(file.read()))
                image = image.convert("RGB")
                image_np = np.array(image)

                # Enhanced image processing for precise debris detection
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)

                # Multi-scale edge detection using Canny
                edges1 = cv2.Canny(gray, 50, 150)  # Fine edges
                edges2 = cv2.Canny(gray, 30, 100)  # Coarse edges
                edges = cv2.bitwise_or(edges1, edges2)  # Combine edge detections

                # Fill edges to create regions
                kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_fill, iterations=2)

                # Multiple threshold approaches with different sensitivities
                blur_fine = cv2.GaussianBlur(gray, (3, 3), 0)
                blur_coarse = cv2.GaussianBlur(gray, (7, 7), 0)

                # Otsu's thresholding on different scales
                _, thresh_otsu_fine = cv2.threshold(blur_fine, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                _, thresh_otsu_coarse = cv2.threshold(blur_coarse, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Adaptive thresholding with different parameters
                thresh_adaptive1 = cv2.adaptiveThreshold(blur_fine, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV, 11, 3)
                thresh_adaptive2 = cv2.adaptiveThreshold(blur_coarse, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV, 21, 5)

                # Combine all thresholding results
                thresh_combined = cv2.bitwise_or(thresh_otsu_fine, thresh_otsu_coarse)
                thresh_combined = cv2.bitwise_or(thresh_combined, thresh_adaptive1)
                thresh_combined = cv2.bitwise_or(thresh_combined, thresh_adaptive2)
                thresh_combined = cv2.bitwise_or(thresh_combined, filled)

                # Advanced morphological operations for precise cleaning
                kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                thresh = cv2.morphologyEx(thresh_combined, cv2.MORPH_OPEN, kernel_clean, iterations=1)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_clean, iterations=1)

                # Remove small noise with area opening
                contours_noise, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                thresh_clean = np.zeros_like(thresh)
                for cnt in contours_noise:
                    if cv2.contourArea(cnt) > 50:  # Filter small noise
                        cv2.drawContours(thresh_clean, [cnt], -1, 255, -1)

                thresh = thresh_clean

                # Find contours with high precision
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

                detections = []
                debris_type_counts = {}

                height, width = image_np.shape[:2]

                for cnt in contours:
                    properties = analyze_contour_properties(cnt, image_np)
                    if properties:
                        x, y, w, h = properties['bbox']

                        debris_type = classify_debris(properties)
                        if debris_type != "unknown":
                            debris_type_counts[debris_type] = debris_type_counts.get(debris_type, 0) + 1
                            debris_removal = get_removal_guidance(debris_type)

                            # Calculate confidence based on classification certainty
                            confidence = calculate_confidence(properties, debris_type)

                            detections.append({
                                "class": "marine_debris",
                                "type": debris_type,
                                "confidence": confidence,
                                "bbox": [x, y, x + w, y + h],
                                "area": properties['area'],
                                "properties": {
                                    "aspect_ratio": round(properties['aspect_ratio'], 2),
                                    "circularity": round(properties['circularity'], 2),
                                    "solidity": round(properties['solidity'], 2)
                                },
                                "removal": debris_removal
                            })

                # Determine if image appears to be clean (no significant debris)
                # Be very conservative - only consider it clean if there are very few high-confidence detections
                is_clean_water = (
                    len(detections) == 0 or  # No detections at all
                    (len(detections) <= 1 and  # At most 1 detection
                     (len(detections) == 0 or
                      (detections[0]['confidence'] < 0.8 and detections[0]['area'] < 1000)))  # Low confidence and small
                )

                if is_clean_water and len(detections) > 0:
                    # Filter out all detections for clean water
                    detections = []

                # Recalculate counts after filtering
                debris_type_counts = {}
                for detection in detections:
                    debris_type = detection['type']
                    debris_type_counts[debris_type] = debris_type_counts.get(debris_type, 0) + 1

                # Generate image-based heatmap
                pixel_coords = [(x + w/2, y + h/2) for x, y, w, h in [d["bbox"] for d in detections]]
                heatmap_image = None
                if pixel_coords and MATPLOTLIB_AVAILABLE and len(detections) > 0:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                    if len(pixel_coords) > 1:
                        sns.kdeplot(x=[p[0] for p in pixel_coords], y=[p[1] for p in pixel_coords], cmap="Reds", fill=True, alpha=0.5, levels=10)
                    else:
                        plt.scatter([p[0] for p in pixel_coords], [p[1] for p in pixel_coords], c='red', s=50, alpha=0.7)
                    plt.axis('off')
                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    heatmap_image = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()

                # Prepare final results
                if len(detections) == 0:
                    results = {
                        "total_detections": 0,
                        "detections": [],
                        "types_found": {},
                        "removal_guidance": {},
                        "heatmap_image": None,
                        "message": "No debris detected - clean water body"
                    }
                else:
                    results = {
                        "total_detections": len(detections),
                        "detections": detections,
                        "types_found": debris_type_counts,
                        "removal_guidance": {debris_type: get_removal_guidance(debris_type) for debris_type in debris_type_counts},
                        "heatmap_image": heatmap_image
                    }
            except Exception as exc:
                error = f"Unable to process the image: {exc}"

    return render_template("index.html", results=results, error=error)

def analyze_contour_properties(contour, image):
    """Analyze detailed properties of a contour for precise classification."""
    area = cv2.contourArea(contour)
    if area < 200:  # Minimum area for meaningful debris
        return None

    # Bounding box and basic properties
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 1.0

    # Skip contours that are too small or have extreme aspect ratios
    if w < 15 or h < 15 or aspect_ratio > 4.0 or aspect_ratio < 0.25:
        return None

    # Advanced shape properties
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None

    # Circularity and compactness
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    # Convex hull analysis
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Shape moments for advanced analysis
    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        return None

    # Hu moments for shape recognition (invariant to scale, rotation, translation)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Normalized central moments
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']

    # Eccentricity (0 = circle, 1 = line)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
    else:
        eccentricity = 0

    # Color analysis with multiple regions
    roi = image[y:y+h, x:x+w]
    if roi.size > 0:
        # Full ROI color
        avg_color = cv2.mean(roi)[:3]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_hsv = cv2.mean(hsv_roi)[:3]

        # Center region color (more representative of object)
        center_size = min(w, h) // 3
        if center_size > 5:
            cx_int, cy_int = int(cx - x), int(cy - y)
            x1 = max(0, cx_int - center_size)
            y1 = max(0, cy_int - center_size)
            x2 = min(w, cx_int + center_size)
            y2 = min(h, cy_int + center_size)

            if x2 > x1 and y2 > y1:
                center_roi = roi[y1:y2, x1:x2]
                if center_roi.size > 0:
                    center_color = cv2.mean(center_roi)[:3]
                    center_hsv_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
                    center_hsv = cv2.mean(center_hsv_roi)[:3]
                else:
                    center_color = avg_color
                    center_hsv = avg_hsv
            else:
                center_color = avg_color
                center_hsv = avg_hsv
        else:
            center_color = avg_color
            center_hsv = avg_hsv
    else:
        avg_color = (0, 0, 0)
        avg_hsv = (0, 0, 0)
        center_color = (0, 0, 0)
        center_hsv = (0, 0, 0)

    # Strict color filtering for water-like features
    hue, saturation, value = avg_hsv
    center_hue, center_sat, center_val = center_hsv

    # Reject water-like colors (blue with low saturation)
    if ((90 <= hue <= 150 and saturation < 25) or
        (90 <= center_hue <= 150 and center_sat < 25)):
        return None

    # Reject over/under exposed areas
    if value < 40 or value > 220 or center_val < 40 or center_val > 220:
        return None

    # Reject desaturated areas
    if saturation < 15 or center_sat < 15:
        return None

    return {
        'area': area,
        'width': w,
        'height': h,
        'aspect_ratio': aspect_ratio,
        'circularity': circularity,
        'solidity': solidity,
        'eccentricity': eccentricity,
        'hu_moments': hu_moments,
        'avg_color': avg_color,
        'avg_hsv': avg_hsv,
        'center_color': center_color,
        'center_hsv': center_hsv,
        'perimeter': perimeter,
        'bbox': (x, y, w, h),
        'centroid': (cx, cy)
    }

def classify_debris(properties):
    """Classify debris into specific types using advanced properties."""
    if not properties:
        return "unknown"

    area = properties['area']
    aspect_ratio = properties['aspect_ratio']
    circularity = properties['circularity']
    solidity = properties['solidity']
    eccentricity = properties['eccentricity']
    avg_hsv = properties['avg_hsv']
    center_hsv = properties['center_hsv']

    # Use center color for more accurate classification
    hue, saturation, value = center_hsv

    # Size-based pre-classification
    if area < 500:
        size_category = "small"
    elif area < 1500:
        size_category = "medium"
    else:
        size_category = "large"

    # Plastic containers: rectangular, solid, medium to large
    if (size_category in ["medium", "large"] and
        0.6 <= aspect_ratio <= 1.6 and
        solidity > 0.75 and
        circularity < 0.7 and
        eccentricity < 0.7 and
        saturation > 20):
        return "plastic_container"

    # Fishing gear: long/thin or tangled shapes
    if ((aspect_ratio > 2.5 and solidity > 0.4) or  # Long and thin
        (aspect_ratio < 0.4 and solidity < 0.7) or  # Very wide
        (solidity < 0.5 and eccentricity > 0.8)):   # Tangled/irregular
        return "fishing_net_or_rope"

    # Wood fragments: brown tones, irregular shapes, medium solidity
    if (size_category in ["medium", "large"] and
        0.5 <= aspect_ratio <= 2.0 and
        solidity > 0.6 and
        circularity < 0.75 and
        5 <= hue <= 45 and saturation < 70):  # Brown hue range
        return "wood_fragment"

    # Foam pieces: light colors, often rectangular or irregular
    if (aspect_ratio > 1.5 and
        value > 160 and saturation < 40 and
        solidity > 0.5):
        return "foam_piece"

    # Buoys or rubber: circular/oval, high solidity, buoyant
    if (0.7 <= aspect_ratio <= 1.4 and
        circularity > 0.5 and
        solidity > 0.8 and
        eccentricity < 0.6):
        return "buoy_or_rubber"

    # Metal scrap: high contrast, sharp edges, reflective
    if (solidity > 0.85 and
        saturation > 50 and
        value > 130 and
        circularity < 0.7):
        return "metal_scrap"

    # Plastic fragments: various shapes, bright colors, small to medium
    if (size_category in ["small", "medium"] and
        saturation > 25 and value > 90 and
        solidity > 0.5):
        return "plastic_fragment"

    # General debris: fallback for unclassified items
    if area > 200 and solidity > 0.4:
        return "general_debris"

    return "unknown"


def calculate_confidence(properties, debris_type):
    """Calculate confidence score based on advanced property analysis."""
    base_confidence = 0.65  # Higher base confidence with better analysis

    area = properties['area']
    aspect_ratio = properties['aspect_ratio']
    circularity = properties['circularity']
    solidity = properties['solidity']
    eccentricity = properties['eccentricity']
    center_hsv = properties['center_hsv']

    hue, saturation, value = center_hsv

    # Size-based confidence adjustments
    if debris_type == "plastic_container":
        if 0.7 <= aspect_ratio <= 1.3 and solidity > 0.8:
            base_confidence += 0.15
        if circularity < 0.6 and eccentricity < 0.6:
            base_confidence += 0.1
        if saturation > 25:
            base_confidence += 0.05

    elif debris_type == "fishing_net_or_rope":
        if aspect_ratio > 2.0 or aspect_ratio < 0.5:
            base_confidence += 0.15
        if solidity < 0.7 and eccentricity > 0.7:
            base_confidence += 0.1
        if circularity < 0.6:
            base_confidence += 0.05

    elif debris_type == "wood_fragment":
        if 0.6 <= aspect_ratio <= 1.8 and solidity > 0.65:
            base_confidence += 0.15
        if 8 <= hue <= 42 and saturation < 75:
            base_confidence += 0.15  # Strong color match
        if circularity < 0.7:
            base_confidence += 0.05

    elif debris_type == "buoy_or_rubber":
        if 0.75 <= aspect_ratio <= 1.25:
            base_confidence += 0.15
        if circularity > 0.55 and solidity > 0.8:
            base_confidence += 0.1
        if eccentricity < 0.5:
            base_confidence += 0.1

    elif debris_type == "foam_piece":
        if value > 170 and saturation < 35:
            base_confidence += 0.2  # Strong color match
        if aspect_ratio > 1.3:
            base_confidence += 0.1

    elif debris_type == "metal_scrap":
        if solidity > 0.87 and saturation > 55:
            base_confidence += 0.15
        if value > 135 and circularity < 0.75:
            base_confidence += 0.1

    elif debris_type == "plastic_fragment":
        if 200 <= area <= 1800:
            base_confidence += 0.1
        if saturation > 30 and value > 95:
            base_confidence += 0.1

    # General quality adjustments
    if solidity > 0.8:
        base_confidence += 0.02
    if saturation > 20:
        base_confidence += 0.02

    return min(base_confidence, 0.95)  # Cap at 95%
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
        "metal_scrap": "Handle sharp metal debris with thick gloves, secure edges, and send to metal recycling facilities.",
        "general_debris": "Collect debris manually with protective equipment; prioritize items that threaten wildlife or navigation."
    }
    return guidance.get(debris_type, guidance["general_debris"])


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

if __name__ == "__main__":
    app.run(debug=True)
