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

DEBRIS_TYPE_LABELS = {
    "plastic_container": "Plastic Container",
    "plastic_fragment": "Plastic Fragment",
    "fishing_net_or_rope": "Fishing Net / Rope",
    "wood_fragment": "Wood Fragment",
    "foam_piece": "Foam Piece",
    "buoy_or_rubber": "Buoy / Rubber",
    "metal_scrap": "Metal Scrap",
    "general_debris": "General Debris"
}

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
                image = Image.open(io.BytesIO(file.read()))
                image = image.convert("RGB")
                image_np = np.array(image)

                # Enhanced image preprocessing for accurate debris detection
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

                # Enhanced preprocessing for marine debris detection
                # Apply CLAHE for better contrast in marine environments
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)

                # Advanced noise reduction with multiple filters
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
                gray = cv2.medianBlur(gray, 3)  # Additional noise reduction

                # Multi-scale edge detection with improved parameters and hysteresis
                edges1 = cv2.Canny(gray, 25, 75)   # Fine edges
                edges2 = cv2.Canny(gray, 40, 120)  # Medium edges
                edges3 = cv2.Canny(gray, 60, 180)  # Strong edges
                edges = cv2.bitwise_or(edges1, edges2)
                edges = cv2.bitwise_or(edges, edges3)

                # Enhanced morphological operations for debris region formation
                kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_fill, iterations=2)
                filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_fill, iterations=1)

                # Advanced thresholding with multiple techniques
                blur_fine = cv2.GaussianBlur(gray, (3, 3), 0)
                blur_medium = cv2.GaussianBlur(gray, (5, 5), 0)
                blur_coarse = cv2.GaussianBlur(gray, (9, 9), 0)

                # Otsu's thresholding with different scales
                _, thresh_otsu_fine = cv2.threshold(blur_fine, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                _, thresh_otsu_medium = cv2.threshold(blur_medium, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                _, thresh_otsu_coarse = cv2.threshold(blur_coarse, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Adaptive thresholding with optimized parameters
                thresh_adaptive1 = cv2.adaptiveThreshold(blur_fine, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV, 11, 3)
                thresh_adaptive2 = cv2.adaptiveThreshold(blur_medium, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV, 21, 6)
                thresh_adaptive3 = cv2.adaptiveThreshold(blur_coarse, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV, 31, 9)

                # Combine all thresholding results with weighted approach
                thresh_combined = cv2.bitwise_or(thresh_otsu_fine, thresh_otsu_medium)
                thresh_combined = cv2.bitwise_or(thresh_combined, thresh_otsu_coarse)
                thresh_combined = cv2.bitwise_or(thresh_combined, thresh_adaptive1)
                thresh_combined = cv2.bitwise_or(thresh_combined, thresh_adaptive2)
                thresh_combined = cv2.bitwise_or(thresh_combined, thresh_adaptive3)
                thresh_combined = cv2.bitwise_or(thresh_combined, filled)

                # Advanced morphological operations for precise debris extraction
                kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                thresh = cv2.morphologyEx(thresh_combined, cv2.MORPH_OPEN, kernel_clean, iterations=1)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_clean, iterations=1)

                # Remove small noise with optimized area filtering and shape analysis
                contours_noise, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                thresh_clean = np.zeros_like(thresh)
                for cnt in contours_noise:
                    area_noise = cv2.contourArea(cnt)
                    if area_noise > 150:  # Higher threshold for better accuracy
                        # Additional shape filtering for noise removal
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0:
                            circularity_noise = 4 * np.pi * area_noise / (perimeter * perimeter)
                            if circularity_noise < 0.95:  # Avoid perfect circles (likely artifacts)
                                cv2.drawContours(thresh_clean, [cnt], -1, 255, -1)

                thresh = thresh_clean

                # Find contours with high precision approximation
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
                                "type_label": DEBRIS_TYPE_LABELS.get(debris_type, debris_type.replace('_', ' ').title()),
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

                # Prepare final results
                if len(detections) == 0:
                    results = {
                        "total_detections": 0,
                        "detections": [],
                        "types_found": {},
                        "removal_guidance": {},
                        "message": "No debris detected - clean water body"
                    }
                else:
                    detected_types = {
                        debris_type: DEBRIS_TYPE_LABELS.get(debris_type, debris_type.replace('_', ' ').title())
                        for debris_type in debris_type_counts
                    }
                    results = {
                        "total_detections": len(detections),
                        "detections": detections,
                        "types_found": debris_type_counts,
                        "detected_types": detected_types,
                        "removal_guidance": {debris_type: get_removal_guidance(debris_type) for debris_type in debris_type_counts}
                    }

                results["supported_types"] = DEBRIS_TYPE_LABELS
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
    if w < 8 or h < 8 or aspect_ratio > 6.0 or aspect_ratio < 0.15:
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

    # Skip contours with very low solidity (likely fragmented water patterns)
    if solidity < 0.6:
        return None

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

    # Enhanced color analysis with multiple regions
    roi = image[y:y+h, x:x+w]
    if roi.size > 0:
        # Full ROI color
        avg_color = cv2.mean(roi)[:3]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_hsv = cv2.mean(hsv_roi)[:3]

        # Center region color (more representative of object core)
        center_size = max(6, min(w, h) // 4)  # Larger center region for better analysis
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

        # Calculate color variance for texture analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness_std = np.std(gray_roi)
        hsv_std = np.std(hsv_roi.reshape(-1, 3), axis=0)

        # Edge sharpness analysis (gradient magnitude)
        sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_sharpness = np.mean(gradient_magnitude)

        # Color histogram analysis for debris characteristics
        hist_h = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])

        # Normalize histograms
        hist_h = hist_h / hist_h.sum() if hist_h.sum() > 0 else hist_h
        hist_s = hist_s / hist_s.sum() if hist_s.sum() > 0 else hist_s
        hist_v = hist_v / hist_v.sum() if hist_v.sum() > 0 else hist_v

        # Calculate entropy for texture complexity
        entropy_h = -np.sum(hist_h * np.log2(hist_h + 1e-10))
        entropy_s = -np.sum(hist_s * np.log2(hist_s + 1e-10))
        entropy_v = -np.sum(hist_v * np.log2(hist_v + 1e-10))
    else:
        avg_color = (0, 0, 0)
        avg_hsv = (0, 0, 0)
        center_color = (0, 0, 0)
        center_hsv = (0, 0, 0)
        brightness_std = 0
        hsv_std = np.array([0, 0, 0])
        edge_sharpness = 0
        entropy_h = entropy_s = entropy_v = 0

    # Strict color filtering for water-like features with improved logic
    hue, saturation, value = avg_hsv
    center_hue, center_sat, center_val = center_hsv

    # Reject water-like colors (blue/cyan with low saturation and uniformity)
    if ((90 <= hue <= 150 and saturation < 35 and brightness_std < 18) or
        (90 <= center_hue <= 150 and center_sat < 35 and hsv_std[1] < 12)):
        return None

    # Reject over/under exposed areas (but allow some range for debris)
    if ((value < 40 or value > 220) and (center_val < 40 or center_val > 220)):
        return None

    # Reject very desaturated uniform regions (water, sky, reflections)
    if saturation < 15 and center_sat < 15 and hsv_std[1] < 10:
        return None

    # Additional texture-based filtering with edge sharpness
    if brightness_std < 15 and hsv_std[1] < 10 and edge_sharpness < 20:
        return None

    # Reject contours with too low entropy (uniform regions)
    if entropy_h < 2.0 and entropy_s < 2.0 and entropy_v < 2.0:
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
        'brightness_std': brightness_std,
        'saturation_std': hsv_std[1],
        'value_std': hsv_std[2],
        'edge_sharpness': edge_sharpness,
        'entropy_h': entropy_h,
        'entropy_s': entropy_s,
        'entropy_v': entropy_v,
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
    center_hsv = properties['center_hsv']
    brightness_std = properties.get('brightness_std', 0)
    saturation_std = properties.get('saturation_std', 0)
    edge_sharpness = properties.get('edge_sharpness', 0)
    entropy_h = properties.get('entropy_h', 0)
    entropy_s = properties.get('entropy_s', 0)
    entropy_v = properties.get('entropy_v', 0)

    # Use center color for more accurate classification
    hue, saturation, value = center_hsv

    # Size-based pre-classification with refined thresholds
    if area < 400:
        size_category = "small"
    elif area < 1200:
        size_category = "medium"
    else:
        size_category = "large"

    # Plastic containers: rectangular, solid, medium to large with enhanced criteria
    if (size_category in ["medium", "large"] and
        0.65 <= aspect_ratio <= 1.5 and
        solidity > 0.78 and
        circularity < 0.65 and
        eccentricity < 0.65 and
        saturation > 25 and
        brightness_std > 15 and
        edge_sharpness > 25 and  # Sharp edges for plastic
        entropy_s > 3.0):  # Color complexity
        return "plastic_container"

    # Fishing gear: long/thin or tangled shapes with improved detection
    if ((aspect_ratio > 2.8 and solidity > 0.45) or  # Long and thin
        (aspect_ratio < 0.35 and solidity < 0.75) or  # Very wide
        (solidity < 0.55 and eccentricity > 0.75) or  # Tangled/irregular
        (circularity < 0.5 and saturation_std > 20)):   # Textured irregular shapes
        if edge_sharpness > 20 and entropy_h > 2.5:  # Has defined edges and color variation
            return "fishing_net_or_rope"

    # Wood fragments: brown tones, irregular shapes with enhanced color detection
    if (size_category in ["medium", "large"] and
        0.45 <= aspect_ratio <= 2.2 and
        solidity > 0.65 and
        circularity < 0.7 and
        8 <= hue <= 45 and saturation < 65 and  # Brown hue range
        brightness_std > 20 and
        edge_sharpness > 30 and  # Wood has distinct edges
        entropy_v > 3.5):  # Brightness variation in wood
        return "wood_fragment"

    # Foam pieces: light colors, often rectangular with enhanced criteria
    if (aspect_ratio > 1.6 and
        value > 165 and saturation < 35 and
        solidity > 0.55 and
        circularity < 0.75 and
        brightness_std > 10 and
        edge_sharpness > 15 and
        entropy_s < 2.5):  # Low saturation entropy for foam
        return "foam_piece"

    # Buoys or rubber: circular/oval, high solidity with enhanced shape detection
    if (0.75 <= aspect_ratio <= 1.35 and
        circularity > 0.55 and
        solidity > 0.82 and
        eccentricity < 0.55 and
        saturation > 20 and
        brightness_std > 12 and
        edge_sharpness > 18):
        return "buoy_or_rubber"

    # Metal scrap: high contrast, sharp edges with enhanced detection
    if (solidity > 0.87 and
        saturation > 55 and
        value > 135 and
        circularity < 0.65 and
        brightness_std > 25 and
        edge_sharpness > 35 and  # Very sharp edges for metal
        entropy_v > 4.0):  # High brightness variation
        return "metal_scrap"

    # Plastic fragments: various shapes, bright colors with enhanced criteria
    if (size_category in ["small", "medium"] and
        saturation > 28 and value > 95 and
        solidity > 0.55 and
        brightness_std > 12 and
        edge_sharpness > 20 and
        entropy_s > 2.8):  # Color complexity for plastic
        return "plastic_fragment"

    # General debris: fallback for unclassified items with stricter criteria
    if (area > 180 and solidity > 0.45 and brightness_std > 8 and
        edge_sharpness > 15 and
        (entropy_h > 2.2 or entropy_s > 2.2 or entropy_v > 2.2)):
        return "general_debris"

    return "unknown"


def calculate_confidence(properties, debris_type):
    """Calculate confidence score based on advanced property analysis."""
    base_confidence = 0.75  # Higher base confidence with enhanced analysis

    area = properties['area']
    aspect_ratio = properties['aspect_ratio']
    circularity = properties['circularity']
    solidity = properties['solidity']
    eccentricity = properties['eccentricity']
    center_hsv = properties['center_hsv']
    brightness_std = properties.get('brightness_std', 0)
    saturation_std = properties.get('saturation_std', 0)
    edge_sharpness = properties.get('edge_sharpness', 0)
    entropy_h = properties.get('entropy_h', 0)
    entropy_s = properties.get('entropy_s', 0)
    entropy_v = properties.get('entropy_v', 0)

    hue, saturation, value = center_hsv

    # Size-based confidence adjustments
    if debris_type == "plastic_container":
        if 0.7 <= aspect_ratio <= 1.4 and solidity > 0.8:
            base_confidence += 0.15
        if circularity < 0.6 and eccentricity < 0.6:
            base_confidence += 0.06
        if saturation > 28 and brightness_std > 20:
            base_confidence += 0.05
        if edge_sharpness > 30 and entropy_s > 3.5:
            base_confidence += 0.08  # Strong edge and color complexity match

    elif debris_type == "fishing_net_or_rope":
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            base_confidence += 0.14
        if solidity < 0.7 and eccentricity > 0.75:
            base_confidence += 0.10
        if circularity < 0.55 and saturation_std > 15:
            base_confidence += 0.06
        if edge_sharpness > 25 and entropy_h > 3.0:
            base_confidence += 0.07

    elif debris_type == "wood_fragment":
        if 0.6 <= aspect_ratio <= 1.9 and solidity > 0.68:
            base_confidence += 0.14
        if 10 <= hue <= 42 and saturation < 70:
            base_confidence += 0.16  # Strong color match
        if circularity < 0.7 and brightness_std > 25:
            base_confidence += 0.05
        if edge_sharpness > 35 and entropy_v > 4.0:
            base_confidence += 0.08

    elif debris_type == "buoy_or_rubber":
        if 0.78 <= aspect_ratio <= 1.25:
            base_confidence += 0.14
        if circularity > 0.58 and solidity > 0.82:
            base_confidence += 0.10
        if eccentricity < 0.5 and saturation > 25:
            base_confidence += 0.06
        if edge_sharpness > 20 and entropy_s > 2.5:
            base_confidence += 0.07

    elif debris_type == "foam_piece":
        if value > 175 and saturation < 32:
            base_confidence += 0.20  # Strong color match
        if aspect_ratio > 1.4 and brightness_std > 15:
            base_confidence += 0.06
        if edge_sharpness > 18 and entropy_s < 2.8:
            base_confidence += 0.08

    elif debris_type == "metal_scrap":
        if solidity > 0.88 and saturation > 58:
            base_confidence += 0.16
        if value > 140 and circularity < 0.65:
            base_confidence += 0.08
        if brightness_std > 30:
            base_confidence += 0.05
        if edge_sharpness > 40 and entropy_v > 4.5:
            base_confidence += 0.10  # Very strong metal characteristics

    elif debris_type == "plastic_fragment":
        if 220 <= area <= 1600:
            base_confidence += 0.10
        if saturation > 32 and value > 100:
            base_confidence += 0.08
        if brightness_std > 18:
            base_confidence += 0.05
        if edge_sharpness > 25 and entropy_s > 3.2:
            base_confidence += 0.07

    elif debris_type == "general_debris":
        if area > 250 and solidity > 0.5:
            base_confidence += 0.06
        if brightness_std > 15:
            base_confidence += 0.04
        if edge_sharpness > 18 and (entropy_h > 2.5 or entropy_s > 2.5 or entropy_v > 2.5):
            base_confidence += 0.08

    # Advanced quality adjustments based on all new features
    if solidity > 0.82:
        base_confidence += 0.03
    if saturation > 25:
        base_confidence += 0.02
    if brightness_std > 20:
        base_confidence += 0.02
    if edge_sharpness > 25:
        base_confidence += 0.03
    if entropy_h > 3.0 or entropy_s > 3.0 or entropy_v > 3.0:
        base_confidence += 0.02

    return min(base_confidence, 0.98)  # Cap at 98%
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
