# Marine Debris Detection

Full-stack application for marine pollution detection with advanced geospatial analysis.

## Features

- **Image-based Debris Detection**: Upload aerial/satellite imagery to detect potential marine debris
- **Geographic Heatmap**: Interactive concentration heatmap overlaid on geographic coordinates
- **Priority Zone Ranking**: AI-powered cleanup priority zones with estimated debris density
- **GeoJSON Export**: Export detection data for fleet coordination tools
- **Real-time Analysis**: Process images and generate actionable reports instantly

## Stack

- Frontend: React.js, HTML, CSS, JavaScript
- Backend: Flask (Python) with OpenCV for image processing
- Mapping: Folium for interactive heatmaps
- Analysis: Scikit-learn for density clustering

## Prerequisites

- Python 3.7+
- Node.js (for React frontend)
- pip for Python package management

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install Node.js dependencies (optional, for React frontend):

```bash
cd Frontend
npm install
```

## Run

1. Start the Flask backend server:

```bash
python backend/app.py
```

2. Open your browser to:

```text
http://localhost:5000
```

3. (Optional) Start the React frontend:

```bash
cd Frontend
npm start
```

## Usage

1. **Upload Image**: Select an aerial or satellite image of ocean area
2. **Enter Coordinates**: Provide latitude, longitude, and zoom level for the image location
3. **Process**: Click "Detect Debris" to analyze the image
4. **View Results**:
   - Detection summary with confidence scores
   - Interactive heatmap showing debris concentration
   - Priority zones ranked by cleanup urgency
   - Export data as GeoJSON for fleet tools

## API Endpoints

- `GET /`: Main application interface
- `POST /`: Process uploaded image for debris detection
- `POST /export_geojson`: Export detection results as GeoJSON

## Notes

- Geographic coordinates are required for accurate mapping and analysis
- Heatmap and priority zones require additional Python packages (folium, scikit-learn)
- The application gracefully degrades if optional packages are not installed
- Detection algorithm uses OpenCV image processing for blob detection

## Commit change reasons

- Commit 1:
  A basic framework is made and the basse modal is created
