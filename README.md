# Blue Sentinel

Full-stack mission demo for PS11 marine pollution detection.

## Stack

- Frontend: HTML, CSS, vanilla JavaScript
- Backend: PowerShell HTTP server with JSON API

## Run

1. Start the backend server:

```bat
start-server.cmd
```

2. Open this URL in your browser:

```text
http://localhost:5000
```

3. To run the frontend UI, install Node.js if needed, then from the `Frontend` folder:

```bat
npm install
npm start
```

> Note: `npm` / `node` must be installed for the frontend. If missing, install Node.js or use the `Node.js` installer.

## API

- `GET /api/mission/sample`
  Returns sample ocean tiles for the dashboard.

- `POST /api/mission/scan`
  Accepts a JSON body with `tiles` and returns detections, zone rankings, metrics, and GeoJSON.

## Notes

- Uploaded images stay in the frontend and are sent to the backend as tile metadata for demo scanning.
- The backend currently simulates AI detections from the mission brief. It is structured so a real model pipeline can replace the scan engine later.