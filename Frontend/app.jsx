import React, { useState } from "react";

function App() {
  const [image, setImage] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!image) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('http://localhost:5000/detect', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Marine Debris Detection AI Pipeline</h1>
      <p>Upload aerial or satellite ocean imagery to detect and map marine debris.</p>
      <input type="file" accept="image/*" onChange={handleImageChange} />
      <button onClick={handleSubmit} disabled={!image || loading}>
        {loading ? 'Processing...' : 'Detect Debris'}
      </button>
      {results && (
        <div>
          <h2>Results</h2>
          <p>Total Detections: {results.total_detections}</p>
          <ul>
            {results.detections.map((det, idx) => (
              <li key={idx}>
                {det.class} - Confidence: {det.confidence.toFixed(2)} - BBox: {JSON.stringify(det.bbox)}
              </li>
            ))}
          </ul>
          <p>Actionable Report: {results.actionable_report}</p>
        </div>
      )}
    </div>
  );
}

export default App;
