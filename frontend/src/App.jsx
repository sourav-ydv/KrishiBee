import { useState } from "react";
import LocationMap from "./components/LocationMap";
import PredictionForm from "./components/PredictionForm";
import ResultsCard, { ResultsSkeleton } from "./components/ResultsCard";
import { predictByLocation } from "./api";
import "leaflet/dist/leaflet.css";
import "./App.css";

function describeError(err) {
  if (!err.response) {
    return "Can't reach the server. Check that the backend is running and try again.";
  }
  if (err.response.status === 502) {
    return "The live soil/weather lookup timed out. This can happen occasionally — please try again.";
  }
  if (err.response.status === 400) {
    return err.response.data?.detail || "Invalid input. Please check your selections.";
  }
  return "Something went wrong. Please try again.";
}

export default function App() {
  const [position, setPosition] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastPayload, setLastPayload] = useState(null);

  const handleSelect = (lat, lng) => setPosition([lat, lng]);

  const runPrediction = async (payload) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setLastPayload(payload);
    try {
      const data = await predictByLocation(payload);
      setResult(data);
    } catch (err) {
      setError(describeError(err));
    } finally {
      setLoading(false);
    }
  };

  const handleRetry = () => {
    if (lastPayload) runPrediction(lastPayload);
  };

  return (
    <div className="app">
      <header>
        <h1>KrishiBee</h1>
        <p>AI-powered sowing depth advisory</p>
      </header>

      <div className="main-grid">
        <div className="map-panel">
          <LocationMap position={position} onSelect={handleSelect} />
        </div>

        <div className="side-panel">
          <PredictionForm position={position} onSubmit={runPrediction} loading={loading} />

          {error && (
            <div className="error-box">
              <p className="error">{error}</p>
              <button className="retry-btn" onClick={handleRetry}>Retry</button>
            </div>
          )}

          {loading && <ResultsSkeleton />}
          {!loading && <ResultsCard result={result} />}
        </div>
      </div>
    </div>
  );
}