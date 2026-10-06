import { useState, useEffect } from "react";
import { getCrops } from "../api";

const STATES = ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Rajasthan"];
const SEASONS = ["Rabi", "Kharif"];

export default function PredictionForm({ position, onSubmit, loading }) {
  const [crops, setCrops] = useState([]);
  const [cropName, setCropName] = useState("");
  const [state, setState] = useState(STATES[0]);
  const [season, setSeason] = useState(SEASONS[0]);
  const [cropsError, setCropsError] = useState(false);
  const [touched, setTouched] = useState(false);

  useEffect(() => {
    getCrops()
      .then((list) => {
        setCrops(list);
        setCropName(list[0]);
      })
      .catch(() => setCropsError(true));
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    setTouched(true);
    if (!position) return;
    onSubmit({
      crop_name: cropName,
      state,
      latitude: position[0],
      longitude: position[1],
      season,
    });
  };

  if (cropsError) {
    return (
      <div className="prediction-form">
        <p className="error">
          Could not reach the backend. Make sure the API server is running, then reload this page.
        </p>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="prediction-form">
      <label>
        Crop
        <select value={cropName} onChange={(e) => setCropName(e.target.value)} disabled={loading || crops.length === 0}>
          {crops.length === 0 && <option>Loading...</option>}
          {crops.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </label>

      <label>
        State
        <select value={state} onChange={(e) => setState(e.target.value)} disabled={loading}>
          {STATES.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </label>

      <label>
        Season
        <select value={season} onChange={(e) => setSeason(e.target.value)} disabled={loading}>
          {SEASONS.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </label>

      <div className={`coords ${touched && !position ? "coords-missing" : ""}`}>
        {position
          ? `Selected: ${position[0].toFixed(3)}, ${position[1].toFixed(3)}`
          : touched
          ? "Please click a location on the map first."
          : "Click the map to select a location"}
      </div>

      <button type="submit" disabled={loading || crops.length === 0}>
        {loading ? (
          <span className="btn-loading">
            <span className="spinner" /> Predicting...
          </span>
        ) : (
          "Get Sowing Depth"
        )}
      </button>
    </form>
  );
}