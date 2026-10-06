export function ResultsSkeleton() {
  return (
    <div className="results-card skeleton">
      <div className="skeleton-line skeleton-title" />
      <div className="skeleton-row">
        <div className="skeleton-line" />
        <div className="skeleton-line" />
      </div>
      <div className="skeleton-line skeleton-wide" />
      <div className="skeleton-line skeleton-wide" />
    </div>
  );
}

export default function ResultsCard({ result }) {
  if (!result) return null;

  const [safeMin, safeMax] = result.agronomic_safe_range_cm;

  return (
    <div className={`results-card ${result.within_safe_range ? "safe" : "warning"}`}>
      <h2>{result.crop}</h2>
      <div className="depth-row">
        <div>
          <span className="label">AI Recommended</span>
          <span className="value">{result.nn_recommended_depth_cm} cm</span>
        </div>
        <div>
          <span className="label">Rule-Based</span>
          <span className="value">{result.rule_based_depth_cm} cm</span>
        </div>
      </div>
      <p className="range">Agronomic safe range: {safeMin}–{safeMax} cm</p>
      <p className="note">{result.note}</p>
      <div className="meta">
        <span>Soil: {result.derived_soil_texture_class}</span>
        <span>Moisture: {result.estimated_soil_moisture_pct}%</span>
      </div>
    </div>
  );
}