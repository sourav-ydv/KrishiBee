export default function DiagnosisResult({ result, onClose }) {
  const { data, url } = result;
  return (
    <div className="diagnosis">
      <div className="diagnosis-head">
        <img src={url} alt="Checked field photo" />
        <div>
          <h4>Photo check</h4>
          {data.observations && <p className="diagnosis-seen">What the AI sees: {data.observations}</p>}
        </div>
        <button type="button" className="link-btn" onClick={onClose}>Close</button>
      </div>

      <p className="diagnosis-message">{data.message}</p>

      {data.candidates.map((c, i) => (
        <div key={c.id} className="candidate">
          <div className="candidate-top">
            <span className="candidate-rank">{i === 0 ? "Closest match" : "Also possible"}</span>
            <span className="candidate-cat">{c.category}</span>
          </div>
          <h5>{c.name}</h5>
          <p><strong>Compare with:</strong> {c.look_for}</p>
          {c.conditions && <p><strong>Favoured by:</strong> {c.conditions}</p>}
          <p><strong>What PAU advises:</strong> {c.manage}</p>
          {c.ask_kvk && <p className="candidate-kvk">{c.ask_kvk}</p>}
          <a href={c.source_url} target="_blank" rel="noreferrer">{c.source_title}</a>
        </div>
      ))}

      <p className="diagnosis-warning">
        The AI can be wrong. In a small pilot on public photos from the UK and Ireland it named the right disease first
        in about 2 of 3 cases, and it has not been tested in Punjab fields. Use this only as a prompt to look closer,
        and confirm with your KVK before spraying anything. The photo is sent to Groq's AI service for analysis
        and is not stored by KrishiBee.
      </p>
    </div>
  );
}