export default function PredictionCard({ prediction, confidence, lowConfidence, model }) {
  return (
    <section className="card prediction" aria-label="Prediction">
      <p className="eyebrow">AI classification result</p>
      <h2 className="pred-class">{prediction}</h2>
      <p className="eyebrow">Model-estimated confidence</p>
      <p className="pred-conf">{(confidence * 100).toFixed(2)}%</p>
      <div className="meter" aria-hidden="true"><div style={{ width: `${confidence * 100}%` }} /></div>
      {lowConfidence && <p className="note">Confidence is relatively low — treat this result as uncertain.</p>}
      {prediction === 'Normal' && <p className="note">A Normal result does not rule out disease.</p>}
      {model && <p className="muted small">{model.name} · v{model.version}</p>}
    </section>
  )
}
