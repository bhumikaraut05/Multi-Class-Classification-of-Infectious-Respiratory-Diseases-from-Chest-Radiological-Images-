export default function ExplanationViewer({ original, result }) {
  const ex = result.explanation
  return (
    <section className="card" aria-label="Model explainability">
      <h3>Model explainability{ex.title ? ` — ${ex.title}` : ''}</h3>
      <div className="triple">
        <figure><img src={original} alt="Original X-ray" /><figcaption>Original X-ray</figcaption></figure>
        <figure><img src={result.processed_image} alt="Preprocessed model input" /><figcaption>Preprocessed (model input)</figcaption></figure>
        {ex.available && <>
          <figure><img src={ex.heatmap} alt={`${ex.title} heatmap`} /><figcaption>{ex.title} heatmap</figcaption></figure>
          <figure><img src={ex.overlay} alt={`${ex.title} overlay on model input`} /><figcaption>Overlay</figcaption></figure>
        </>}
      </div>
      {!ex.available && <p className="muted">The explanation could not be generated for this image; the prediction itself is unaffected.</p>}
      <p className="note">{ex.note} It is for model interpretability and research only and must not be read as a clinical finding.</p>
    </section>
  )
}
