// Compact model summary card built from the real /api/model-info response.
export default function ModelInfo({ info }) {
  if (!info) return null
  const m = info.model
  const rows = m
    ? [['Algorithm', m.name], ['Version', m.version], ['Input', `${m.input_size} × ${m.input_size}`],
       ['Parameters', m.parameters ? m.parameters.toLocaleString() : '—'], ['Explanation', m.explainability.method.replace('_', ' ')]]
    : [['Algorithm', 'Not trained yet'], ['Classes', info.classes.length]]
  return (
    <section className="card" aria-label="Model information">
      <h3>Final model</h3>
      <dl className="kv">{rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
    </section>
  )
}
