export default function PreprocessingView({ steps }) {
  return (
    <details className="card">
      <summary>View preprocessing steps (before → after)</summary>
      <ol className="steps">
        {steps.map((s, i) => (
          <li key={s.name}>
            <figure><img src={s.image} alt={s.name} /><figcaption>{s.name}</figcaption></figure>
            {i < steps.length - 1 && <span className="arrow" aria-hidden="true">→</span>}
          </li>
        ))}
      </ol>
    </details>
  )
}
