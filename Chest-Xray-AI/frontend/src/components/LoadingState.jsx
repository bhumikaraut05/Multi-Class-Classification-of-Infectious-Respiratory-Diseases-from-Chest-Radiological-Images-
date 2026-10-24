export default function LoadingState({ label = 'Analyzing X-ray…' }) {
  return (
    <div className="loading" role="status" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <p>{label}</p>
      <small>Validating → preprocessing → trained model → explanation</small>
    </div>
  )
}
