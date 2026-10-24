export default function QualityWarnings({ warnings }) {
  if (!warnings?.length) return null
  return (
    <div className="alert alert-warn" role="status">
      <strong>Image-quality notes</strong>
      <ul>{warnings.map((w) => <li key={w.code}>{w.message}</li>)}</ul>
    </div>
  )
}
