export default function ErrorMessage({ message, onRetry }) {
  if (!message) return null
  return (
    <div className="alert alert-error" role="alert">
      <strong>Something went wrong</strong>
      <p>{message}</p>
      {onRetry && <button className="btn btn-ghost" onClick={onRetry}>Dismiss</button>}
    </div>
  )
}
