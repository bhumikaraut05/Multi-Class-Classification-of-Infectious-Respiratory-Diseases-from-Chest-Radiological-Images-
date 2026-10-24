import { useEffect, useState } from 'react'
import { getHealth } from '../services/api.js'

// Shows whether the backend is reachable and whether the trained model is loaded.
export default function ModelStatus({ onChange }) {
  const [state, setState] = useState({ loading: true })

  useEffect(() => {
    let alive = true
    getHealth()
      .then((h) => { if (alive) { setState({ ok: h.model_loaded, detail: h.detail }); onChange?.(h.model_loaded) } })
      .catch(() => { if (alive) { setState({ unreachable: true }); onChange?.(false) } })
    return () => { alive = false }
  }, [onChange])

  if (state.loading) return <div className="pill" role="status">Checking AI server…</div>
  if (state.unreachable) {
    return <div className="alert alert-warn" role="status"><strong>AI server unreachable</strong><p>Unable to connect to the AI server. Start the application with start.bat / start.sh and reload this page.</p></div>
  }
  if (!state.ok) {
    return (
      <div className="alert alert-warn" role="status">
        <strong>Model unavailable</strong>
        <p>{state.detail}</p>
        <small>Predictions are disabled until the trained model is exported to backend/models/ (see README, “Train the model”).</small>
      </div>
    )
  }
  return <div className="pill pill-ok" role="status">● Trained model loaded</div>
}
