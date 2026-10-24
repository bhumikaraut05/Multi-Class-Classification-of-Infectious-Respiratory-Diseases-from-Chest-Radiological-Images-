import { useState } from 'react'
import { downloadReport } from '../services/api.js'

export default function ReportView({ result }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function download() {
    setBusy(true); setError('')
    try {
      const blob = await downloadReport(result.prediction_id)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = `cxr-report-${result.prediction_id}.pdf`
      document.body.appendChild(a); a.click(); a.remove()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card report" aria-label="AI-assisted analysis report">
      <div className="row-between">
        <h3>AI-assisted analysis report</h3>
        <button className="btn btn-primary" onClick={download} disabled={busy}>{busy ? 'Preparing PDF…' : 'Download PDF'}</button>
      </div>
      {error && <div className="alert alert-error" role="alert"><p>{error}</p></div>}
      {result.report.map((s) => (
        <div key={s.title} className="report-sec">
          <h4>{s.title}</h4>
          {s.body && <p>{s.body}</p>}
          {s.items && <ul>{s.items.map((it) => <li key={it}>{it}</li>)}</ul>}
        </div>
      ))}
    </section>
  )
}
