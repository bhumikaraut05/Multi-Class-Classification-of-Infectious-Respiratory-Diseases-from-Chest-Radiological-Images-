import { Link } from 'react-router-dom'
import ModelStatus from '../components/ModelStatus.jsx'
import ModelInfo from '../components/ModelInfo.jsx'
import useModelInfo from '../hooks/useModelInfo.js'

const CLASSES = ['COVID-19', 'Lung Opacity', 'Normal', 'Viral Pneumonia', 'Tuberculosis']
const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

export default function Home() {
  const { info } = useModelInfo()
  const m = info?.metrics
  return (
    <>
      <section className="hero">
        <span className="pill">Academic research prototype</span>
        <h1>AI-Assisted Chest X-ray Analysis</h1>
        <p className="lead">Upload a chest X-ray and get an AI classification among five categories, with model-estimated probabilities, an explanation map and a downloadable report.</p>
        <ModelStatus />
        <Link to="/analyze" className="btn btn-primary btn-lg">Start Analysis</Link>
        <ul className="chips" aria-label="Supported classes">{CLASSES.map((c) => <li key={c}>{c}</li>)}</ul>
      </section>

      <div className="grid-2">
        <ModelInfo info={info} />
        <section className="card" aria-label="Evaluation summary">
          <h3>Held-out test results</h3>
          {m ? (
            <>
              <dl className="kv">
                <div><dt>Accuracy</dt><dd>{pct(m.accuracy)}</dd></div>
                <div><dt>Macro F1</dt><dd>{pct(m.macro_f1)}</dd></div>
                <div><dt>Macro AUC</dt><dd>{pct(m.macro_auc_ovr)}</dd></div>
                <div><dt>Test images</dt><dd>{m.n_samples.toLocaleString()}</dd></div>
              </dl>
              <p className="muted small">Computed once on the untouched test split. <Link to="/model">Full evaluation →</Link></p>
            </>
          ) : (
            <p className="muted">No evaluation results are available yet. They appear here after the model is trained and evaluated (README → “Train the model”). Nothing is shown until real results exist.</p>
          )}
        </section>
      </div>

      <section className="grid-3">
        <div className="card"><h3>1 · Upload</h3><p>Drop a frontal chest X-ray (JPG or PNG). Images are processed in memory and never stored.</p></div>
        <div className="card"><h3>2 · Classify</h3><p>The same preprocessing used in training feeds the single selected deep-learning model.</p></div>
        <div className="card"><h3>3 · Review</h3><p>See probabilities, an explanation map, and a report to discuss with a qualified clinician.</p></div>
      </section>

      <section className="card" aria-label="Purpose and limitations">
        <h3>Purpose and limitations</h3>
        <ul>
          <li>For education and research on chest X-ray classification — not for clinical use.</li>
          <li>The output is a classification among five labels, not a diagnosis, and does not locate lesions, grade severity, or suggest treatment.</li>
          <li>Accuracy on images from other hospitals or scanners than the training data is unknown.</li>
        </ul>
      </section>
    </>
  )
}
