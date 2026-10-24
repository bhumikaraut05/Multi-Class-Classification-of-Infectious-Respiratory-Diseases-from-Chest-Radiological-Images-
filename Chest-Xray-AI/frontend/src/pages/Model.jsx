import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import ConfusionMatrix from '../components/ConfusionMatrix.jsx'
import ModelStatus from '../components/ModelStatus.jsx'
import useModelInfo from '../hooks/useModelInfo.js'
import { plotUrl } from '../services/api.js'

const ARCH = {
  baseline_cnn: 'Three convolution + max-pool blocks, one extra convolution, global average pooling, dense head (softmax over 5 classes).',
  custom_cnn_sa: 'Four convolutional blocks (conv-BN-ReLU ×2) with CBAM-style spatial-attention masks after blocks 3 and 4, global average pooling, dense head.',
  vgg19: 'VGG19 convolutional backbone (ImageNet initialisation, fine-tuned) with global average pooling and a dense head.',
  resnet50: 'ResNet50 backbone (ImageNet initialisation, fine-tuned) with global average pooling and a dense head.',
  vit: 'Vision Transformer: 20×20 patches, 5 pre-LN transformer blocks, 5 heads, 768-d embeddings, dense head.',
}
const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

function Curves({ history }) {
  if (!history) return null
  const data = history.loss.map((_, i) => ({
    epoch: i + 1, train_acc: history.accuracy?.[i], val_acc: history.val_accuracy?.[i],
    train_loss: history.loss[i], val_loss: history.val_loss?.[i],
  }))
  const chart = (a, b, title) => (
    <div style={{ height: 240 }}>
      <h4>{title}</h4>
      <ResponsiveContainer>
        <LineChart data={data}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="epoch" /><YAxis /><Tooltip /><Legend />
          <Line dataKey={a} name="train" stroke="#1d4ed8" dot={false} isAnimationActive={false} />
          <Line dataKey={b} name="validation" stroke="#f59e0b" dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
  return <div className="grid-2">{chart('train_acc', 'val_acc', 'Accuracy')}{chart('train_loss', 'val_loss', 'Loss')}</div>
}

export default function Model() {
  const { info, loading, error } = useModelInfo()
  if (loading) return <p role="status">Loading model information…</p>
  if (error || !info) return <ModelStatus />
  const m = info.model, mt = info.metrics, tr = info.training
  const weakest = mt ? Object.entries(mt.per_class).sort((a, b) => a[1].recall - b[1].recall).slice(0, 2) : []
  return (
    <section>
      <h1>Model information</h1>
      <ModelStatus />

      <section className="card" aria-label="Selected algorithm">
        <h3>Selected algorithm</h3>
        {m ? (
          <>
            <dl className="kv">
              <div><dt>Name</dt><dd>{m.name}</dd></div><div><dt>Version</dt><dd>{m.version}</dd></div>
              <div><dt>Parameters</dt><dd>{m.parameters?.toLocaleString() ?? '—'}</dd></div>
              <div><dt>Input</dt><dd>{m.input_size}×{m.input_size}×3</dd></div>
              {mt && <div><dt>Model size</dt><dd>{mt.weights_size_mb} MB</dd></div>}
              {mt && <div><dt>CPU inference</dt><dd>{mt.inference_ms_per_image_cpu_single} ms / image</dd></div>}
            </dl>
            <p>{ARCH[m.architecture]}</p>
            <p className="muted small">Selection criterion: {m.selection_criterion}. Preprocessing: {Object.entries(m.preprocessing).map(([k, v]) => `${k}=${v}`).join(', ')}.</p>
            {m.smoke_test_only && <div className="alert alert-warn">This model was produced by a smoke test, not a full training run.</div>}
          </>
        ) : <p className="muted">No trained model has been exported yet.</p>}
        <h4>Artifact status</h4>
        <ul>{Object.entries(info.artifacts).map(([k, ok]) => <li key={k}>{k}: <strong>{ok ? 'present' : 'missing'}</strong></li>)}</ul>
      </section>

      {info.model_selection && (
        <section className="card" aria-label="Candidate comparison">
          <h3>Candidate comparison (validation split)</h3>
          <div className="table-wrap"><table className="tbl"><thead><tr><th>Model</th><th>Val accuracy</th><th>Val macro-F1</th><th>Val macro-AUC</th></tr></thead>
            <tbody>{info.model_selection.candidates.map((c) => (
              <tr key={c.model}><td>{c.display_name}{c.model === info.model_selection.selected_model ? ' ✔ selected' : ''}</td>
                {c.status === 'ok' ? <><td>{pct(c.val_accuracy)}</td><td>{pct(c.val_macro_f1)}</td><td>{pct(c.val_macro_auc)}</td></> : <td colSpan="3">{c.status}</td>}</tr>))}
            </tbody></table></div>
        </section>
      )}

      <section className="card" aria-label="Test metrics">
        <h3>Test-set evaluation</h3>
        {mt ? (
          <>
            <dl className="kv">
              <div><dt>Accuracy</dt><dd>{pct(mt.accuracy)}</dd></div><div><dt>Macro F1</dt><dd>{pct(mt.macro_f1)}</dd></div>
              <div><dt>Weighted F1</dt><dd>{pct(mt.weighted_f1)}</dd></div><div><dt>Macro AUC</dt><dd>{pct(mt.macro_auc_ovr)}</dd></div>
              <div><dt>Macro AUPRC</dt><dd>{pct(mt.macro_auprc)}</dd></div><div><dt>Test images</dt><dd>{mt.n_samples.toLocaleString()}</dd></div>
            </dl>
            <div className="table-wrap"><table className="tbl"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th>Support</th></tr></thead>
              <tbody>{Object.entries(mt.per_class).map(([c, v]) => (
                <tr key={c}><td>{c}</td><td>{pct(v.precision)}</td><td>{pct(v.recall)}</td><td>{pct(v.f1)}</td><td>{pct(v.roc_auc)}</td><td>{v.support}</td></tr>))}
              </tbody></table></div>
            <h4>Confusion matrix</h4>
            <ConfusionMatrix matrix={mt.confusion_matrix} classes={mt.class_names} />
            <div className="grid-2">
              <figure><img className="plot" src={plotUrl('roc_curves')} alt="ROC curves" /></figure>
              <figure><img className="plot" src={plotUrl('pr_curves')} alt="Precision-recall curves" /></figure>
            </div>
          </>
        ) : <p className="muted">No evaluation results yet. They are written to reports/final/ by <code>python -m ml.finalize</code> and shown here automatically.</p>}
      </section>

      {info.history && <section className="card"><h3>Training curves</h3><Curves history={info.history} /></section>}

      {tr && (
        <section className="card" aria-label="Training configuration">
          <h3>Dataset and training configuration</h3>
          <dl className="kv">
            {Object.entries(tr.split_sizes || {}).map(([s, n]) => <div key={s}><dt>{s} images</dt><dd>{n.toLocaleString()}</dd></div>)}
            <div><dt>Epochs run (best)</dt><dd>{tr.epochs_run} ({tr.best_epoch})</dd></div>
            <div><dt>Learning rate / batch</dt><dd>{tr.hyperparameters?.lr} / {tr.hyperparameters?.batch}</dd></div>
            <div><dt>Optimizer</dt><dd>{tr.optimizer}</dd></div><div><dt>Seed</dt><dd>{tr.seed}</dd></div>
            <div><dt>Class weights</dt><dd>{tr.class_weights?.used ? `used (imbalance ${tr.class_weights.imbalance_ratio.toFixed(2)}×)` : 'not needed'}</dd></div>
            <div><dt>Environment</dt><dd>TF {tr.environment?.tensorflow}, Python {tr.environment?.python}</dd></div>
          </dl>
        </section>
      )}

      <section className="card" aria-label="Limitations">
        <h3>Limitations and known failure cases</h3>
        <ul>
          {weakest.map(([c, v]) => <li key={c}>Lowest recall on the test split: <strong>{c}</strong> ({pct(v.recall)}) — expect more missed cases for this class.</li>)}
          {!mt && <li>Per-class failure modes cannot be listed until the model has been evaluated.</li>}
          <li>Only five classes are modelled; anything else is forced into one of them.</li>
          <li>Patient identifiers were not available, so patient-level leakage between splits could not be excluded.</li>
          <li>Generalisation to other hospitals, scanners and populations has not been tested.</li>
        </ul>
      </section>
    </section>
  )
}
