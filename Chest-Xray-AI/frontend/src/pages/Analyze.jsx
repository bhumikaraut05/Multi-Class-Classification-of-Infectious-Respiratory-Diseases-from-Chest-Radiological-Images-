import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import UploadBox from '../components/UploadBox.jsx'
import ImagePreview from '../components/ImagePreview.jsx'
import LoadingState from '../components/LoadingState.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import ModelStatus from '../components/ModelStatus.jsx'
import Disclaimer from '../components/Disclaimer.jsx'
import { predict } from '../services/api.js'

export default function Analyze() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [url, setUrl] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [modelReady, setModelReady] = useState(false)
  const keepUrl = useRef(false)
  const onStatus = useCallback((ok) => setModelReady(ok), [])

  useEffect(() => () => { if (url && !keepUrl.current) URL.revokeObjectURL(url) }, [url])

  function choose(f) {
    if (url) URL.revokeObjectURL(url)
    setError(''); setFile(f); setUrl(URL.createObjectURL(f))
  }
  function reset() { setFile(null); setUrl(null); setError('') }

  async function analyze() {
    if (!file) return setError('Please upload an X-ray image.')
    setBusy(true); setError('')
    try {
      const result = await predict(file)
      keepUrl.current = true   // the result page still needs the preview URL
      navigate('/result', { state: { result, imageUrl: url, name: file.name } })
    } catch (e) {
      setError(e.message)
      setBusy(false)
    }
  }

  return (
    <section className="narrow">
      <h1>Analyze X-ray</h1>
      <Disclaimer />
      <ModelStatus onChange={onStatus} />
      <ErrorMessage message={error} onRetry={() => setError('')} />
      {busy ? <LoadingState /> : !file ? (
        <UploadBox onFile={choose} onReject={setError} />
      ) : (
        <>
          <ImagePreview src={url} name={file.name} onRemove={reset} onReplace={reset} />
          <button className="btn btn-primary btn-lg" onClick={analyze} disabled={!modelReady}>Analyze Image</button>
          {!modelReady && <p className="muted">Analysis is disabled until the trained model is available.</p>}
        </>
      )}
    </section>
  )
}
