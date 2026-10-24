import { Link, Navigate, useLocation } from 'react-router-dom'
import PredictionCard from '../components/PredictionCard.jsx'
import ProbabilityChart from '../components/ProbabilityChart.jsx'
import ExplanationViewer from '../components/ExplanationViewer.jsx'
import PreprocessingView from '../components/PreprocessingView.jsx'
import QualityWarnings from '../components/QualityWarnings.jsx'
import ReportView from '../components/ReportView.jsx'
import Disclaimer from '../components/Disclaimer.jsx'

export default function Result() {
  const { state } = useLocation()
  if (!state?.result) return <Navigate to="/analyze" replace />
  const { result, imageUrl } = state
  return (
    <section>
      <div className="row-between">
        <h1>Results</h1>
        <Link to="/analyze" className="btn btn-primary">Analyze another X-ray</Link>
      </div>
      <Disclaimer />
      <QualityWarnings warnings={result.quality_warnings} />
      <div className="grid-2">
        <PredictionCard prediction={result.prediction} confidence={result.confidence} lowConfidence={result.low_confidence} model={result.model} />
        <ProbabilityChart probabilities={result.probabilities} top={result.prediction} />
      </div>
      <ExplanationViewer original={imageUrl} result={result} />
      <PreprocessingView steps={result.preprocessing_steps} />
      <ReportView result={result} />
    </section>
  )
}
