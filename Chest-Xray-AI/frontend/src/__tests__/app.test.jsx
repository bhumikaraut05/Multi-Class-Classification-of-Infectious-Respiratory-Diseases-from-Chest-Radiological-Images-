import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { vi } from 'vitest'
import Analyze from '../pages/Analyze.jsx'
import Result from '../pages/Result.jsx'
import Home from '../pages/Home.jsx'

vi.mock('../services/api.js', () => ({
  getHealth: vi.fn(),
  getModelInfo: vi.fn(),
  predict: vi.fn(),
  downloadReport: vi.fn(),
  plotUrl: (n) => `/api/evaluation-plot/${n}`,
}))
import { downloadReport, getHealth, getModelInfo, predict } from '../services/api.js'

global.URL.createObjectURL = vi.fn(() => 'blob:x')
global.URL.revokeObjectURL = vi.fn()
global.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} }
// jsdom does not decode images: pretend every image is 300x300.
global.Image = class { set src(_) { this.naturalWidth = 300; this.naturalHeight = 300; setTimeout(() => this.onload?.(), 0) } }

const png = () => new File([new Uint8Array([1, 2])], 'x.png', { type: 'image/png' })
const renderApp = (path = '/analyze') => render(
  <MemoryRouter initialEntries={[path]}>
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/analyze" element={<Analyze />} />
      <Route path="/result" element={<Result />} />
    </Routes>
  </MemoryRouter>,
)

const RESULT = {
  prediction_id: 'abc123', created_at: '2026-01-01T00:00:00', filename: 'x.png',
  model: { name: 'Test model', version: '9.9', architecture: 'custom_cnn_sa' },
  prediction: 'Tuberculosis', confidence: 0.9, low_confidence: false,
  probabilities: { 'COVID-19': 0.02, 'Lung-Opacity': 0.03, Normal: 0.03, 'Viral Pneumonia': 0.02, Tuberculosis: 0.9 },
  processed_image: 'data:,', preprocessing_steps: [{ name: 'CLAHE', image: 'data:,' }],
  explanation: { available: true, method: 'grad_cam', title: 'Grad-CAM', heatmap: 'data:,', overlay: 'data:,', note: 'Explanation note.' },
  quality_warnings: [{ code: 'low_resolution', message: 'Low resolution (100x100).' }],
  report: [{ title: 'Analysis summary', items: ['Report ID: abc123'] }, { title: 'Disclaimer', body: 'Not a diagnosis.' }],
}

beforeEach(() => {
  vi.clearAllMocks()
  getHealth.mockResolvedValue({ status: 'healthy', model_loaded: true, detail: 'Model loaded' })
  getModelInfo.mockResolvedValue({ model_loaded: true, classes: [], model: null, metrics: null, artifacts: {} })
})

test('upload shows preview, remove resets', async () => {
  renderApp()
  await userEvent.upload(screen.getByLabelText('Choose X-ray image'), png())
  expect(await screen.findByAltText(/preview/i)).toBeInTheDocument()
  await userEvent.click(screen.getByText('Remove'))
  expect(screen.getByText(/Drag & Drop/)).toBeInTheDocument()
})

test('rejects wrong type and oversized files before calling the backend', async () => {
  renderApp()
  const input = screen.getByLabelText('Choose X-ray image')
  await userEvent.upload(input, new File(['x'], 'x.gif', { type: 'image/gif' }), { applyAccept: false })
  expect(await screen.findByRole('alert')).toHaveTextContent('Please upload a JPG, JPEG, or PNG image.')
  const big = new File([new Uint8Array(11 * 1024 * 1024)], 'big.png', { type: 'image/png' })
  await userEvent.upload(input, big)
  expect(await screen.findByRole('alert')).toHaveTextContent('larger than 10 MB')
  expect(predict).not.toHaveBeenCalled()
})

test('analysis disabled and explained when the model is unavailable', async () => {
  getHealth.mockResolvedValue({ status: 'healthy', model_loaded: false, detail: 'Missing model files: final_model.h5.' })
  renderApp()
  await userEvent.upload(screen.getByLabelText('Choose X-ray image'), png())
  expect(await screen.findByRole('button', { name: 'Analyze Image' })).toBeDisabled()
  expect(screen.getAllByText(/Missing model files/).length).toBeGreaterThan(0)
})

test('loading, then the actual backend response is displayed, and the report downloads', async () => {
  let resolve
  predict.mockReturnValue(new Promise((r) => { resolve = r }))
  downloadReport.mockResolvedValue(new Blob(['%PDF']))
  renderApp()
  await userEvent.upload(screen.getByLabelText('Choose X-ray image'), png())
  await userEvent.click(await screen.findByRole('button', { name: 'Analyze Image' }))
  expect(await screen.findByText('Analyzing X-ray…')).toBeInTheDocument()
  resolve(RESULT)
  expect(await screen.findByText('Tuberculosis', { selector: 'h2' })).toBeInTheDocument()
  expect(screen.getByText('90.00%')).toBeInTheDocument()
  expect(screen.getByText('Low resolution (100x100).')).toBeInTheDocument()
  expect(screen.getByText(/Report ID: abc123/)).toBeInTheDocument()
  expect(screen.getAllByText(/academic research and educational prototype/i).length).toBeGreaterThan(0)
  await userEvent.click(screen.getByRole('button', { name: 'Download PDF' }))
  await waitFor(() => expect(downloadReport).toHaveBeenCalledWith('abc123'))
})

test('error state shows server message', async () => {
  predict.mockRejectedValue(new Error('The uploaded image could not be processed.'))
  renderApp()
  await userEvent.upload(screen.getByLabelText('Choose X-ray image'), png())
  await userEvent.click(await screen.findByRole('button', { name: 'Analyze Image' }))
  await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('could not be processed'))
})

test('home shows no metrics until real results exist', async () => {
  renderApp('/')
  expect(await screen.findByText(/No evaluation results are available yet/)).toBeInTheDocument()
})
