import { useRef, useState } from 'react'

export const MAX_MB = 10
const MIN_SIDE = 64
const MAX_SIDE = 10000
const OK_TYPES = ['image/jpeg', 'image/png']
const OK_EXT = /\.(jpe?g|png)$/i

function readDimensions(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => { URL.revokeObjectURL(url); resolve([img.naturalWidth, img.naturalHeight]) }
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('decode')) }
    img.src = url
  })
}

export default function UploadBox({ onFile, onReject }) {
  const input = useRef(null)
  const [over, setOver] = useState(false)

  async function accept(file) {
    if (!file) return onReject?.('Please upload an X-ray image.')
    if (!OK_TYPES.includes(file.type) || !OK_EXT.test(file.name)) return onReject?.('Please upload a JPG, JPEG, or PNG image.')
    if (file.size > MAX_MB * 1024 * 1024) return onReject?.(`This image is larger than ${MAX_MB} MB. Please upload a smaller file.`)
    if (file.size === 0) return onReject?.('This file is empty.')
    try {
      const [w, h] = await readDimensions(file)
      if (Math.min(w, h) < MIN_SIDE) return onReject?.(`The image is too small (${w}×${h}). Minimum is ${MIN_SIDE}×${MIN_SIDE} pixels.`)
      if (Math.max(w, h) > MAX_SIDE) return onReject?.(`The image is too large (${w}×${h}). Maximum is ${MAX_SIDE} pixels per side.`)
    } catch {
      return onReject?.('The selected file could not be read as an image.')
    }
    onFile(file)
  }

  return (
    <div
      className={`drop ${over ? 'drop-over' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); accept(e.dataTransfer.files?.[0]) }}
    >
      <div className="drop-icon" aria-hidden="true">⇪</div>
      <h3>Drag &amp; Drop X-ray</h3>
      <p className="muted">or</p>
      <button className="btn btn-primary" onClick={() => input.current?.click()}>Choose Image</button>
      <input
        ref={input} type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" hidden
        aria-label="Choose X-ray image"
        onChange={(e) => { accept(e.target.files?.[0]); e.target.value = '' }}
      />
      <small className="muted">JPG / JPEG / PNG · up to {MAX_MB} MB · at least {MIN_SIDE}×{MIN_SIDE} px</small>
    </div>
  )
}
