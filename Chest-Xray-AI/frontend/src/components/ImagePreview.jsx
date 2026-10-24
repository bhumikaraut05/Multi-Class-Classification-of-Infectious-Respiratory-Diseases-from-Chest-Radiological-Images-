export default function ImagePreview({ src, name, onRemove, disabled, onReplace }) {
  return (
    <figure className="preview">
      <img src={src} alt="Uploaded chest X-ray preview" />
      <figcaption>
        <span className="muted">{name}</span>
        <span className="row-gap">
          {onReplace && <button className="btn btn-ghost" onClick={onReplace} disabled={disabled}>Replace</button>}
          <button className="btn btn-ghost" onClick={onRemove} disabled={disabled}>Remove</button>
        </span>
      </figcaption>
    </figure>
  )
}
