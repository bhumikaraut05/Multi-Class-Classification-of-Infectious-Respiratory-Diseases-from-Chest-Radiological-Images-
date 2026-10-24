export default function ConfusionMatrix({ matrix, classes }) {
  return (
    <div className="cm-wrap">
      <table className="cm" aria-label="Confusion matrix (rows: true class, columns: predicted class)">
        <thead><tr><th scope="col">True ↓ / Predicted →</th>{classes.map((c) => <th key={c} scope="col">{c}</th>)}</tr></thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1
            return (
              <tr key={classes[i]}>
                <th scope="row">{classes[i]}</th>
                {row.map((v, j) => (
                  <td key={j} style={{ background: `rgba(29,78,216,${(v / total).toFixed(3)})`, color: v / total > 0.5 ? '#fff' : 'inherit' }}>
                    <strong>{v}</strong><small>{((v / total) * 100).toFixed(0)}%</small>
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
