import { Bar, BarChart, Cell, LabelList, ResponsiveContainer, XAxis, YAxis, Tooltip } from 'recharts'

export default function ProbabilityChart({ probabilities, top }) {
  const data = Object.entries(probabilities).map(([name, p]) => ({ name, value: +(p * 100).toFixed(2) }))
  return (
    <section className="card" aria-label="Probability distribution">
      <h3>Class probabilities</h3>
      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 48 }}>
            <XAxis type="number" domain={[0, 100]} hide />
            <YAxis type="category" dataKey="name" width={120} tickLine={false} axisLine={false} />
            <Tooltip formatter={(v) => `${v}%`} cursor={{ fill: 'rgba(29,78,216,.06)' }} />
            <Bar dataKey="value" radius={6} barSize={20} isAnimationActive={false}>
              {data.map((d) => <Cell key={d.name} fill={d.name === top ? '#1d4ed8' : '#bfdbfe'} />)}
              <LabelList dataKey="value" position="right" formatter={(v) => `${v}%`} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table className="sr-only">
        <caption>Class probabilities</caption>
        <tbody>{data.map((d) => <tr key={d.name}><th>{d.name}</th><td>{d.value}%</td></tr>)}</tbody>
      </table>
    </section>
  )
}
