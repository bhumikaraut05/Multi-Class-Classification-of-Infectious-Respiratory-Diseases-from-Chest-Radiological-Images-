import { useEffect, useState } from 'react'
import { getModelInfo } from '../services/api.js'

// { info, error, loading } from the real /api/model-info endpoint.
export default function useModelInfo() {
  const [state, setState] = useState({ loading: true, info: null, error: false })
  useEffect(() => {
    let alive = true
    getModelInfo()
      .then((info) => alive && setState({ loading: false, info, error: false }))
      .catch(() => alive && setState({ loading: false, info: null, error: true }))
    return () => { alive = false }
  }, [])
  return state
}
