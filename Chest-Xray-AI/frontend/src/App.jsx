import { Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import Home from './pages/Home.jsx'
import Analyze from './pages/Analyze.jsx'
import Result from './pages/Result.jsx'
import Model from './pages/Model.jsx'
import About from './pages/About.jsx'

export default function App() {
  return (
    <div className="shell">
      <Navbar />
      <div className="content">
        <main className="container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/result" element={<Result />} />
            <Route path="/model" element={<Model />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <footer className="footer">CXR-AI · Academic research prototype · Not a medical device · Not a substitute for professional medical assessment</footer>
      </div>
    </div>
  )
}
