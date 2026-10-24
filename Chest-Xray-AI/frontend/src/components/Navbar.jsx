import { Link, NavLink } from 'react-router-dom'

const LINKS = [['/', 'Dashboard', true], ['/analyze', 'Analyze X-ray'], ['/model', 'Model information'], ['/about', 'About & disclaimer']]

export default function Navbar() {
  return (
    <header className="side">
      <Link to="/" className="brand" aria-label="CXR-AI home">
        <span className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M12 3v8M12 11c-1.5 2-4 3-6 3-1.5 0-2.5 1-2.5 2.5S5 20 7 20c3 0 5-3 5-9zM12 11c1.5 2 4 3 6 3 1.5 0 2.5 1 2.5 2.5S19 20 17 20c-3 0-5-3-5-9z" />
          </svg>
        </span>
        CXR-AI
      </Link>
      <nav className="side-links" aria-label="Main">
        {LINKS.map(([to, label, end]) => <NavLink key={to} to={to} end={end}>{label}</NavLink>)}
      </nav>
      <p className="side-note">Research prototype</p>
    </header>
  )
}
