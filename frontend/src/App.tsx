import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import History from './pages/History'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/dashboard/:videoId" element={<Dashboard />} />
      <Route path="/history" element={<History />} />
    </Routes>
  )
}
