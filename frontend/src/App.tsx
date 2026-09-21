import type { ReactNode } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { useAppStore } from './store'
import { ParentView } from './views/ParentView'
import { StudentView } from './views/StudentView'

// Gates /parent on the store's role, mirroring the backend's role check
// (api/role.py) -- a lightweight local switch, not real auth. A student who
// navigates here directly (bookmark, typed URL) is bounced back to /.
function RequireParentRole({ children }: { children: ReactNode }) {
  const role = useAppStore((state) => state.role)
  if (role !== 'parent') return <Navigate to="/" replace />
  return <>{children}</>
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<StudentView />} />
        <Route
          path="/parent"
          element={
            <RequireParentRole>
              <ParentView />
            </RequireParentRole>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
