import CursorStars from "./components/CursorStar.jsx";
import Landing from "./pages/Landing.jsx";

function App() {

  return (
    <div className="min-h-screen bg-#FAF5F1 text-slate-900">
      <CursorStars color="#79A9E6" density={1} />
      <Landing />
    </div>
  )
}

export default App
