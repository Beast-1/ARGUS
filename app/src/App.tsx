import { useState } from "react";
import { Dashboard } from "./routes/Dashboard/Dashboard";
import { Reveal } from "./routes/Reveal/Reveal";
import { Workbench } from "./routes/Workbench/Workbench";
import type { Route } from "./routes";

function App() {
  const [route, setRoute] = useState<Route>("workbench");
  const [selectedProject, setSelectedProject] = useState<string | null>(null);

  function openProject(name: string) {
    setSelectedProject(name);
    setRoute("reveal");
  }

  if (route === "reveal" && selectedProject) {
    return <Reveal name={selectedProject} onNavigate={setRoute} />;
  }

  if (route === "dashboard") {
    return <Dashboard onNavigate={setRoute} onOpenProject={openProject} />;
  }

  return <Workbench onNavigate={setRoute} onOpenProject={openProject} />;
}

export default App;
