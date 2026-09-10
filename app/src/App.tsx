import { useState } from "react";
import { isConfigured } from "./api/config";
import { Setup } from "./components/Setup";
import { Dashboard } from "./routes/Dashboard/Dashboard";
import { Reveal } from "./routes/Reveal/Reveal";
import { Workbench } from "./routes/Workbench/Workbench";
import type { Route } from "./routes";

function App() {
  const [route, setRoute] = useState<Route>("workbench");
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  // Desktop is configured by definition (Tauri supplies the token), so this only
  // ever gates the browser build.
  const [configured, setConfigured] = useState(isConfigured);

  function openProject(name: string) {
    setSelectedProject(name);
    setRoute("reveal");
  }

  if (!configured) {
    return <Setup onDone={() => setConfigured(true)} />;
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
