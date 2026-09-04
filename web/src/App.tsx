import { useEffect } from "react";
import { useGameStore } from "./store";
import Home from "./components/Home";
import Game from "./components/Game";

export default function App() {
  const roomId = useGameStore((s) => s.roomId);
  const loadCatalog = useGameStore((s) => s.loadCatalog);

  useEffect(() => {
    void loadCatalog();
  }, [loadCatalog]);

  return (
    <div className="min-h-screen">
      {roomId ? <Game /> : <Home />}
    </div>
  );
}
