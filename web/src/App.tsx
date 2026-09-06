import { useEffect } from "react";
import { useGameStore } from "./store";
import Home from "./components/Home";
import Game from "./components/Game";
import { IconSprite } from "./lib/icons";

export default function App() {
  const roomId = useGameStore((s) => s.roomId);
  const loadCatalog = useGameStore((s) => s.loadCatalog);
  const loadAgents = useGameStore((s) => s.loadAgents);

  useEffect(() => {
    void loadCatalog();
    void loadAgents();
  }, [loadCatalog, loadAgents]);

  return (
    <>
      <IconSprite />
      {roomId ? <Game /> : <Home />}
    </>
  );
}
