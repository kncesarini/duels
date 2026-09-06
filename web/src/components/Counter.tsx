// A counter that ticks to its new value instead of snapping, and floats the
// delta beside itself. Coins and victory points both change for reasons that
// are easy to miss if the number simply replaces itself.

import { useEffect, useRef, useState } from "react";

export default function Counter({ value, showDelta = true }: { value: number; showDelta?: boolean }) {
  const [shown, setShown] = useState(value);
  const [delta, setDelta] = useState<number | null>(null);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    const from = shown;
    if (from === value) return;
    if (showDelta) setDelta(value - from);
    const start = performance.now();
    const duration = 450;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      setShown(Math.round(from + (value - from) * t));
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    const clear = setTimeout(() => setDelta(null), 1300);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
      clearTimeout(clear);
    };
    // `shown` is deliberately not a dependency: re-running mid-tween would
    // restart the animation on every frame.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, showDelta]);

  return (
    <>
      {shown}
      {delta !== null && delta !== 0 && (
        <span className={`delta float ${delta > 0 ? "up" : "dn"}`} aria-hidden>
          {delta > 0 ? "+" : ""}
          {delta}
        </span>
      )}
    </>
  );
}
