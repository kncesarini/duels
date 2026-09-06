// Remembers where every card was drawn last time, so a card that has just
// left the structure can still be animated *from* the slot it left. The map
// is written after every paint and never cleared, so the entry for a card is
// its position in the frame before the one that removed it - which is exactly
// the start point the take animation needs.

export interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

const rects = new Map<string, Rect>();

function box(el: Element): Rect {
  const r = el.getBoundingClientRect();
  return { top: r.top, left: r.left, width: r.width, height: r.height };
}

/** Call after every paint. */
export function recordRects(): void {
  document.querySelectorAll<HTMLElement>("[data-card]").forEach((el) => {
    const id = el.dataset.card;
    if (id) rects.set(id, box(el));
  });
}

export function lastRect(cardId: string): Rect | undefined {
  return rects.get(cardId);
}

/** Where a card has landed now: its name chip in a city strip, its wonder
 * tile, or its chip in the discard list. */
export function destinationRect(cardId: string): Rect | undefined {
  const el = document.querySelector<HTMLElement>(`[data-dest="${CSS.escape(cardId)}"]`);
  return el ? box(el) : undefined;
}
