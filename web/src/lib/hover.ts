// The hover/long-press detail panel's target. Hover adds *depth* (rule text,
// exact arithmetic), never access: everything it shows is already on screen
// somewhere, just smaller.

import { create } from "zustand";

export type HoverKind = "card" | "wonder" | "token";

export interface HoverTarget {
  kind: HoverKind;
  id: string;
  /** Where the anchor element is, so the panel can sit beside it. */
  rect: { top: number; left: number; right: number; bottom: number };
  /** For a structure card: how many face-down cards taking it would uncover. */
  uncovers?: number;
}

interface HoverStore {
  target: HoverTarget | null;
  show: (kind: HoverKind, id: string, el: HTMLElement | null, uncovers?: number) => void;
}

let timer: ReturnType<typeof setTimeout> | null = null;

export const useHover = create<HoverStore>((set) => ({
  target: null,
  show: (kind, id, el, uncovers) => {
    if (timer) clearTimeout(timer);
    if (!el) {
      set({ target: null });
      return;
    }
    const r = el.getBoundingClientRect();
    const rect = { top: r.top, left: r.left, right: r.right, bottom: r.bottom };
    timer = setTimeout(() => set({ target: { kind, id, rect, uncovers } }), 250);
  },
}));
