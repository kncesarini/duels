import type { Action } from "../generated/Action";

/** Every slot index that at least one of `legal` refers to. */
export function actionableSlotsFrom(legal: Action[]): Set<number> {
  const out = new Set<number>();
  for (const a of legal) {
    if (a.type === "Build" || a.type === "Discard" || a.type === "BuildWonder") {
      out.add(a.slot);
    }
  }
  return out;
}
