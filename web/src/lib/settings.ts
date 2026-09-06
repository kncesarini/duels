// User settings: persisted in `localStorage`, applied to <html> as data
// attributes and custom properties so the CSS in `index.css` can react
// without any component knowing about them.

export type ThemeSetting = "system" | "light" | "dark";
export type AnimationSetting = "slow" | "normal" | "fast" | "off";

export interface Settings {
  theme: ThemeSetting;
  /** Multiplies every animation duration; "off" replaces travel animations
   * with instant placement plus a lingering seat-coloured rim. */
  animation: AnimationSetting;
  /** 0.9 / 1 / 1.15 / 1.3. */
  scale: number;
  /** Adds a hatch pattern per card type, on top of colour and glyph. */
  marks: boolean;
  /** Turns Build/Discard into a two-step confirm. */
  confirm: boolean;
  /** Compact log entries drop the payment line. */
  compactLog: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  theme: "system",
  animation: "normal",
  scale: 1,
  marks: false,
  confirm: false,
  compactLog: false,
};

const KEY = "duels.settings";

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // Private-mode browsers refuse to write; the session still works.
  }
}

export const SPEED_FACTOR: Record<AnimationSetting, number> = {
  slow: 1.5,
  normal: 1,
  fast: 0.5,
  // Not zero: the sequence still has to *happen* so the log and ticker stay
  // ordered; only the travel animations are replaced (see `index.css`).
  off: 0.25,
};

/** True when the player (or their OS) has asked for no motion, in which case
 * a taken card is placed rather than flown. */
export function motionOff(s: Settings): boolean {
  if (s.animation === "off") return true;
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function applySettings(s: Settings): void {
  const root = document.documentElement;
  if (s.theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", s.theme);
  root.setAttribute("data-marks", s.marks ? "on" : "off");
  root.setAttribute("data-motion", motionOff(s) ? "off" : "on");
  root.style.setProperty("--speed", String(SPEED_FACTOR[s.animation]));
  root.style.setProperty("--ui-scale", String(s.scale));
}
