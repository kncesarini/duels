// Theme, UI scale, animation speed, colour-vision marks, confirmation and
// log detail, plus the way out of the game.

import type { Settings } from "../lib/settings";

interface Props {
  settings: Settings;
  onChange: (patch: Partial<Settings>) => void;
  onLeave: () => void;
  onClose: () => void;
}

export default function SettingsMenu({ settings, onChange, onLeave, onClose }: Props) {
  return (
    <div className="menu" role="dialog" aria-label="Settings" data-testid="settings">
      <div className="row">
        <label htmlFor="set-theme">Theme</label>
        <select
          id="set-theme"
          value={settings.theme}
          onChange={(e) => onChange({ theme: e.target.value as Settings["theme"] })}
          data-testid="set-theme"
        >
          <option value="system">System</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </div>
      <div className="row">
        <label htmlFor="set-scale">UI scale</label>
        <select id="set-scale" value={settings.scale} onChange={(e) => onChange({ scale: Number(e.target.value) })}>
          <option value={0.9}>90%</option>
          <option value={1}>100%</option>
          <option value={1.15}>115%</option>
          <option value={1.3}>130%</option>
        </select>
      </div>
      <div className="row">
        <label htmlFor="set-anim">Animation speed</label>
        <select
          id="set-anim"
          value={settings.animation}
          onChange={(e) => onChange({ animation: e.target.value as Settings["animation"] })}
        >
          <option value="slow">Slow</option>
          <option value="normal">Normal</option>
          <option value="fast">Fast</option>
          <option value="off">Off (reduced motion)</option>
        </select>
      </div>
      <div className="row">
        <label htmlFor="set-marks">High-contrast type marks</label>
        <input id="set-marks" type="checkbox" checked={settings.marks} onChange={(e) => onChange({ marks: e.target.checked })} />
      </div>
      <div className="row">
        <label htmlFor="set-confirm">Confirm before Build / Discard</label>
        <input
          id="set-confirm"
          type="checkbox"
          checked={settings.confirm}
          onChange={(e) => onChange({ confirm: e.target.checked })}
        />
      </div>
      <div className="row">
        <label htmlFor="set-compact">Compact log entries</label>
        <input
          id="set-compact"
          type="checkbox"
          checked={settings.compactLog}
          onChange={(e) => onChange({ compactLog: e.target.checked })}
        />
      </div>
      <div className="row">
        <label htmlFor="set-advanced" title="Also switchable with ?advanced=1 in the URL">
          Advanced (AI analysis) mode
        </label>
        <input
          id="set-advanced"
          type="checkbox"
          checked={settings.advanced}
          onChange={(e) => onChange({ advanced: e.target.checked })}
          data-testid="set-advanced"
        />
      </div>
      <hr />
      <div style={{ fontSize: 10.5, color: "var(--mute)" }}>
        Keyboard: ← → select a card · B build · D discard · 1-4 spend it on that wonder · Alt peek the other
        player&apos;s costs · L log · [ ] step through review · Esc back to live
      </div>
      <hr />
      <div className="row">
        <button type="button" className="btn" onClick={onClose}>
          Close
        </button>
        <button type="button" className="btn ghost" onClick={onLeave} data-testid="nav-leave-game">
          Leave game
        </button>
      </div>
    </div>
  );
}
