// Icon components over the sprite. Data (orders, labels, glyph ids) lives in
// `iconData.ts` so this module only exports components.

import { TYPE_GLYPH } from "./iconData";
import type { CardType } from "../generated/CardType";

interface IcoProps {
  /** Sprite id without the leading `i-`/`g-`. */
  id: string;
  /** Extra classes, e.g. the resource name (which colours it). */
  className?: string;
  title?: string;
  size?: number;
}

/** A resource / science / misc icon from the sprite. */
export function Ico({ id, className = "", title, size }: IcoProps) {
  return (
    <svg
      className={`ic ${className}`}
      aria-hidden={title ? undefined : true}
      role={title ? "img" : undefined}
      style={size ? { width: size, height: size } : undefined}
    >
      {title && <title>{title}</title>}
      <use href={`#i-${id}`} />
    </svg>
  );
}

/** A card-type glyph from the sprite. */
export function TypeGlyph({ type, className = "" }: { type: CardType; className?: string }) {
  return (
    <svg className={`ic ${className}`} aria-hidden>
      <use href={`#${TYPE_GLYPH[type]}`} />
    </svg>
  );
}

/** Mounted once at the app root. */
export function IconSprite() {
  return (
    <svg style={{ display: "none" }} xmlns="http://www.w3.org/2000/svg">
      <symbol id="i-wood" viewBox="0 0 16 16">
        <rect x="1.5" y="3" width="3.6" height="10" rx="1" />
        <rect x="6.2" y="3" width="3.6" height="10" rx="1" />
        <rect x="10.9" y="3" width="3.6" height="10" rx="1" />
      </symbol>
      <symbol id="i-stone" viewBox="0 0 16 16">
        <path d="M3 6.2 8 2.8l5 3.4v5.2L8 14.4 3 11.4z" />
      </symbol>
      <symbol id="i-clay" viewBox="0 0 16 16">
        <path d="M1.8 4.5h12.4L12.6 13H3.4z" />
      </symbol>
      <symbol id="i-glass" viewBox="0 0 16 16">
        <circle cx="8" cy="8" r="5.6" />
        <circle cx="6" cy="6" r="1.6" fill="#fff" opacity=".75" />
      </symbol>
      <symbol id="i-papyrus" viewBox="0 0 16 16">
        <rect x="3.5" y="1.5" width="9" height="13" rx="1.6" />
        <path d="M6 5h4M6 8h4M6 11h4" stroke="#1B1D24" strokeWidth="1.1" opacity=".55" />
      </symbol>
      <symbol id="i-shield" viewBox="0 0 16 16">
        <path d="M8 1.2 13.6 3.4V8c0 3.5-2.4 5.6-5.6 7C4.8 13.6 2.4 11.5 2.4 8V3.4z" />
      </symbol>
      <symbol id="i-coin" viewBox="0 0 16 16">
        <circle cx="8" cy="8" r="7" />
        <circle cx="8" cy="8" r="4.6" fill="none" stroke="#000" strokeOpacity=".3" strokeWidth="1.2" />
      </symbol>
      <symbol id="i-link" viewBox="0 0 16 16">
        <path
          d="M6.5 9.5 9.5 6.5M5 11a2.5 2.5 0 0 1 0-3.5l1.5-1.5M11 5a2.5 2.5 0 0 1 0 3.5L9.5 10"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </symbol>
      <symbol id="i-repeat" viewBox="0 0 16 16">
        <path
          d="M3 7a5 5 0 0 1 9-3M13 9a5 5 0 0 1-9 3"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
        <path d="M12 1.5 12.5 5 9 4.6zM4 14.5 3.5 11 7 11.4z" />
      </symbol>
      {/* science symbols */}
      <symbol id="i-wheel" viewBox="0 0 16 16">
        <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path d="M8 2v12M2 8h12M3.8 3.8l8.4 8.4M12.2 3.8 3.8 12.2" stroke="currentColor" strokeWidth="1.3" />
      </symbol>
      <symbol id="i-mortar" viewBox="0 0 16 16">
        <path d="M2.5 7h11c0 3.6-2.2 6-5.5 6S2.5 10.6 2.5 7z" />
        <path d="M9 6.5 13 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </symbol>
      <symbol id="i-sundial" viewBox="0 0 16 16">
        <path d="M2 13A8 8 0 0 1 14 6.5V13z" />
        <path d="M8 13V3" stroke="currentColor" strokeWidth="1.6" />
      </symbol>
      <symbol id="i-gyroscope" viewBox="0 0 16 16">
        <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <ellipse cx="8" cy="8" rx="6" ry="2.4" fill="none" stroke="currentColor" strokeWidth="1.3" />
        <path d="M8 2v12" stroke="currentColor" strokeWidth="1.3" />
      </symbol>
      <symbol id="i-pendulum" viewBox="0 0 16 16">
        <path d="M8 1.5 14.5 13H1.5z" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <path d="M8 4v6" stroke="currentColor" strokeWidth="1.3" />
        <circle cx="8" cy="11" r="1.4" />
      </symbol>
      <symbol id="i-inkwell" viewBox="0 0 16 16">
        <rect x="3" y="2" width="10" height="12" rx="1.2" />
        <path d="M5.5 5.5h5M5.5 8h5M5.5 10.5h3" stroke="#1B1D24" strokeWidth="1.1" opacity=".55" />
      </symbol>
      <symbol id="i-balance" viewBox="0 0 16 16">
        <path d="M8 2v12M3 5h10" stroke="currentColor" strokeWidth="1.5" />
        <path d="M1.5 10 3 5l1.5 5zM11.5 10 13 5l1.5 5z" />
      </symbol>
      {/* card-type glyphs */}
      <symbol id="g-raw" viewBox="0 0 16 16">
        <path d="M2 13 8 3l6 10z" />
      </symbol>
      <symbol id="g-man" viewBox="0 0 16 16">
        <path d="M5 2h6v4l3 7H2l3-7z" />
      </symbol>
      <symbol id="g-civ" viewBox="0 0 16 16">
        <path d="M2 4h12v2H2zM3 7h2v6H3zM7 7h2v6H7zM11 7h2v6h-2zM2 13h12v1.5H2z" />
      </symbol>
      <symbol id="g-sci" viewBox="0 0 16 16">
        <path d="M6 2h4v1.5L9 4v3l4 6.5H3L7 7V4L6 3.5z" />
      </symbol>
      <symbol id="g-com" viewBox="0 0 16 16">
        <circle cx="8" cy="8" r="6.5" />
        <path
          d="M8 4.5v7M6 6.3h3.2a1.3 1.3 0 0 1 0 2.6H6.8a1.3 1.3 0 0 0 0 2.6H10"
          fill="none"
          stroke="#1B1D24"
          strokeWidth="1.2"
          strokeOpacity=".6"
        />
      </symbol>
      <symbol id="g-mil" viewBox="0 0 16 16">
        <path d="M8 1.2 13.6 3.4V8c0 3.5-2.4 5.6-5.6 7C4.8 13.6 2.4 11.5 2.4 8V3.4z" />
      </symbol>
      <symbol id="g-gld" viewBox="0 0 16 16">
        <path d="M8 1.5 12 4v8l-4 2.5L4 12V4z" fill="none" stroke="currentColor" strokeWidth="1.6" />
        <circle cx="8" cy="8" r="2" />
      </symbol>
      <symbol id="g-won" viewBox="0 0 16 16">
        <path d="M2 13 8 2l6 11H2zm3-1.5h6L8 5.5z" />
      </symbol>
    </svg>
  );
}
