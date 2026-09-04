// Talks to duels-server exclusively over the REST/WebSocket protocol from
// `crates/duels-server/src/protocol.rs`, using the ts-rs-generated types in
// `src/generated/`. Nothing in this module (or anywhere else in this app)
// computes a rule, a legal move, or a score - it only shapes fetch/WS calls.

import type { Catalog } from "../generated/Catalog";
import type { CreateRoomRequest } from "../generated/CreateRoomRequest";
import type { CreateRoomResponse } from "../generated/CreateRoomResponse";
import type { RoomInfo } from "../generated/RoomInfo";
import type { ClientMessage } from "../generated/ClientMessage";
import type { ServerMessage } from "../generated/ServerMessage";

/** Base HTTP origin of `duels-server`. Overridable at build time for the
 * Docker Compose / e2e setup, where the server isn't on localhost:8080. */
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8080";

const WS_BASE = API_BASE.replace(/^http/, "ws");

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body.error) message = body.error;
    } catch {
      // body wasn't JSON; keep the status line
    }
    throw new Error(message);
  }
  return (await res.json()) as T;
}

export function fetchCatalog(): Promise<Catalog> {
  return fetch(`${API_BASE}/catalog`).then((r) => asJson<Catalog>(r));
}

/** Every agent name `POST /rooms` will accept for an agent seat, in the
 * order the opponent picker should offer them (`"random"` first). Backed by
 * `duels-server`'s `room::KNOWN_AGENTS` so the UI never hand-maintains its
 * own copy that could drift from what the server actually knows how to
 * construct. */
export function fetchAgents(): Promise<string[]> {
  return fetch(`${API_BASE}/agents`).then((r) => asJson<string[]>(r));
}

export function createRoom(req: CreateRoomRequest): Promise<CreateRoomResponse> {
  return fetch(`${API_BASE}/rooms`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  }).then((r) => asJson<CreateRoomResponse>(r));
}

export function fetchRoomInfo(roomId: string): Promise<RoomInfo> {
  return fetch(`${API_BASE}/rooms/${roomId}`).then((r) => asJson<RoomInfo>(r));
}

export interface RoomSocketHandlers {
  onMessage: (msg: ServerMessage) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: () => void;
}

/** Open the room's WebSocket and wire up the given handlers. Returns the
 * raw socket; callers get a `send(action)` helper via `sendAction`. */
export function connectRoomSocket(roomId: string, handlers: RoomSocketHandlers): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/rooms/${roomId}/ws`);
  ws.addEventListener("open", () => handlers.onOpen?.());
  ws.addEventListener("close", () => handlers.onClose?.());
  ws.addEventListener("error", () => handlers.onError?.());
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(ev.data as string) as ServerMessage;
    handlers.onMessage(msg);
  });
  return ws;
}

export function sendAction(ws: WebSocket, action: ClientMessage["action"]): void {
  const msg: ClientMessage = { type: "Action", action };
  ws.send(JSON.stringify(msg));
}
