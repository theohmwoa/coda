/**
 * coda WebSocket client.
 *
 * Public surface (consumed by `App.tsx`):
 *   - `connect({ onFrame, onSkills })` → `{ send, close }`
 *   - `ServerFrame` discriminated union of incoming server frames
 *   - `ClientFrame` discriminated union of outgoing client frames
 *
 * Server protocol detail abstracted here: each "run" is one fresh
 * WebSocket connection on the backend. `connect()` exposes a single
 * persistent façade — internally it opens a new WS the first time you
 * `send({kind: 'prompt'})`, and reuses the same socket for
 * approval_response frames until the run ends, then resets.
 */

import type { Skill } from "./lib/types";

// ---- Server → client frames -----------------------------------------------

export type ServerFrame =
  | {
      kind: "ready";
      model: string;
      mcp_servers: string[];
      sub_agents: string[];
      skills: string[];
      workspace: string;
    }
  | {
      kind: "event";
      type: string;
      ts: string;
      payload: Record<string, unknown>;
    }
  | {
      kind: "approval_needed";
      action: { type: string; id: string; payload: unknown };
    }
  | { kind: "final"; text: string }
  | { kind: "error"; error: string; traceback?: string }
  | {
      kind: "done";
      elapsed_s: number;
      turns: number;
      tokens_in: number;
      tokens_out: number;
    };

// ---- Client → server frames -----------------------------------------------

export type ClientFrame =
  | { kind: "prompt"; text: string }
  | {
      kind: "approval_response";
      action_id: string;
      edit: {
        actionId: string;
        outcome: { kind: "accept" | "modify" | "reject"; payload?: unknown; reason?: string; notes?: string };
      };
    };

// ---- public API -----------------------------------------------------------

export interface ConnectOptions {
  onFrame: (frame: ServerFrame) => void;
  onSkills?: (skills: Skill[]) => void;
}

export interface Connection {
  send: (frame: ClientFrame | unknown) => void;
  close: () => void;
}

export function connect(opts: ConnectOptions): Connection {
  let ws: WebSocket | null = null;
  let closed = false;

  // Fire-and-forget initial skills fetch
  if (opts.onSkills) {
    fetch("/skills")
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => opts.onSkills?.(Array.isArray(data) ? data : []))
      .catch(() => opts.onSkills?.([]));
  }

  function openForRun(prompt: string) {
    if (closed) return;
    const url =
      (location.protocol === "https:" ? "wss://" : "ws://") +
      location.host +
      "/ws/run";
    const sock = new WebSocket(url);
    ws = sock;
    sock.onopen = () => {
      sock.send(JSON.stringify({ prompt }));
    };
    sock.onmessage = (m) => {
      let frame: ServerFrame;
      try {
        frame = JSON.parse(m.data) as ServerFrame;
      } catch {
        return;
      }
      opts.onFrame(frame);
    };
    sock.onerror = () => {
      opts.onFrame({ kind: "error", error: "WebSocket error" });
    };
    sock.onclose = () => {
      if (ws === sock) ws = null;
    };
  }

  function send(frame: ClientFrame | unknown) {
    if (closed) return;
    const f = frame as ClientFrame;
    if (f.kind === "prompt") {
      // Each prompt starts a new server run on a fresh socket. If a prior
      // run's socket is still open (shouldn't be, but just in case), close it.
      if (ws && ws.readyState <= 1) {
        try { ws.close(); } catch { /* noop */ }
      }
      openForRun(f.text);
      return;
    }
    if (f.kind === "approval_response") {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(f));
      }
      return;
    }
  }

  function close() {
    closed = true;
    if (ws) {
      try { ws.close(); } catch { /* noop */ }
      ws = null;
    }
  }

  return { send, close };
}

// ---- legacy named exports (kept for any older code) -----------------------

/** @deprecated retained for backward compat; new code should use `connect()`. */
export async function fetchSkills(): Promise<Skill[]> {
  try {
    const r = await fetch("/skills");
    if (!r.ok) return [];
    const j = await r.json();
    return Array.isArray(j) ? j : [];
  } catch {
    return [];
  }
}
