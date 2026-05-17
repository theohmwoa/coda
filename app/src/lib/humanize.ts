// Map raw `event` frames from the WebSocket into user-facing step strings.
// The end-user UI should never see raw event types (tool_called, etc.) — every
// event we want to surface gets a friendly emoji + sentence here.
//
// Extend this freely as the protocol grows; just keep the output shape stable.

import type { Step } from './types';

type RawEvent = {
  type: string;
  ts: string;
  payload: Record<string, unknown>;
};

/**
 * Convert one raw event into a `Step` ready to render in the live status pane,
 * or `null` if the event should be hidden from end users.
 *
 * The caller is responsible for state transitions (current → done) and for
 * stamping `id`/`elapsedMs`.
 */
export function humanize(ev: RawEvent): Omit<Step, 'id' | 'state' | 'elapsedMs'> | null {
  const p = ev.payload || {};

  switch (ev.type) {
    case 'llm_request':
      return { icon: '🤖', label: 'Thinking' };

    case 'llm_response':
      // intentionally hidden — done state is implied by the next step
      return null;

    case 'code_emitted':
      // end users don't see code; show a generic "Working" beat instead
      return { icon: '✦', label: 'Working' };

    case 'code_executed':
    case 'line_executed':
      // too noisy for end users
      return null;

    case 'tool_called': {
      const tool = String(p.tool || p.name || '');
      const args = (p.args as Record<string, unknown>) || {};
      return humanizeToolCall(tool, args);
    }

    case 'subagent_called': {
      const name = String(p.name || 'sub-agent');
      const task = p.task ? ` — ${String(p.task)}` : '';
      return { icon: '🛟', label: `Asking ${prettyName(name)}${task}` };
    }

    case 'subagent_returned':
      return null;

    case 'execution_error': {
      const msg = String(p.message || 'Execution error');
      return { icon: '⚠️', label: 'Hit an error', detail: msg };
    }

    case 'turn_complete':
      return null;

    default:
      // Unknown event types are silently dropped — never show raw types.
      return null;
  }
}

function humanizeToolCall(
  tool: string,
  args: Record<string, unknown>,
): Omit<Step, 'id' | 'state' | 'elapsedMs'> | null {
  switch (tool) {
    // ── native primitives ──────────────────────────────────────
    case 'ls':       return { icon: '📂', label: 'Listing files', detail: String(args.path ?? '') || undefined };
    case 'glob':     return { icon: '🔎', label: 'Searching files', detail: String(args.pattern ?? '') };
    case 'grep':     return { icon: '🔎', label: 'Searching content', detail: String(args.pattern ?? '') };
    case 'read':     return { icon: '📄', label: 'Reading file', detail: shortPath(args.path) };
    case 'write':    return { icon: '📝', label: 'Writing file', detail: shortPath(args.path) };
    case 'edit':     return { icon: '✏️', label: 'Editing file', detail: shortPath(args.path) };
    case 'bash':     return { icon: '⚙️', label: 'Running shell', detail: truncate(String(args.cmd ?? ''), 80) };

    // ── Gmail MCP (gongrzhe/gmail-mcp-server) — bare tool names ─
    // Coda's MCPRuntime emits tool_called with the un-prefixed tool name.
    case 'search_emails':
    case 'gmail.search_emails':
    case 'gmail.search':
    case 'gmail.list_messages': {
      const q = String(args.query || args.q || '');
      return { icon: '📧', label: 'Reading recent emails', detail: q || undefined };
    }
    case 'read_email':
    case 'gmail.read_email':
    case 'gmail.get_message': {
      const id = args.messageId ?? args.id;
      return { icon: '📧', label: 'Reading one email', detail: id ? String(id).slice(0, 10) : undefined };
    }
    case 'send_email':
    case 'gmail.send_email':
    case 'gmail.send_message':
      return { icon: '✉️', label: `Sending email to ${prettyRecipient(args.to)}` };
    case 'draft_email':
    case 'gmail.draft_email':
    case 'compose_email':
      return { icon: '✍️', label: 'Drafting an email' };
    case 'modify_email':
    case 'batch_modify_emails':
      return { icon: '🏷️', label: 'Updating email labels' };
    case 'list_email_labels':
    case 'gmail.list_email_labels':
      return { icon: '🏷️', label: 'Listing labels' };

    // ── inline tools ───────────────────────────────────────────
    case 'post_to_discord':
      return { icon: '💬', label: 'Posting to Discord' };

    // approve() is rendered via the approval_needed frame instead;
    // hide its tool_called noise.
    case 'approve':
      return null;

    default:
      // Unknown tools — generic "Working" beat, never the raw name.
      return { icon: '✦', label: 'Working' };
  }
}

function shortPath(p: unknown): string | undefined {
  if (!p) return undefined;
  const s = String(p);
  const parts = s.split('/');
  return parts.length <= 2 ? s : parts.slice(-2).join('/');
}

function prettyRecipient(v: unknown): string {
  if (!v) return 'recipient';
  const s = Array.isArray(v) ? String(v[0] ?? '') : String(v);
  const m = s.match(/^([^<]+)\s*</);
  return (m?.[1] ?? s).trim().replace(/"/g, '') || 'recipient';
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

function prettyName(s: string): string {
  return s.replace(/_/g, ' ');
}
