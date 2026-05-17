import { useState } from 'react';
import type {
  ApprovalAction,
  ApprovalOutcome,
  ApprovalResolution,
  EmailPayload,
} from '../lib/types';
// EmailModal is vendored from agent-feedback-ui; do not edit that file.
// It exposes the .afu-email-modal__* class names we style in styles.css.
import { EmailModal } from '../lib/agent-feedback/email';

type Props = {
  action: ApprovalAction;
  resolution?: ApprovalResolution;
  /** called when the user submits/edits/discards — caller sends the
   *  approval_response frame down the WebSocket. */
  onResolve: (action: ApprovalAction, outcome: ApprovalOutcome) => void;
  /** start collapsed? defaults to false (expanded on arrival, per brief) */
  defaultCollapsed?: boolean;
};

function ChevronIcon() {
  return (
    <svg className="approval__chevron" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function summarizeEmail(p: EmailPayload): string {
  return `to ${p.to} · ${p.subject}`;
}

function headerFor(action: ApprovalAction, resolution?: ApprovalResolution): {
  icon: string;
  title: string;
  meta: string;
} {
  if (resolution) {
    return {
      icon: action.type === 'send_email' ? '📧' : action.type === 'shell_command' ? '⌘' : action.type === 'sql_query' ? '🗃' : '⚡',
      title: resolution.summary,
      meta: '',
    };
  }
  switch (action.type) {
    case 'send_email': {
      const p = action.payload as EmailPayload;
      return { icon: '📧', title: 'Email ready for review', meta: summarizeEmail(p) };
    }
    case 'shell_command': {
      const p = action.payload as { command: string };
      return { icon: '⌘', title: 'Shell command ready to run', meta: p.command };
    }
    case 'sql_query': {
      const p = action.payload as { query: string };
      return { icon: '🗃', title: 'SQL query ready to run', meta: p.query };
    }
    case 'slack_message': {
      const p = action.payload as { channel: string };
      return { icon: '💬', title: 'Slack message ready to send', meta: `to ${p.channel}` };
    }
    default:
      return { icon: '⚡', title: `${action.type} ready to run`, meta: '' };
  }
}

export function ApprovalCard({ action, resolution, onResolve, defaultCollapsed }: Props) {
  const startExpanded = !resolution && !defaultCollapsed;
  const [expanded, setExpanded] = useState(startExpanded);
  const { icon, title, meta } = headerFor(action, resolution);

  const resolvedKind = resolution
    ? resolution.outcome === 'sent'
      ? 'sent'
      : resolution.outcome === 'edited'
      ? 'edited'
      : 'discarded'
    : null;

  return (
    <article
      className="approval fade-in"
      data-expanded={expanded ? 'true' : 'false'}
      data-resolved={resolvedKind ?? 'false'}
    >
      <button
        type="button"
        className="approval__header"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        <span className="approval__icon" aria-hidden="true">{icon}</span>
        <span className="approval__title-line">
          <span className="approval__title">{title}</span>
          {meta ? (
            <>
              <span className="approval__sep">·</span>
              <span className="approval__meta">{meta}</span>
            </>
          ) : null}
          {resolution ? (
            <span className="approval__status-tag">{resolution.outcome.toUpperCase()}</span>
          ) : null}
        </span>
        <ChevronIcon />
      </button>

      {expanded && !resolution ? (
        <>
          <div className="approval__pending-hint" aria-live="polite">
            agent paused on approve() — your call
          </div>
          <div className="approval__body">
            {renderBody(action, onResolve)}
          </div>
        </>
      ) : null}
    </article>
  );
}

function renderBody(
  action: ApprovalAction,
  onResolve: (a: ApprovalAction, o: ApprovalOutcome) => void,
) {
  switch (action.type) {
    case 'send_email':
      return (
        <EmailModal
          value={action.payload as EmailPayload}
          onSend={(payload, notes) =>
            onResolve(action, { kind: 'accept', payload, notes })
          }
          onEditAndSend={(payload, notes) =>
            onResolve(action, { kind: 'modify', payload, notes })
          }
          onDiscard={(reason) => onResolve(action, { kind: 'reject', reason })}
        />
      );
    default:
      // type-aware fallback: JSON preview + Accept / Reject
      return (
        <div className="approval__fallback">
          <pre className="approval__json">{JSON.stringify(action.payload, null, 2)}</pre>
          <div className="approval__fallback-actions">
            <button
              type="button"
              className="afu-email-modal__btn afu-email-modal__btn--danger"
              onClick={() => onResolve(action, { kind: 'reject' })}
            >
              Reject
            </button>
            <button
              type="button"
              className="afu-email-modal__btn afu-email-modal__btn--primary"
              onClick={() => onResolve(action, { kind: 'accept' })}
            >
              Accept
            </button>
          </div>
        </div>
      );
  }
}
