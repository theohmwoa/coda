/**
 * EmailModal — review/edit/send a drafted email.
 *
 * Adapted from the agent-feedback-ui catalog's `email.tsx`. The
 * upstream component exposed an Action/Edit envelope shape; this
 * version inverts the call shape to suit the coda end-user UI:
 *
 *   <EmailModal
 *     value={{to, subject, body, cc?, bcc?}}
 *     onSend={(payload, notes) => ...}         // user pressed Send without editing
 *     onEditAndSend={(payload, notes) => ...}  // user edited, then sent
 *     onDiscard={(reason?) => ...}             // user discarded
 *   />
 *
 * Switching back to a proper `agent-feedback-ui` git/npm dep should be
 * a one-line import change once upstream ships 0.1.0; until then this
 * file is the source of truth for the email-review surface.
 *
 * Class names follow the `afu-email-modal__*` convention so the styles
 * in src/styles.css can theme it without forking.
 */

import { useMemo, useState } from "react";

export interface EmailPayload {
  to: string;
  cc?: string;
  bcc?: string;
  subject: string;
  body: string;
}

interface Props {
  value: EmailPayload;
  onSend: (payload: EmailPayload, notes?: string) => void;
  onEditAndSend: (payload: EmailPayload, notes?: string) => void;
  onDiscard: (reason?: string) => void;
}

export function EmailModal({ value, onSend, onEditAndSend, onDiscard }: Props) {
  const [to, setTo] = useState(value.to);
  const [cc, setCc] = useState(value.cc ?? "");
  const [bcc, setBcc] = useState(value.bcc ?? "");
  const [subject, setSubject] = useState(value.subject);
  const [body, setBody] = useState(value.body);
  const [notes, setNotes] = useState("");

  const isEdited = useMemo(
    () =>
      to !== value.to ||
      cc !== (value.cc ?? "") ||
      bcc !== (value.bcc ?? "") ||
      subject !== value.subject ||
      body !== value.body,
    [value, to, cc, bcc, subject, body],
  );

  function buildPayload(): EmailPayload {
    const p: EmailPayload = { to, subject, body };
    if (cc) p.cc = cc;
    if (bcc) p.bcc = bcc;
    return p;
  }

  function handleSend() {
    if (!isEdited) onSend(value, notes || undefined);
    else onEditAndSend(buildPayload(), notes || undefined);
  }

  function handleDiscard() {
    onDiscard(notes || undefined);
  }

  const idBase = useMemo(() => `afu-${Math.random().toString(36).slice(2, 8)}`, []);

  return (
    <div className="afu-modal afu-email-modal" role="dialog" aria-label="Review email before sending">
      <div className="afu-email-modal__field">
        <label className="afu-email-modal__label" htmlFor={`${idBase}-to`}>To</label>
        <input
          id={`${idBase}-to`}
          className="afu-email-modal__input"
          type="text"
          value={to}
          onChange={(e) => setTo(e.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
      </div>

      <div className="afu-email-modal__field afu-email-modal__field--cc">
        <label className="afu-email-modal__label" htmlFor={`${idBase}-cc`}>Cc</label>
        <input
          id={`${idBase}-cc`}
          className="afu-email-modal__input"
          type="text"
          value={cc}
          onChange={(e) => setCc(e.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
      </div>

      <div className="afu-email-modal__field afu-email-modal__field--bcc">
        <label className="afu-email-modal__label" htmlFor={`${idBase}-bcc`}>Bcc</label>
        <input
          id={`${idBase}-bcc`}
          className="afu-email-modal__input"
          type="text"
          value={bcc}
          onChange={(e) => setBcc(e.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
      </div>

      <div className="afu-email-modal__field">
        <label className="afu-email-modal__label" htmlFor={`${idBase}-subj`}>Subject</label>
        <input
          id={`${idBase}-subj`}
          className="afu-email-modal__input"
          type="text"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
        />
      </div>

      <textarea
        className="afu-email-modal__body"
        value={body}
        onChange={(e) => setBody(e.target.value)}
        rows={10}
        aria-label="Body"
      />

      <div className="afu-email-modal__notes">
        <label className="afu-email-modal__label" htmlFor={`${idBase}-notes`}>
          Notes for the agent (optional)
        </label>
        <input
          id={`${idBase}-notes`}
          className="afu-email-modal__input"
          type="text"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Anything the agent should learn from this edit?"
        />
      </div>

      <div className="afu-email-modal__actions">
        <button
          type="button"
          className="afu-email-modal__btn afu-email-modal__btn--secondary"
          onClick={handleDiscard}
        >
          Discard
        </button>
        <button
          type="button"
          className="afu-email-modal__btn afu-email-modal__btn--primary"
          onClick={handleSend}
        >
          {isEdited ? "Send edited" : "Send"}
        </button>
      </div>
    </div>
  );
}
