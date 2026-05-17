import { useEffect, useRef, useState } from 'react';

type Props = {
  disabled?: boolean;
  /** caller-controlled status label below the textarea ("running…", etc.) */
  statusHint?: string;
  onSubmit: (text: string) => void;
};

function SendIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M3 8l10-5-3 12-2.5-5L3 8z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

export function InputDock({ disabled, statusHint, onSubmit }: Props) {
  const [value, setValue] = useState('');
  const taRef = useRef<HTMLTextAreaElement | null>(null);

  // auto-grow
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 9 * 24) + 'px';
  }, [value]);

  function send() {
    const t = value.trim();
    if (!t || disabled) return;
    onSubmit(t);
    setValue('');
  }

  function onKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="dock">
      <div className="dock__inner">
        <div className="dock__field">
          <textarea
            ref={taRef}
            className="dock__textarea"
            placeholder={
              disabled
                ? 'Agent is working — your turn in a moment.'
                : 'Type a task — e.g. "give me a morning brief".'
            }
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={onKey}
            disabled={disabled}
            rows={1}
          />
          <button
            type="button"
            className="dock__send"
            onClick={send}
            disabled={disabled || value.trim().length === 0}
            aria-label="Send"
          >
            <SendIcon />
          </button>
        </div>
        <div className={`dock__hint ${statusHint ? 'dock__hint--running' : ''}`}>
          <span>
            {statusHint ?? (
              <>
                <kbd>⌘</kbd><kbd>↵</kbd> to send
              </>
            )}
          </span>
          <span>type <span className="mono">/</span> for skills</span>
        </div>
      </div>
    </div>
  );
}
