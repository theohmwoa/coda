import { useState } from 'react';

export function ErrorCard({ message, traceback }: { message: string; traceback?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="error-card fade-in" role="alert">
      <div className="error-card__head">Run failed</div>
      <div className="error-card__msg">{message}</div>
      {traceback ? (
        <>
          <button
            type="button"
            className="error-card__trace-toggle"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? 'hide traceback' : 'show traceback'}
          </button>
          {open ? <pre className="error-card__trace">{traceback}</pre> : null}
        </>
      ) : null}
    </div>
  );
}
