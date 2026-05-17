import type { Step } from '../lib/types';

function fmtElapsed(ms?: number): string {
  if (ms == null) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  const m = Math.floor(s / 60);
  const rs = Math.round(s % 60);
  return `${m}m ${rs}s`;
}

export function LiveStatus({
  steps,
  totalElapsedMs,
  paused = false,
}: {
  steps: Step[];
  totalElapsedMs?: number;
  paused?: boolean;
}) {
  return (
    <section
      className="status fade-in"
      data-paused={paused ? 'true' : 'false'}
      aria-live="polite"
      aria-label="Agent activity"
    >
      <div className="status__kicker">
        <span className="status__kicker-state">
          {paused ? 'Paused — awaiting your review' : 'Running'}
        </span>
        {totalElapsedMs != null ? (
          <span className="status__elapsed">{fmtElapsed(totalElapsedMs)}</span>
        ) : null}
      </div>
      <ol className="status__list">
        {steps.map((step) => (
          <li key={step.id} className="status__step" data-state={step.state}>
            <span className="status__node" />
            <span className="status__label">
              <span className="status__icon" aria-hidden="true">{step.icon}</span>
              <span>{step.label}</span>
              {step.detail ? <span className="status__detail">{step.detail}</span> : null}
            </span>
            <span className="status__t">{fmtElapsed(step.elapsedMs)}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
