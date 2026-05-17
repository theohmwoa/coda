import type { RuntimeChip as Chip } from '../lib/types';
import { RuntimeChip } from './RuntimeChip';

export function Header({ chips }: { chips: Chip[] }) {
  return (
    <header className="header">
      <div className="header__brand">
        <span className="brand-mark">coda<span className="brand-dot" /></span>
        <span className="brand-kicker">personal agent</span>
      </div>
      <div className="chips" role="status" aria-label="active runtime">
        {chips.map((c, i) => (
          <RuntimeChip key={`${c.kind}-${c.label}-${i}`} chip={c} />
        ))}
      </div>
    </header>
  );
}
