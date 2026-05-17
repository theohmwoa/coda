import type { RuntimeChip as Chip } from '../lib/types';

const PREFIX: Record<Chip['kind'], string> = {
  model: '',
  mcp: 'mcp:',
  sub: 'sub:',
  skill: 'skill:',
};

const TYPE_LABEL: Record<Chip['kind'], string> = {
  model: 'model',
  mcp: 'mcp',
  sub: 'sub',
  skill: 'skill',
};

export function RuntimeChip({ chip }: { chip: Chip }) {
  return (
    <span className={`chip chip--${chip.kind}`} title={`${TYPE_LABEL[chip.kind]} — ${chip.label}`}>
      <span className="chip__dot" />
      <span>{PREFIX[chip.kind]}{chip.label}</span>
    </span>
  );
}
