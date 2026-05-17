// Types for the coda end-user UI.
// Mirrors the WebSocket protocol described in the brief (§8).

export type RuntimeChipKind = 'model' | 'mcp' | 'sub' | 'skill';

export type RuntimeChip = {
  kind: RuntimeChipKind;
  label: string;
};

export type Skill = {
  name: string;
  display_name: string;
  description: string;
  emoji?: string;
  signature: string;
};

export type StepState = 'current' | 'done' | 'failed';

export type Step = {
  id: string;
  /** emoji rendered before the label */
  icon: string;
  /** humanized label, e.g. "Reading recent emails" */
  label: string;
  /** optional secondary monospaced detail, e.g. "newer_than:1d in:inbox" */
  detail?: string;
  state: StepState;
  /** seconds spent in this step (live for `current`, frozen for `done`/`failed`) */
  elapsedMs?: number;
};

export type ApprovalAction =
  | { type: 'send_email'; id: string; payload: EmailPayload }
  | { type: 'shell_command'; id: string; payload: { command: string; cwd?: string } }
  | { type: 'sql_query'; id: string; payload: { query: string; database?: string } }
  | { type: 'slack_message'; id: string; payload: { channel: string; text: string } }
  | { type: string; id: string; payload: unknown };

export type EmailPayload = {
  to: string;
  cc?: string;
  bcc?: string;
  subject: string;
  body: string;
};

export type ApprovalOutcome =
  | { kind: 'accept'; payload?: unknown; notes?: string }
  | { kind: 'modify'; payload: unknown; notes?: string }
  | { kind: 'reject'; reason?: string };

export type ApprovalResolution = {
  outcome: 'sent' | 'edited' | 'discarded';
  /** what to show in the collapsed header after resolution */
  summary: string;
};

export type Message =
  | { kind: 'user'; id: string; text: string }
  | { kind: 'agent'; id: string; markdown: string }
  | { kind: 'approval'; id: string; action: ApprovalAction; resolution?: ApprovalResolution }
  | { kind: 'error'; id: string; message: string; traceback?: string }
  | {
      kind: 'run-summary';
      id: string;
      elapsedS: number;
      turns: number;
      tokensIn: number;
      tokensOut: number;
    };

export type RunState = 'idle' | 'running' | 'awaiting_approval' | 'done' | 'error';
