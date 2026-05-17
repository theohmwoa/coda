import { useEffect, useMemo, useRef, useState } from 'react';
import { Header } from './components/Header';
import { SkillGrid } from './components/SkillCard';
import { UserMessage } from './components/UserMessage';
import { AgentResponse } from './components/AgentResponse';
import { LiveStatus } from './components/LiveStatus';
import { ApprovalCard } from './components/ApprovalCard';
import { ErrorCard } from './components/ErrorCard';
import { InputDock } from './components/InputDock';
import type {
  ApprovalAction,
  ApprovalOutcome,
  Message,
  RuntimeChip,
  RunState,
  Skill,
  Step,
} from './lib/types';
import { humanize } from './lib/humanize';
// api.ts is the existing WebSocket client; we just consume from it.
import { connect, type ServerFrame } from './api';

export default function App() {
  const [chips, setChips] = useState<RuntimeChip[]>([]);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [steps, setSteps] = useState<Step[]>([]);
  const [runState, setRunState] = useState<RunState>('idle');
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const sendRef = useRef<((frame: unknown) => void) | null>(null);

  // tick for the elapsed-time meter while running
  useEffect(() => {
    if (runState !== 'running' && runState !== 'awaiting_approval') return;
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, [runState]);

  useEffect(() => {
    const conn = connect({
      onFrame: handleFrame,
      onSkills: (s) => setSkills(s),
    });
    sendRef.current = conn.send;
    return conn.close;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleFrame(frame: ServerFrame) {
    switch (frame.kind) {
      case 'ready': {
        const next: RuntimeChip[] = [
          { kind: 'model', label: frame.model },
          ...frame.mcp_servers.map((m) => ({ kind: 'mcp' as const, label: m })),
          ...frame.sub_agents.map((s) => ({ kind: 'sub' as const, label: s })),
          ...frame.skills.map((s) => ({ kind: 'skill' as const, label: s })),
        ];
        setChips(next);
        break;
      }
      case 'event': {
        const h = humanize({ type: frame.type, ts: frame.ts, payload: frame.payload });
        if (!h) break;
        setSteps((prev) => transitionToStep(prev, h, now));
        break;
      }
      case 'approval_needed': {
        setRunState('awaiting_approval');
        setMessages((m) => [
          ...m,
          { kind: 'approval', id: frame.action.id, action: frame.action as ApprovalAction },
        ]);
        break;
      }
      case 'final': {
        setMessages((m) => [
          ...m,
          { kind: 'agent', id: `agent-${Date.now()}`, markdown: frame.text },
        ]);
        break;
      }
      case 'done': {
        setSteps([]);
        setRunState('done');
        setRunStartedAt(null);
        setMessages((m) => [
          ...m,
          {
            kind: 'run-summary',
            id: `done-${Date.now()}`,
            elapsedS: frame.elapsed_s,
            turns: frame.turns,
            tokensIn: frame.tokens_in,
            tokensOut: frame.tokens_out,
          },
        ]);
        break;
      }
      case 'error': {
        setRunState('error');
        setMessages((m) => [
          ...m,
          { kind: 'error', id: `err-${Date.now()}`, message: frame.error, traceback: frame.traceback },
        ]);
        break;
      }
    }
  }

  function transitionToStep(prev: Step[], next: Omit<Step, 'id' | 'state' | 'elapsedMs'>, nowMs: number): Step[] {
    const closed = prev.map((s, i) =>
      i === prev.length - 1 && s.state === 'current'
        ? { ...s, state: 'done' as const, elapsedMs: s.elapsedMs ?? 0 }
        : s,
    );
    return [
      ...closed,
      { id: `step-${nowMs}-${Math.random().toString(36).slice(2, 7)}`, state: 'current', ...next },
    ];
  }

  function submit(text: string) {
    setMessages((m) => [...m, { kind: 'user', id: `u-${Date.now()}`, text }]);
    setSteps([]);
    setRunStartedAt(Date.now());
    setRunState('running');
    sendRef.current?.({ kind: 'prompt', text });
  }

  function resolveApproval(action: ApprovalAction, outcome: ApprovalOutcome) {
    sendRef.current?.({
      kind: 'approval_response',
      action_id: action.id,
      edit: { actionId: action.id, outcome },
    });
    setMessages((m) =>
      m.map((msg) =>
        msg.kind === 'approval' && msg.id === action.id
          ? {
              ...msg,
              resolution: {
                outcome:
                  outcome.kind === 'accept' ? 'sent' : outcome.kind === 'modify' ? 'edited' : 'discarded',
                summary: summaryFor(action, outcome),
              },
            }
          : msg,
      ),
    );
    if (runState === 'awaiting_approval') setRunState('running');
  }

  const totalElapsedMs = runStartedAt ? now - runStartedAt : undefined;

  const inputDisabled = runState === 'running' || runState === 'awaiting_approval';
  const inputHint = useMemo(() => {
    if (runState === 'running') return 'running…';
    if (runState === 'awaiting_approval') return 'waiting on your review';
    return undefined;
  }, [runState]);

  const showSkillGrid = messages.length === 0 && runState === 'idle';

  return (
    <div className="app">
      <Header chips={chips} />
      <main className="main">
        <div className="column">
          {showSkillGrid ? (
            <>
              <div className="greeting">
                <h1 className="greeting__h">Good morning. <em>What's on the docket?</em></h1>
                <div className="greeting__sub">
                  Pick a skill below or type a task — coda will work through it and ask before sending anything.
                </div>
              </div>
              <SkillGrid skills={skills} onRun={(s) => submit(s.display_name)} />
            </>
          ) : null}

          {messages.map((msg) => {
            if (msg.kind === 'user') return <UserMessage key={msg.id} text={msg.text} />;
            if (msg.kind === 'agent') return <AgentResponse key={msg.id} markdown={msg.markdown} />;
            if (msg.kind === 'error') return <ErrorCard key={msg.id} message={msg.message} traceback={msg.traceback} />;
            if (msg.kind === 'approval')
              return (
                <ApprovalCard
                  key={msg.id}
                  action={msg.action}
                  resolution={msg.resolution}
                  onResolve={resolveApproval}
                />
              );
            if (msg.kind === 'run-summary')
              return (
                <div key={msg.id} className="run-summary">
                  <span className="run-summary__metric"><strong>{msg.elapsedS.toFixed(1)}s</strong> elapsed</span>
                  <span className="run-summary__sep">·</span>
                  <span className="run-summary__metric"><strong>{msg.turns}</strong> turns</span>
                  <span className="run-summary__sep">·</span>
                  <span className="run-summary__metric"><strong>{msg.tokensIn.toLocaleString()}</strong> in / <strong>{msg.tokensOut.toLocaleString()}</strong> out</span>
                </div>
              );
            return null;
          })}

          {(runState === 'running' || runState === 'awaiting_approval') && steps.length > 0 ? (
            <LiveStatus
              steps={steps}
              totalElapsedMs={totalElapsedMs}
              paused={runState === 'awaiting_approval'}
            />
          ) : null}
        </div>
      </main>
      <InputDock disabled={inputDisabled} statusHint={inputHint} onSubmit={submit} />
    </div>
  );
}

function summaryFor(action: ApprovalAction, outcome: ApprovalOutcome): string {
  if (action.type === 'send_email') {
    if (outcome.kind === 'accept') return 'Email sent';
    if (outcome.kind === 'modify') return 'Email edited & sent';
    return 'Email discarded';
  }
  if (outcome.kind === 'reject') return `${prettyType(action.type)} discarded`;
  if (outcome.kind === 'modify') return `${prettyType(action.type)} edited & ran`;
  return `${prettyType(action.type)} ran`;
}
function prettyType(t: string): string {
  return t.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
}
