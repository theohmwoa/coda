/**
 * The core protocol types.
 *
 * The library is a thin layer over four primitives:
 *
 *   - `Action`            — the agent's drafted intent (an email to send,
 *                           an issue to create, an SQL query to run).
 *   - `Edit`              — what the human did when shown the action's UI.
 *   - `FeedbackStrategy`  — how the user's edit reshapes the agent's chain.
 *   - `RewriteResult`     — the protocol-neutral instructions a runtime
 *                           adapter consumes.
 *
 * Adapter packages translate `RewriteResult.chainOps` into framework-specific
 * shapes (Vercel AI SDK message arrays, LangGraph state edits, raw provider
 * tool-use loops).
 */

/**
 * An agent's drafted intent. Generic over the payload so each action type
 * keeps its own typed schema. The library doesn't dictate what payloads
 * look like — `send_email` carries an email object, `run_query` carries a
 * SQL string, etc.
 */
export interface Action<TPayload = unknown> {
  /** Stable identifier for the action's category. e.g. "send_email". */
  type: string;
  /** Unique identifier for this specific action invocation. */
  id: string;
  /** The agent's drafted payload. */
  payload: TPayload;
  /** Optional context that adapters thread through from the agent runtime. */
  context?: ActionContext;
}

/** Optional metadata attached to an action by the runtime adapter. */
export interface ActionContext {
  /** Step id in the agent's chain (forge / similar) this action came from. */
  stepId?: string;
  /** Conversation / run id the action belongs to. */
  runId?: string;
  /** Free-form passthrough metadata. */
  meta?: Record<string, unknown>;
}

/**
 * What the human did when the action's UI was presented. Three outcomes
 * cover the meaningful options. UIs that only need approve / reject can
 * ignore `modify` entirely; UIs that always edit can ignore `accept`.
 */
export interface Edit<TPayload = unknown> {
  /** The `Action.id` this edit applies to. */
  actionId: string;
  outcome: EditOutcome<TPayload>;
}

export type EditOutcome<TPayload = unknown> =
  | { kind: "accept" }
  | { kind: "modify"; payload: TPayload; notes?: string }
  | { kind: "reject"; reason?: string };

/**
 * How the user's edit reshapes the agent's chain. The four strategies
 * trade off agent-coherence vs in-session learning vs token cost. See
 * the README's strategy table for the comparison.
 */
export type FeedbackStrategy =
  | { kind: "silent" }
  | { kind: "visible" }
  | { kind: "retry"; maxRetries?: number }
  | {
      kind: "constraint";
      /**
       * Caller-supplied rule extractor: given the action + edit, produce a
       * single-sentence system-prompt addition that captures the lesson
       * for the rest of the session.
       */
      extractRule: (action: Action, edit: Edit) => string;
    };

/**
 * Result of applying a strategy to a (action, edit) pair. The agent
 * runtime consumes this and reshapes its chain accordingly.
 */
export interface RewriteResult<TPayload = unknown> {
  /**
   * What the agent should treat as its draft going forward. Equal to the
   * original `action.payload` if the user accepted, the user's edit if
   * the user modified, or a passthrough for retry / rejected (where the
   * agent is expected to redraft or stop entirely).
   */
  effectivePayload: TPayload;
  /** Provider-neutral chain operations the runtime should apply. */
  chainOps: ChainOp[];
  /** True iff the agent should be re-prompted to redraft (retry strategy). */
  retry: boolean;
  /** True iff the user rejected outright and the action should not proceed. */
  rejected: boolean;
  /**
   * System-prompt addition derived from the edit (constraint strategy).
   * Echoed here so the runtime can also persist it to a session-level
   * constraints store, not only as a one-shot chain op.
   */
  systemConstraint?: string;
}

/**
 * A protocol-neutral instruction for the runtime: "in your chain
 * representation, do this." Adapters translate these into framework-
 * specific edits.
 */
export type ChainOp =
  | {
      /** Replace the existing step's payload with `payload`. */
      kind: "replace_step";
      stepId: string;
      payload: unknown;
    }
  | {
      /** Append a new message after the named step (or at the end if omitted). */
      kind: "append_message";
      afterStepId?: string;
      role: string;
      content: string;
    }
  | {
      /** Append a system-prompt constraint that lives for the rest of the session. */
      kind: "append_system_constraint";
      content: string;
    };
