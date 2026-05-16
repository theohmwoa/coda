# On-Call Triage Report

| Field | Value |
|-------|-------|
| **Repo** | anthropics/claude-agent-sdk-python |
| **Date** | 2026-05-16 |
| **Period** | Last 7 days (updated >= 2026-05-09) |
| **Total items** | 23 |
| **Critical** | 1 |
| **High** | 9 |
| **Medium** | 8 |
| **Low** | 5 |

## 🔴 Critical (1)

### #502 [BUG] StructuredOutput tool wraps data in 'output' field causing schema validation to fail
- **Type**: Issue
- **Category**: bug
- **Author**: @AgentWrapper
- **Updated**: 2026-05-13
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Non-deterministic wrapping of structured output in an extra 'output' field causes intermittent schema validation failures. This is a production-breaking defect in the StructuredOutput tool with direct impact: (1) ResultMessage.structured_output becomes None, (2) agents see conflicting validation errors and degrade output quality (47 items → 1 item), (3) identical prompts succeed or fail randomly. Issue created 2026-01-21, last updated 2026-05-13 (3 days ago) but remains unresolved. Critical impact on agent SDK core functionality warrants immediate attention.

## 🟠 High (9)

### #963 fix(transport): guard against None allowed_tools in _apply_skills_defaults
- **Type**: PR
- **Category**: bug
- **Author**: @Aispotlightlab
- **Updated**: 2026-05-16
- **Draft**: No
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Production-breaking bug: SubprocessCLITransport crashes with TypeError when allowed_tools is None, preventing SDK initialization before CLI starts. Valid use case (wrapper libraries expressing 'no restriction' via None) is unhandled. Affects current versions (0.1.81, 0.2.82). High severity for affected users but not a security or data-loss issue. PR is freshly submitted (today) and doesn't require author follow-up.

### #962 fix: spill long system prompts on Windows
- **Type**: PR
- **Category**: bug
- **Author**: @MukundaKatta
- **Updated**: 2026-05-15
- **Draft**: No
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Windows-specific bug fix addressing command-line length limitations (Issue #883) that prevent users with long system prompts from using the tool. The PR implements a solid solution by converting oversized inline args to temporary file args, includes proper cleanup, better error reporting, and lists all required checks. Fresh PR ready for review; no action needed from author.

### #961 fix: preserve error context in CLI failure exceptions
- **Type**: PR
- **Category**: bug
- **Author**: @itxaiohanglover
- **Updated**: 2026-05-15
- **Draft**: No
- **Labels**: none
- **Ping**: no ping
- **Rationale**: This PR fixes a critical usability bug where error context is destroyed during CLI subprocess failures, making diagnostics impossible. The issue spans three layers (subprocess stderr handling, exception stringification, and bare exception re-raising) and directly impacts all users trying to debug CLI failures. High severity because broken error messages prevent effective troubleshooting. PR is fresh (created and updated same day) with clear explanation and appears ready for review; no author ping needed.

### #960 fix: atomically write forked session transcripts
- **Type**: PR
- **Category**: bug
- **Author**: @MukundaKatta
- **Updated**: 2026-05-15
- **Draft**: No
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Fresh PR fixing issue #906 with atomic writes for forked session transcripts. Addresses a data integrity issue by preventing partial/corrupt transcript files on write failures through temp file + atomic replace pattern. Includes validation evidence (passing tests, linting, and diff checks). High severity due to reliability implications—atomic write failures can leave corrupted session state. PR is ready for review; author doesn't need pinging.

### #715 fix: keep stdin open for can_use_tool control responses
- **Type**: PR
- **Category**: bug
- **Author**: @Ch1ldKing
- **Updated**: 2026-05-09
- **Draft**: No
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Bug fix for a defect where stdin is closed prematurely in `query()`, preventing valid `can_use_tool` permission control responses from being processed. Valid permission decisions are replaced by generic 'Stream closed' transport failures when using async prompt streams with `can_use_tool` configured. This breaks a core feature (permission control flow) in specific scenarios. PR has been open ~53 days but was updated recently (7 days ago), indicating active work; no ping needed to author.

### #952 Bundled CLI silently drops inbound TRACEPARENT on 2nd+ query() call when ~/.claude/ has state from a prior run
- **Type**: Issue
- **Category**: bug
- **Author**: @NBTDx
- **Updated**: 2026-05-12
- **Labels**: bug
- **Ping**: no ping
- **Rationale**: Functional bug where W3C trace context (TRACEPARENT) is silently dropped on 2nd+ query() invocations in the same process. Trace nesting works on first call but fails thereafter when ~/.claude/ state persists. Affects distributed tracing and observability. Clear reproduction path and root cause identified. Well-documented issue requiring team assignment for fix.

### #949 MessageParseError: missing signature in thinking block with claude-agent-sdk 0.1.79 and Anthropic-compatible backend
- **Type**: Issue
- **Category**: bug
- **Author**: @googs1025
- **Updated**: 2026-05-11
- **Labels**: bug
- **Ping**: no ping
- **Rationale**: Reproducible crash in claude-agent-sdk 0.1.79 where MessageParseError is raised when thinking content blocks omit the 'signature' field. The SDK parser incorrectly assumes all thinking blocks include this field. While it affects a specific subset (Anthropic-compatible backends with thinking blocks), it completely blocks SDK usage for those users. This is a regression—similar to #339—and needs urgent attention. The issue is fresh and well-documented with clear reproduction context.

### #882 [Bug] TaskNotificationMessage fires while subagent is still running (premature, not cross-turn leak)
- **Type**: Issue
- **Category**: bug
- **Author**: @Davidrjx
- **Updated**: 2026-05-14
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Race condition in Agent SDK's Task tool: TaskNotificationMessage fires prematurely while subagents are still executing, with actual completion arriving minutes later (76-850s delay). Non-deterministic behavior makes it particularly problematic. Affects core orchestrator functionality and parallel subagent coordination. Well-documented issue with clear symptom description. High severity as a correctness bug in SDK infrastructure, though not security or data-loss critical. Does not require author follow-up—needs development team investigation.

### #798 Error context destroyed through exception flattening — CLI failures produce unhelpful 'Command failed with exit code 1'
- **Type**: Issue
- **Category**: bug
- **Author**: @itsnotatbug
- **Updated**: 2026-05-15
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Systematic error handling defect that systematically destroys debugging context through a 3-stage flattening chain (ProcessError → string → bare Exception). Prevents SDK users from diagnosing root causes of CLI subprocess failures. Critical for debuggability given the frequency of CLI interactions in agent code. Well-documented with clear technical reproduction path. Recent activity (updated 2026-05-15) suggests maintainer awareness.

## 🟡 Medium (8)

### #950 fix(parser): clearer error when thinking block lacks signature
- **Type**: PR
- **Category**: bug
- **Author**: @googs1025
- **Updated**: 2026-05-12
- **Draft**: No
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Bug fix that improves error messaging when thinking blocks lack signatures (addresses #339, #949). Adds helpful context pointing to likely cause (non-Anthropic upstream) and includes CLAUDE_AGENT_SDK_SKIP_UNSIGNED_THINKING opt-out. Not urgent but meaningfully improves developer experience. PR is fresh (updated 2026-05-12) and in normal review queue, not stalled.

### #947 Support list-form system_prompt for subprocess CLI (fixes #899)
- **Type**: PR
- **Category**: feature
- **Author**: @zuhabul
- **Updated**: 2026-05-10
- **Draft**: No
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Feature PR implementing list-form system_prompt support in subprocess CLI (addresses #899). Clear, focused approach: writes temporary JSON file and passes via --system-prompt-file flag to forward structured content blocks to Anthropic. Not stale (6 days old), appears ready for review with no obvious blockers requiring author attention.

### #944 feat: add subprocess environment inheritance control
- **Type**: PR
- **Category**: feature
- **Author**: @httpsVishu
- **Updated**: 2026-05-09
- **Draft**: No
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Feature addition that introduces an explicit `inherit_env` option to ClaudeAgentOptions, allowing control over subprocess environment inheritance. Maintains backward compatibility by defaulting to current behavior (inherit_env=True). PR is fresh (one week old), well-tested with before/after code examples, and ready for review. Not stalled or low-priority, but doesn't require author ping—needs reviewer attention.

### #942 Fix #941: Python Agent SDK docs omit public options/types still present in `clau
- **Type**: PR
- **Category**: review_needed
- **Author**: @sherlock-488
- **Updated**: 2026-05-09
- **Draft**: Yes
- **Labels**: none
- **Ping**: sherlock-488
- **Rationale**: Adds regression test coverage for public API fields and types in the Python Agent SDK that were previously untested. Fixes issue #941 with clear scope: session/options fields, hook/tool-use types, permission context, and PermissionMode. Tests pass and linting is clean. Draft status suggests final polish before review—author should confirm readiness to move out of draft and address any reviewer feedback.

### #958 Bug: Autocompact is thrashing due to rapid context refill
- **Type**: Issue
- **Category**: bug
- **Author**: @naidanLuo
- **Updated**: 2026-05-15
- **Labels**: bug
- **Ping**: no ping
- **Rationale**: Legitimate bug report about autocompact thrashing when context refills rapidly within 3 turns of a previous compact. The issue is clear and reproducible despite incomplete steps. Medium severity because it degrades user experience but has a workaround (/clear) and isn't production-breaking. Requires developer investigation of autocompact logic to prevent rapid context refill cycles, but this is a triage issue for maintainers rather than a stalled PR needing author attention.

### #941 Python Agent SDK docs omit public options/types still present in `claude-agent-sdk==0.1.80`
- **Type**: Issue
- **Category**: docs
- **Author**: @oubeichen
- **Updated**: 2026-05-09
- **Labels**: documentation
- **Ping**: no ping
- **Rationale**: Documentation gap: public types and options present in claude-agent-sdk v0.1.80 (PermissionMode, ToolPermissionContext, DeferredToolUse, HookEventMessage, TaskBudget, SystemPromptFile) are missing from the official reference docs. Some were present in v0.1.74 and appear to have been removed accidentally. This is a regression affecting users trying to reference the complete API. The issue is well-documented with reproduction code but not urgent (doesn't break functionality). Needs docs maintainer attention to restore missing entries.

### #934 discussion: ClaudeAgentOptions.env only supports implicit parent-env inheritance with no documented isolated environment mode
- **Type**: Issue
- **Category**: docs
- **Author**: @httpsVishu
- **Updated**: 2026-05-14
- **Labels**: documentation, enhancement, question
- **Ping**: no ping
- **Rationale**: User has identified that ClaudeAgentOptions.env always inherits the parent environment when spawning the subprocess, but this behavior is not explicitly documented. While the implementation works as intended (env acts as an override/addition mechanism), the lack of clarity could confuse users into thinking an isolated environment mode exists or is possible. This is a documentation gap that maintainers should address by clarifying the inheritance behavior and documenting whether isolated environments are a supported use case.

### #848 [Feature Request] Seed externally-persisted conversation history into a fresh ClaudeSDKClient as role-based messages
- **Type**: Issue
- **Category**: feature
- **Author**: @KBB99
- **Updated**: 2026-05-15
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Well-articulated feature request for seeding externally-persisted conversation history into ClaudeSDKClient as structured messages. Addresses a real integration pattern (Bedrock AgentCore, stateless containers) that currently lacks native support. Not a bug or blocker, but represents a capability gap affecting SDK users managing conversation history in external systems. Recently updated (5 days ago), indicating some team engagement. Needs architectural review to determine API design.

## 🟢 Low (5)

### #954 examples: add Synap memory integration example
- **Type**: PR
- **Category**: feature
- **Author**: @visy-ani
- **Updated**: 2026-05-14
- **Draft**: No
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Adds example code demonstrating Synap memory integration with Claude Agent. Author has completed manual testing and verified both hook injection and MCP tool functionality. This is new documentation/example content with no core code changes, making it low-risk and low-priority. Ready for review but does not require urgent attention.

### #945 Add example: external HTTP MCP server (TensorFeed.ai)
- **Type**: PR
- **Category**: docs
- **Author**: @RipperMercs
- **Updated**: 2026-05-09
- **Draft**: No
- **Labels**: none
- **Ping**: RipperMercs
- **Rationale**: This PR adds a valuable documentation example demonstrating external HTTP MCP server integration with the Claude Agent SDK using TensorFeed.ai. However, the PR description is incomplete—it ends abruptly mid-sentence ("so the u")—and has not been updated in the seven days since creation. The author should complete the description to clarify the full scope and rationale before review proceeds.

### #691 feat(types): add PostCompact hook event support
- **Type**: PR
- **Category**: feature
- **Author**: @husniadil
- **Updated**: 2026-05-15
- **Draft**: No
- **Labels**: none
- **Ping**: no ping
- **Rationale**: Feature PR adding PostCompact hook event support to the Python SDK, matching CLI v2.1.76+ specification. Includes comprehensive type definitions (PostCompactHookInput, PostCompactHookSpecificOutput), proper union type updates, and tests covering both manual and auto triggers plus behavioral validation. PR was updated 2026-05-15 (yesterday), indicating active development. No breaking changes, security concerns, or blocking issues. Straightforward feature addition that doesn't require immediate attention.

### #624 Refine permission control typing
- **Type**: PR
- **Category**: review_needed
- **Author**: @shuofengzhang
- **Updated**: 2026-05-14
- **Draft**: No
- **Labels**: none
- **Ping**: no ping
- **Rationale**: This is a typing refinement PR for permission control in the SDK. Tests pass, the PR is ready for review, and there's no indication that author action is needed. It improves code quality by aligning control request typing with existing permission models and eliminating fallback to `Any`/`str`, but doesn't address blocking issues or production concerns. Recent activity (updated May 14) and clear description indicate it's ready for maintainer review.

### #948 No way to access the resolved system prompt (from CLAUDE.md and similar files) via the Python SDK
- **Type**: Issue
- **Category**: feature
- **Author**: @Nithish-R-Nair
- **Updated**: 2026-05-14
- **Labels**: enhancement, question
- **Ping**: no ping
- **Rationale**: Feature request for SDK observability—user seeks visibility into the resolved system prompt (including CLAUDE.md content) for debugging agent behavior. Valid use case but not urgent or blocking; enhancement to SDK transparency. No author follow-up needed; this is a capability question/feature proposal for maintainers to evaluate.
