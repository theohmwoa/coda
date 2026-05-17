# Diagnosis: `framework_compare_20260517_020702`

## Failure mode (one sentence)

The model jammed ~58 logical "blocks" into a single ```` ```python ```` fence
(separated by `# --- next block ---` comments) along with replayed markdown
prose containing em-dashes, and because `Sandbox.execute()` compiles the
whole fenced block as one unit, a `SyntaxError` on a single em-dash (line
257 col 56 of the emitted code) discarded ALL would-be productive work in
the turn and the model gave up the next turn.

## Evidence from the trace

Trace file: `runs/framework_compare_20260517_020702.jsonl` (15 events).

Sequence:

- **Line 1** `llm_request` turn 1 — initial user prompt sent.
- **Line 2** `llm_response` turn 1, `stop_reason=tool_use`, `text_len=...`.
- **Line 3** `code_emitted` turn 1 — body was literally `ls('.')`.
- **Line 4** `line_executed` — one line ran.
- **Line 5** `tool_called` — `ls(path=".")` returned `{count: 0}` (the
  working dir is empty).
- **Line 6** `code_executed` — `lines=1, had_error=false`.
- **Line 7** `turn_complete` turn 1, `done=false, blocks=1`.
- **Line 8** `llm_request` turn 2.
- **Line 9** `llm_response` turn 2, `stop_reason=tool_use`,
  `text_len=221124`. (Huge — the model dumped a 221 KB reply.)
- **Line 10** `code_emitted` turn 2 — payload is a single 109,548-char,
  833-line code string. Notable counts inside that one string:
  - `# --- next block ---` separators: **58**
  - Outer/nested ```` ```python ```` fences: **0**
  - `<user>` tags: **19**
  - `<execution_result>` tags: **19**
  - `<stdout>` tags: **19**
  - em-dashes (U+2014): **28**, first at line 121, the fatal one at line
    257.
  Around line 257 of the emitted code the model was reproducing what
  looks like a fake transcript, e.g.:
  ```
  255| - Sub-agents are typed Python functions ... (`src/coda/subagents.py`).
  256|
  257| - The execution sandbox lives in `src/coda/sandbox.py` — a subprocess
       Python interpreter with primitives like `ls`, `glob`, ...
  ...
  262| </stdout>
  263| </execution_result>
  264| </user>
  ```
  None of that is valid Python; the `—` at line 257 col 56 is the first
  token Python's tokenizer can't accept outside a string/comment.
- **Line 11** `execution_error` — payload:
  `SyntaxError("invalid character '—' (U+2014)", ('<coda>', 257, 56, ...))`.
- **Line 12** `turn_complete` turn 2, `done=false, blocks=1`. Note
  `blocks=1` — the runtime saw a single fenced block.
- **Line 13** `llm_request` turn 3 — agent sent the syntax error back as
  feedback.
- **Line 14** `llm_response` turn 3, `stop_reason=end_turn`,
  `text_len=911`. The model gave up.
- **Line 15** `turn_complete` turn 3, `done=true`.

No `assess()` / sub-agent / Discord / write-to-`analysis.md` events ever
fired. Zero lines of code executed on turn 2 (compile failed; nothing
ran).

## Root cause in coda's source

Two cooperating pieces:

1. `src/coda/agent.py`, function `_extract_code_blocks` (lines 263-265)
   and `CODE_BLOCK_RE` (lines 34-37). The regex splits on
   ```` ```python``` ```` fences only. Because the model wrapped its
   entire 833-line response in a single fence (using `# --- next block
   ---` comments as internal separators), the extractor returned a list
   of length 1. The per-block-with-state-preservation loop in
   `Agent.run` (lines 207-227) therefore had only one item to iterate
   over.

2. `src/coda/sandbox.py`, `Sandbox.execute` (lines 369-413). The very
   first thing `execute` does is `compile(code, "<coda>", "exec")` on
   the entire string. On `SyntaxError` it returns an empty
   `ExecutionResult` (lines 402-413) — no lines run, no partial
   progress, no recovery attempt.

The combined effect: any single stray em-dash, smart-quote, or stray
prose line that the model glues into a multi-segment fenced block voids
every segment before AND after it. The comment at `agent.py:198-204`
already acknowledges this exact failure-mode in `framework_compare` —
the block-by-block loop was meant to prevent it — but the fix only
helps when the model uses multiple ```` ```python``` ```` fences. The
"one fence, many `# --- next block ---` separators" case still
collapses to a single unit.

## Proposed fix

Treat the `# --- next block ---` separator (which the system prompt and
the model both already think of as a block boundary) as equivalent to
a fence boundary at extraction time. Each segment is then compiled and
exec'd independently by the existing loop, so a SyntaxError in segment
N preserves the state built up by segments 1..N-1 (and the per-block
feedback already tells the model which segment failed).

### `src/coda/agent.py`

Add a sub-splitter and use it in `_extract_code_blocks`. Diff:

```diff
@@ -34,6 +34,13 @@
 CODE_BLOCK_RE = re.compile(
     r"```(?:python|py)?\s*\n(.*?)```",
     re.DOTALL | re.IGNORECASE,
 )

+# A line consisting solely of a "# --- next block ---" comment (any
+# number of dashes, optional surrounding whitespace) is treated as a
+# soft fence boundary. The model is taught to emit multiple ```python
+# fences per turn, but in practice it sometimes wraps the whole reply
+# in one fence and uses these comment separators instead — without
+# splitting, a single SyntaxError voids every segment.
+BLOCK_SEPARATOR_RE = re.compile(
+    r"^[ \t]*#[ \t]*-{2,}[ \t]*next[ \t]+block[ \t]*-{2,}[ \t]*$",
+    re.IGNORECASE | re.MULTILINE,
+)
+
@@ -263,7 +270,18 @@
-def _extract_code_blocks(text: str) -> list[str]:
-    """Return every Python code block in `text`, in order. Empty list if none."""
-    return [m.group(1).rstrip() for m in CODE_BLOCK_RE.finditer(text)]
+def _extract_code_blocks(text: str) -> list[str]:
+    """Return every Python code block in `text`, in order. Empty list if none.
+
+    A fenced block that contains `# --- next block ---` separator
+    comments is further split on those markers. Each segment is run
+    independently by `Agent.run`, so a SyntaxError in one segment can
+    no longer wipe out the productive segments before it.
+    """
+    out: list[str] = []
+    for m in CODE_BLOCK_RE.finditer(text):
+        body = m.group(1)
+        parts = BLOCK_SEPARATOR_RE.split(body)
+        for p in parts:
+            s = p.strip("\n").rstrip()
+            if s.strip():  # skip empty segments
+                out.append(s)
+    return out
```

Before / after on the offending payload:

- **Before**: `_extract_code_blocks(text)` → `[<109,548-char string>]`.
  The runtime sees 1 block; compile fails; nothing runs; `turn_complete`
  records `blocks=1, had_error=true`.
- **After**: `_extract_code_blocks(text)` → list of 59 segments
  (58 separators + 1 trailing). The runtime iterates: segments 1-3
  (`ls`, fetch helpers, GitHub API calls) compile and run, populating
  `globals` with `fetch_text`, `meta`, etc. Segments containing pure
  markdown / `<user>` prose fail compile, the loop stops at the first
  failing one (existing `break` at agent.py:227), and the feedback
  already produced by `_format_block_results` reports
  `blocks_run='N' blocks_emitted='59' succeeded='N-1'`, telling the
  model exactly which segment broke. The early productive sub-agent
  calls survive into turn 3.

(Optional, additive defence-in-depth: in `Sandbox.execute`, when
`compile` raises `SyntaxError` AND the code contains the same
`BLOCK_SEPARATOR_RE`, re-attempt by splitting and exec'ing parts that
compile, collecting per-segment stdout. I'd hold off on this until we
see whether the extractor-level split is sufficient — it covers the
observed failure cleanly without changing sandbox semantics.)

## How you'd test the fix

Add a unit test against `_extract_code_blocks` in `tests/` that feeds
in a synthetic assistant reply containing one ```` ```python``` ````
fence whose body is `"print('a')\n# --- next block ---\nraise
SyntaxError\n# --- next block ---\nprint('c')"`, and assert it returns
three segments. Then an integration test that runs `Agent.run` with a
stub LLM that replays exactly the turn-2 payload from
`runs/framework_compare_20260517_020702.jsonl`: assert that at least
the first segment executes (e.g. a global it sets is present in
`sandbox.globals`) and that the feedback message contains
`succeeded='N'` with `N >= 1` rather than the current N=0. For a final
end-to-end check, replay `framework_compare_20260517_020702.jsonl`
through `src/coda/replay.py` against the patched runtime and confirm
the run no longer collapses to `end_turn` after one bad fence.
