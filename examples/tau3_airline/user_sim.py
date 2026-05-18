"""τ³-aligned user simulator on top of coda's ClaudeAgentSDKClient.

τ³ ships a litellm-backed `UserSimulator` (see
`tau2/user/user_simulator.py`) that drives the customer side with the
guidelines in `data/tau2/user_simulator/simulation_guidelines.md` plus
the per-task `UserScenario.instructions`. Coda is authed through Claude
Code OAuth via ClaudeAgentSDKClient, which doesn't match τ³'s auth.

Rather than wrestle with two auth paths, this re-implements the same
contract (`reset()` + `step(message) -> str`) on top of coda's own LLM
client. The system prompt loads τ³'s guidelines file directly so any
upstream change to user-sim behavior flows through with no edits here.

Token conventions copied verbatim from τ³: '###STOP###', '###TRANSFER###',
'###OUT-OF-SCOPE###' all terminate the conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from coda.llm import ClaudeAgentSDKClient, CompletionResponse, Message

from tau3_airline.airline_primitive import TAU2_BENCH_PATH


_GUIDELINES_PATH = (
    Path(TAU2_BENCH_PATH)
    / "data"
    / "tau2"
    / "user_simulator"
    / "simulation_guidelines.md"
)


_STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")


def _load_guidelines() -> str:
    return _GUIDELINES_PATH.read_text(encoding="utf-8").strip()


@dataclass
class Tau3UserSimulator:
    """Customer-side simulator for a τ³ task.

    Args:
        instructions_text: the rendered task instructions (see
            `task_instruction_text` in `airline_primitive`). Becomes the
            customer's `<scenario>...</scenario>` block.
        model: which Claude model plays the customer. Sonnet 4.6 is plenty
            for this role and noticeably cheaper than Opus.
    """

    instructions_text: str
    model: str = "claude-sonnet-4-6"
    _client: ClaudeAgentSDKClient = field(default_factory=ClaudeAgentSDKClient, init=False)
    _system_prompt: str = field(init=False)
    _history: list[Message] = field(default_factory=list, init=False)
    _calls: int = field(default=0, init=False)
    stopped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        guidelines = _load_guidelines()
        # τ³ wraps PERSONA_GUIDELINES into the markdown via a placeholder;
        # we have no runtime persona, so substitute an empty string the
        # same way τ³'s `UserSimulator.system_prompt` does.
        guidelines = guidelines.replace("<PERSONA_GUIDELINES>", "")
        self._system_prompt = (
            f"{guidelines}\n\n<scenario>\n{self.instructions_text}\n</scenario>"
        )

    def reset(self) -> str:
        """Open the conversation — returns the customer's first message.

        Mirrors τ³'s pattern of seeding the customer with a synthetic
        "Hi, how can I help?" so they state their request in their own
        words.
        """
        self._history = [
            Message(role="user", content="Hi! How can I help you today?"),
        ]
        return self._complete()

    def step(self, agent_message: str) -> str:
        """Send the agent's message to the customer; return the customer's reply.

        If the reply contains any of the τ³ termination tokens
        ('###STOP###', '###TRANSFER###', '###OUT-OF-SCOPE###'), the
        simulator latches `stopped=True` and subsequent calls return
        '###STOP###' without hitting the LLM.
        """
        if self.stopped:
            return "###STOP###"
        self._history.append(Message(role="user", content=agent_message))
        reply = self._complete()
        if any(tok in reply for tok in _STOP_TOKENS):
            self.stopped = True
        return reply

    def _complete(self) -> str:
        resp: CompletionResponse = self._client.complete(
            system=self._system_prompt,
            messages=list(self._history),
            model=self.model,
            max_tokens=512,
        )
        self._calls += 1
        text = resp.text.strip()
        self._history.append(Message(role="assistant", content=text))
        return text

    @property
    def call_count(self) -> int:
        return self._calls
