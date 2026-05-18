"""User simulator for tau-bench tasks, using ClaudeAgentSDKClient.

Sierra's tau-bench ships a litellm-based simulator (`LLMUserSimulationEnv`)
that needs ANTHROPIC_API_KEY / OPENAI_API_KEY set explicitly. Coda is
already wired through Claude Code OAuth via ClaudeAgentSDKClient, which
has different auth. This module re-implements the same simulator
contract (`reset(instruction)` and `step(content)`) on top of coda's
own LLM client, so we can drive tau-bench tasks without a second auth
path.

The conversation script and the "###STOP###" termination convention are
copied verbatim from tau-bench's `LLMUserSimulationEnv.build_system_prompt`
so any future swap-in of the original simulator is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coda.llm import ClaudeAgentSDKClient, CompletionResponse, Message


# Verbatim from tau_bench/envs/user.py:LLMUserSimulationEnv.build_system_prompt
_USER_SIM_RULES = """\
You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information \
that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. \
For example, if the agent asks for the order id but it is not mentioned in \
the instruction, do not make up an order id, just say you do not remember \
or have it.
- If the instruction goal is satisified, generate '###STOP###' as a \
standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use \
your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the \
personalities in the instruction."""


@dataclass
class CodaUserSimulator:
    """A user simulator that plays the customer for a tau-bench task.

    Construct with the task instruction; call `step(agent_message)` for each
    agent turn. Returns the user's reply as a string. When the simulator
    decides the goal is satisfied, the returned string contains
    `###STOP###` — the agent (or runner) should treat that as end-of-task.

    Args:
        instruction: the task.instruction string from tau-bench. Becomes
            the customer's "goal" that the simulator plays out.
        model: which Claude model to use as the customer brain. Defaults
            to claude-sonnet-4-6 — cheaper and fast enough for this role;
            the customer doesn't need Opus-level reasoning.
    """

    instruction: str
    model: str = "claude-sonnet-4-6"
    _client: ClaudeAgentSDKClient = field(default_factory=ClaudeAgentSDKClient, init=False)
    _system_prompt: str = field(init=False)
    _history: list[Message] = field(default_factory=list, init=False)
    _calls: int = field(default=0, init=False)
    stopped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._system_prompt = _USER_SIM_RULES.format(
            instruction_display=f"\n\nInstruction: {self.instruction}\n"
        )
        # Seed the conversation the same way tau-bench does: the simulator
        # is the assistant, and our first turn is the agent saying "Hi!"
        # (which we generate on reset() so the simulator has something to
        # respond to).

    def reset(self) -> str:
        """Open the conversation — returns the customer's opening message.

        Called once at the start of the task. The simulator responds as if
        the agent just said "Hi! How can I help you today?" — this gets
        the customer to state their request in their own words.
        """
        self._history = [
            Message(role="user", content="Hi! How can I help you today?"),
        ]
        return self._complete()

    def step(self, agent_message: str) -> str:
        """Send the agent's message to the customer; return the customer's reply.

        If the customer decides the task is done, the reply contains
        '###STOP###' and `self.stopped` is set to True. Caller should
        check `stopped` after each step.
        """
        if self.stopped:
            # Idempotent post-stop calls return STOP again rather than
            # hitting the LLM. The agent's prompt teaches it to stop
            # emitting code once it sees STOP, but a stray call shouldn't
            # cost tokens.
            return "###STOP###"
        # The simulator-as-assistant's prior reply was the last message
        # in history. The agent's response to it goes in as "user".
        self._history.append(Message(role="user", content=agent_message))
        reply = self._complete()
        if "###STOP###" in reply:
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
        # Append the simulator's reply to history (it's the "assistant" in
        # this conversation — note role from coda's Message protocol is
        # constrained to user/assistant, which is correct here).
        self._history.append(Message(role="assistant", content=text))
        return text

    @property
    def call_count(self) -> int:
        """How many simulator LLM calls have fired this run. Useful for
        cost accounting — every conversation turn doubles the API spend."""
        return self._calls
