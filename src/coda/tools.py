"""@tool decorator — register an inline Python function as a sandbox tool.

Two ways to expose tools to a coda Agent:

1. Filesystem registry (the scalable pattern; see README "Option B"):
   put `.py` files under `interfaces/` and let the model `ls`/`read`/`import`
   them. No registration needed.

2. Inline `@tool` (this module): for quick scripts, decorate a Python
   function and pass it via `Agent(tools=[fn])`. coda injects the function
   into the sandbox globals under its `__name__`, so user-emitted code calls
   it directly: `result = my_tool(arg)`.

The decorator doesn't currently do anything magic — it stamps the function
with a `Tool` marker so the Agent can distinguish "things to inject" from
"random callables you passed in by mistake". The marker also carries the
docstring so the Agent can include it in the prompt's tool listing.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    fn: Callable[..., Any]
    name: str
    description: str
    signature: str

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def tool(fn: Callable[..., Any]) -> Tool:
    """Mark `fn` as a coda tool.

    The function keeps its original behavior; the decorator just attaches
    metadata coda's prompt assembler will use to describe it to the model.
    """
    if isinstance(fn, Tool):
        return fn
    return Tool(
        fn=fn,
        name=fn.__name__,
        description=(inspect.getdoc(fn) or "").strip(),
        signature=str(inspect.signature(fn)),
    )
