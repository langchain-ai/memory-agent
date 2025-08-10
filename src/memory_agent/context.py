"""Define the runtime context information for the agent."""

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated

from memory_agent import prompts


@dataclass(kw_only=True, init=False)
class Context:
    """Main context class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )

    system_prompt: str = prompts.SYSTEM_PROMPT

    def __init__(self, **kwargs):
        """Initialize configuration, preferring environment variables over passed values."""
        for f in fields(self):
            if not f.init:
                continue

            # Check env var first, fall back to passed value, then field default
            value = os.environ.get(f.name.upper(), kwargs.get(f.name, f.default))
            setattr(self, f.name, value)
