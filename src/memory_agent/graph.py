"""Graphs that extract memories on a schedule."""

import asyncio
import logging
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore

from memory_agent import configuration, tools, utils
from memory_agent.state import State

logger = logging.getLogger(__name__)

llm = init_chat_model()


async def call_model(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    configurable = configuration.Configuration.from_runnable_config(config)
    memories = await store.asearch(
        ("memories", config["configurable"]["user_id"]), limit=10
    )
    formatted = "\n".join(f"[{mem.key}]: {mem.value}" for mem in memories)
    if formatted:
        formatted = f"""
<memories>
{formatted}
</memories>"""

    sys = configurable.system_prompt.format(
        user_info=formatted, time=datetime.now().isoformat()
    )
    msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
        [{"role": "system", "content": sys}, *state.messages],
        {"configurable": utils.split_model_and_provider(configurable.model)},
    )
    return {"messages": [msg]}


async def store_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    tool_calls = state.messages[-1].tool_calls
    saved_memories = await asyncio.gather(
        *(
            tools.upsert_memory(**tc["args"], config=config, store=store)
            for tc in tool_calls
        )
    )

    results = [
        {
            "role": "tool",
            "content": mem,
            "tool_call_id": tc["id"],
        }
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}


def route_message(state: State):
    msg = state.messages[-1]
    if msg.tool_calls:
        return "store_memory"
    return END


# Create the graph + all nodes
builder = StateGraph(State, config_schema=configuration.Configuration)

builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])

graph = builder.compile()
graph.name = "MemoryAgent"


__all__ = ["graph"]
