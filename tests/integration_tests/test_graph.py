from typing import List

import langsmith as ls
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from memory_agent.context import Context
from memory_agent.graph import builder

pytestmark = pytest.mark.anyio


@ls.unit
@pytest.mark.parametrize(
    "conversation",
    [
        ["My name is Alice and I love pizza. Remember this."],
        [
            "Hi, I'm Bob and I enjoy playing tennis. Remember this.",
            "Yes, I also have a pet dog named Max.",
            "Max is a golden retriever and he's 5 years old. Please remember this too.",
        ],
        [
            "Hello, I'm Charlie. I work as a software engineer and I'm passionate about AI. Remember this.",
            "I specialize in machine learning algorithms and I'm currently working on a project involving natural language processing.",
            "My main goal is to improve sentiment analysis accuracy in multi-lingual texts. It's challenging but exciting.",
            "We've made some progress using transformer models, but we're still working on handling context and idioms across languages.",
            "Chinese and English have been the most challenging pair so far due to their vast differences in structure and cultural contexts.",
        ],
    ],
    ids=["short", "medium", "long"],
)
async def test_memory_storage(conversation: List[str]):
    mem_store = InMemoryStore()

    graph = builder.compile(store=mem_store, checkpointer=InMemorySaver())
    user_id = "test-user"

    for content in conversation:
        await graph.ainvoke(
            {"messages": [("user", content)]},
            {"thread_id": "thread"},
            context=Context(user_id=user_id),
        )

    namespace = ("memories", user_id)
    memories = mem_store.search(namespace)

    ls.expect(len(memories)).to_be_greater_than(0)

    bad_namespace = ("memories", "wrong-user")
    bad_memories = mem_store.search(bad_namespace)
    ls.expect(len(bad_memories)).to_equal(0)
