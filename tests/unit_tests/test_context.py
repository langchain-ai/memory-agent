import os

from memory_agent.context import Context


def test_context_init() -> None:
    context = Context(user_id="test-user")
    assert context.user_id == "test-user"


def test_context_init_with_env_vars() -> None:
    os.environ["USER_ID"] = "test-user"
    context = Context()
    assert context.user_id == "test-user"


def test_context_init_with_env_vars_and_passed_values() -> None:
    os.environ["USER_ID"] = "test-user"
    context = Context(user_id="actual-user")
    assert context.user_id == "actual-user"
