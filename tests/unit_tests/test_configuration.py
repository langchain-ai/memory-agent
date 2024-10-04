from memory_agent.configuration import Configuration


def test_configuration_from_none() -> None:
    Configuration.from_runnable_config()
