def split_model_and_provider(fully_specified_name: str) -> dict:
    """Split a fully specified model name into provider and model components.
    
    Args:
        fully_specified_name: String in format "provider/model" (e.g., "anthropic/claude-3").
                            If no provider is specified, returns provider=None.
    
    Returns:
        Dictionary with keys "model" (str) and "provider" (str | None).
    """
    if not isinstance(fully_specified_name, str):
        raise ValueError("Input must be a string")
    
    if not fully_specified_name:
        raise ValueError("Input string cannot be empty")
    
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
        if not provider or not model:
            raise ValueError("Provider and model must both be specified when using '/'")
    else:
        provider = None
        model = fully_specified_name
    
    return {"model": model, "provider": provider}
