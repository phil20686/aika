import inspect


def required_args(signature: inspect.Signature):
    """
    Extracts from a function signature those arguments which have not yet been bound.
    See eg: https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind

    """
    return {
        name
        for name, param in signature.parameters.items()
        if param.default == param.empty
        and param.kind not in [param.VAR_KEYWORD, param.VAR_POSITIONAL]
    }
