import inspect


def required_args(signature: inspect.Signature):
    return {
        name
        for name, param in signature.parameters.items()
        if param.default == param.empty
        and param.kind not in [param.VAR_KEYWORD, param.VAR_POSITIONAL]
    }
