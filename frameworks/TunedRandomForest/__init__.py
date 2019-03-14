
def run(*args, **kwargs):
    from .exec import run
    return run(*args, **kwargs)


__all__ = (run)