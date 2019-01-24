
def setup(*args):
  print("setting up decision tree")

def run(*args, **kwargs):
    from .exec import run
    run(*args, **kwargs)


__all__ = (setup, run)
