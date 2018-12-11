"""
the **frameworks** package contains all the automl framework subpackages:
each of those framework package can expose the following functions:

- ``run(*args, **kvargs)``: this function is mandatory for each package and is called by each job,
    providing a ``Dataset`` and a ``TaskConfig`` instance to the framework.

    The framework should run its automl implementation against the provided dataset,
    and should try to honour the constraints provided by the task_config.

    This function is usually implemented by importing the ``exec`` module dynamically
    and forwarding the parameters to its own ``run`` function::
        def run(*args, **kwargs):
            from .exec import run
            run(*args, **kwargs)

    this provides the possibility, if necessary – for example if the framework depends on libraries incompatible with the app –,
    to delegates the execution to a different process after serializing the parameters (using json or pickle for example).

- ``setup(*args)``: this function is optional and is called to let the framework install its required dependencies.
- ``docker_commands()``: this function is optional and should return a string with docker commands
    necessary to install the framework dependencies when building the docker image.


important
    Ideally, frameworks entry packages should not import any automl module outside **utils** for the reason explained above.

"""