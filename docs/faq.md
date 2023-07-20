# Frequently Asked Questions

If your question is not answered here, please check our Github [issue tracker](https://github.com/openml/automlbenchmark/issues) and [discussion board](https://github.com/openml/automlbenchmark/discussions). 
If you still can not find an answer, please [open a Q&A discussion on Github](https://github.com/openml/automlbenchmark/discussions/new?category=q-a).

## (When) will you add framework X?

We are currently not focused on integrating additional AutoML systems.
However, we process any pull requests that add frameworks and will assist with the integration.
The best way to make sure framework X gets included is to start with the integration 
yourself or encourage the package authors to do so. For technical details see 
[Adding an AutoML Framework](./extending/framework.md).

It is also possible to open a Github issue indicating the framework you would like added.
Please use a clear title (e.g. "Add framework: X") and provide some relevant information 
(e.g. a link to the documentation).
This helps us keep track of which frameworks people are interested in seeing included.


## Framework setup is not executed
First, it is important to note that we officially only officially support Ubuntu 22.04 LTS,
though other versions and MacOS generally work too. If that does not work, for 
example with Windows, use docker mode as per [the installation instructions](getting_started.md#installation).
For MacOS, it may be required to have [brew](https://brew.sh) installed.

If you are experiencing issues with the framework setup not being executed, please
try the following steps before opening an issue:

  - delete the `.marker_setup_safe_to_delete` from the framework module and try to run 
    the benchmark again. This marker file is automatically created after a successful 
    setup to avoid having to execute it each tine (setup phase can be time-consuming), 
    this marker then prevents auto-setup, except if the `-s only` or `-s force` args below are used.

  - force the setup using the  `--setup=only` arg on the command line. This forces the
    setup to take place. If the setup is now done correctly, you can run the commands
    as normal to start the benchmark. If not, continue.

  - manually clean the installation files by deleting the `lib`, `venv` and `.setup` folders
    in the given framework folder (e.g. `frameworks/MyFramework`), and try again.

