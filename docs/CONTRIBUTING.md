# Contributing to the AutoML Benchmark
We appreciate you are considering contributing to the AutoML Benchmark.
Remote collaboration may be hard sometimes, so we provide guidelines in this document
to make the experience as smooth as possible.

In this document there is information on:

 - [Reporting a Bug](#reporting-a-bug)
 - [Suggesting a Feature](#features)
 - [Suggesting a Dataset](#datasets)
 - [Suggesting Ideas on Benchmark Design](#ideas)
 - [Contributing Code or Documentation Changes](#contributing-changes)

## Reporting a Bug
If you find a bug with the software, please first search our [issue tracker](https://github.com/openml/automlbenchmark/issues) to see if it has been reported before.
If it has been, please see if there is relevant information missing that may help reproduce the issue and add it if necessary.
If there is nothing to add, simply leave a 👍 on the issue. This lets us know more people are affected by it.

### Creating a Bug Report
After confirming your bug isn't reported on our issue tracker, please open a new issue to make a bug report.
A good bug report should describe the error and also provide:

 * A minimal script (and/or configuration) to reproduce the issue.
 * The _observed_ behavior, for example a stack trace with the error.
 * The _expected_ behavior. What did you expect to happen?
 * Any additional information you may have.
 * Information on your installed versions. If applicable, please provide both information about the `runbenchmark` environment and the `framework` environment (typically in `frameworks/FRAMEWORK/venv`).

Observing these guidelines greatly improves the chance that we are able to help you.
It also allows us to address the issue more quickly, which means we can help more people.

## Features
If you want to suggest a new feature for the benchmark software, please [open an issue](https://github.com/openml/automlbenchmark/issues/new).
Please motivate why we should consider adding the feature and how the user is expected to use it.

## Datasets
If you have a suggestion for a new dataset to include in the benchmark,
please [open a discussion on the datasets board](https://github.com/openml/automlbenchmark/discussions/new?category=datasets).
Please motivate why the dataset is a good inclusion for the benchmark.
Examples of good motivations may include:

 * Evidence that it produces interesting results, for example by reporting a small scale benchmark on the dataset.
 * Evidence that is represents a very relevant problem, e.g., because it is frequently used in scientific literature.

Additionally, please provide a link to the data, preferably on [OpenML](openml.org), and indicate its license (if known).
Please note that the benchmark currently supports limited data types.
Suggestions for datasets with data types which are currently not yet be supported are still welcome,
as they may help us more effectively great a good benchmark later when support is added.

## Ideas
If you have other suggestions about benchmark design, [please open a suggestion on the general board](https://github.com/openml/automlbenchmark/discussions/new?category=general).
Please motivate why we should consider changing (or adding to) the benchmark design.


## Contributing Changes
We welcome all contributions by the community. To contribute changes to the 
code or documentation, we follow a default git workflow which is also outlined below.

!!! note "For text changes"

    If you only want to contribute minor text changes, it is possible to do so 
    directly on Github. Click the pencil icon on the relevant file(s) to edit the documents,
    and Github should allow you to automatically commit to your own fork.
    After that, set up a pull request as specified below under 'Opening a Pull Request'.

### Volunteering an Issue
In order to avoid a scenario where multiple people do the same work, the first thing
to do is to make sure we (and other contributors) know you are working on a particular issue or feature.
Please ensure that a related issue is open on the issue board or open one if necessary, and ask to be assigned to that issue.
This communicates with all collaborators that they should not work on that issue, and thus we can avoid double work.
It also gives us a chance to indicate whether we are (still) interested in the proposed changes.
If it is unclear how to add the feature, or if you are unsure of which fix to apply to remove a bug, please discuss this in the issue.

### Setting up the Development Environment
Fork the repository by clicking on the `fork` button on the top right of our [Github](https://github.com/openml/automlbenchmark) page.
This should create a repository named `automlbenchmark` under your Github account.
Clone this repository (replace `GITHUB_USERNAME`):

```text
git clone https://github.com/GITHUB_USERNAME/automlbenchmark.git
```

!!! warning "Use Python 3.9"

    AutoML benchmark currently only officially supports Python 3.9.
    We advise you use that version when developing locally. 

then set up your local virtual environment:

```text
cd automlbenchmark
python -m venv venv
source venv\bin\activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

this should set up the minimum requirements for running the benchmark and running our developer tools.
The following commands should now both successfully:

```text
python runbenchmark.py constantpredictor -f 0
python -m pytest 
python -m mkdocs serve
```

When `python -m mkdocs serve` is running, you should be able to navigate to the 
local documentation server (by default at `127.0.0.1:8000`) and see the documentation.

### Make Code Changes 
Please make sure that:

 * All added code has annotated type hints and functions have docstrings.
 * Changed or added code is covered by unit tests.
 * The pull request does not add/change more than it has to in order to fix the bug/add the feature and meet the above criteria.
 * The tests and `runbenchmark.py` script still work the same as above.

In case the PR is a bug fix, please try to convert the minimal reproducing example of 
the original issue to a unit test and include it in the test suite to help avoid future regressions.
Finally, commit the changes with a meaningful commit message about what was changed and why. 

### Make Documentation Changes
The software documentation pages are written on `mkdocs` using [`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/getting-started/),
when editing these pages you can see live updates when running the `python -m mkdocs serve` command.
The main landing page with information about the project is written in pure `html` and `css`.

### Open a Pull Request
When opening a pull request, reference the issue that it closes.
Please also provide any additional context that helps review the pull request that may not have been appropriate as code comments.

