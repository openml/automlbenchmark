# AutoML Benchmark
The OpenML AutoML Benchmark provides a framework for evaluating and comparing open-source AutoML systems.  
The system is *extensible* because you can [add your own](https://openml.github.io/automlbenchmark/docs/extending/) 
AutoML frameworks and datasets. For a thorough explanation of the benchmark, and evaluation of results, 
you can read our [paper](https://arxiv.org/abs/2207.12560).

Automatic Machine Learning (AutoML) systems automatically build machine learning pipelines
or neural architectures in a data-driven, objective, and automatic way. They automate a lot 
of drudge work in designing machine learning systems, so that better systems can be developed, 
faster. However, AutoML research is also slowed down by two factors:

* We currently lack standardized, easily-accessible benchmarking suites of tasks (datasets) that are curated to reflect important problem domains, practical to use, and sufficiently challenging to support a rigorous analysis of performance results. 

* Subtle differences in the problem definition, such as the design of the hyperparameter search space or the way time budgets are defined, can drastically alter a task’s difficulty. This issue makes it difficult to reproduce published research and compare results from different papers.

This toolkit aims to address these problems by setting up standardized environments for in-depth experimentation with a wide range of AutoML systems.

Website: <https://openml.github.io/automlbenchmark/index.html>

Documentation: <https://openml.github.io/automlbenchmark/docs/index.html>

Installation: <https://openml.github.io/automlbenchmark/docs/getting_started/>

### Features:

* Curated suites of benchmarking datasets from [OpenML](https://www.openml.org) ([regression](https://www.openml.org/s/269), [classification](https://www.openml.org/s/271)).
* Includes code to benchmark a number of [popular AutoML systems](https://openml.github.io/automlbenchmark/frameworks.html) on regression and classification tasks.
* [New AutoML systems can be added](https://openml.github.io/automlbenchmark/docs/extending/framework/)
* Experiments can be run in Docker or Singularity containers
* Execute experiments locally or on AWS
