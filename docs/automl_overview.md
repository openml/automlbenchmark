---
---

# AutoML systems

List of AutoML systems in the benchmark, in alphabetical order:

- [auto-sklearn](#auto-sklearn)
- [Auto-WEKA](#auto-weka)
- [H2O AutoML](#h2o-automl)
- [hyperopt-sklearn](#hyperopt-sklearn)
- [oboe](#oboe)
- [TPOT](#tpot)

There is more to an AutoML system than just its performance.
An AutoML system may only be available through an API for a specific programming language, while others can work stand-alone.
Some systems might output models which can be used without further dependency on the AutoML package,
in other cases the AutoML system is still required to use the model.
Some systems might be developed with a specific domain in mind. 
When choosing an AutoML system, it is essential to consider things that are important to you.


On this page a brief description and further references for the AutoML systems in the benchmark is provided.

## auto-sklearn

[source](https://github.com/automl/auto-sklearn) | [documentation](http://automl.github.io/auto-sklearn/stable/) | Python | Optimization: Bayesian Optimization

> auto-sklearn is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.

Auto-sklearn is declared the overall winner of the [ChaLearn AutoML](http://automl.chalearn.org/) Challenge
[1](https://docs.google.com/a/chalearn.org/viewer?a=v&pid=sites&srcid=Y2hhbGVhcm4ub3JnfGF1dG9tbHxneDoyYThjZjhhNzRjMzI3MTg4)
in 2015-2016 and
[2](https://www.4paradigm.com/competition/pakdd2018)
in 2017-2018.
It provides a scikit-learn-like interface in Python and uses Bayesian optimization to find good machine learning pipelines.


It features automatic ensemble construction.
Unfortunately, multiprocessing is not supported out of the box, but there is a 
[work-around](http://automl.github.io/auto-sklearn/stable/examples/example_parallel.html).

#### Papers

Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Springenberg, Manuel Blum and Frank Hutter (2015).
[Efficient and Robust Automated Machine Learning](http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf)
*Advances in Neural Information Processing Systems 28 (NIPS 2015)*.

## Auto-WEKA

Ideally we let the authors write a one/two paragraph description of their own package?
It must be factual, can state e.g. the most important aspects of the package and its goal.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi tempus, risus et condimentum blandit, erat ex eleifend neque, et fermentum est est vitae dolor. Morbi rutrum quam sit amet mi iaculis, in eleifend nulla blandit. Nam et lacus et enim viverra pulvinar a a libero. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. In convallis eros in scelerisque blandit. Nam iaculis purus id dui tincidunt vehicula. Maecenas porttitor ex quis volutpat pretium. Sed nec ante lacus. Cras sollicitudin pulvinar lorem, et dignissim mi consectetur vel. Vestibulum sed feugiat felis.

## H2O AutoML
 
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi tempus, risus et condimentum blandit, erat ex eleifend neque, et fermentum est est vitae dolor. Morbi rutrum quam sit amet mi iaculis, in eleifend nulla blandit. Nam et lacus et enim viverra pulvinar a a libero. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. In convallis eros in scelerisque blandit. Nam iaculis purus id dui tincidunt vehicula. Maecenas porttitor ex quis volutpat pretium. Sed nec ante lacus. Cras sollicitudin pulvinar lorem, et dignissim mi consectetur vel. Vestibulum sed feugiat felis.

## hyperopt-sklearn
 
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi tempus, risus et condimentum blandit, erat ex eleifend neque, et fermentum est est vitae dolor. Morbi rutrum quam sit amet mi iaculis, in eleifend nulla blandit. Nam et lacus et enim viverra pulvinar a a libero. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. In convallis eros in scelerisque blandit. Nam iaculis purus id dui tincidunt vehicula. Maecenas porttitor ex quis volutpat pretium. Sed nec ante lacus. Cras sollicitudin pulvinar lorem, et dignissim mi consectetur vel. Vestibulum sed feugiat felis.

## oboe
 
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi tempus, risus et condimentum blandit, erat ex eleifend neque, et fermentum est est vitae dolor. Morbi rutrum quam sit amet mi iaculis, in eleifend nulla blandit. Nam et lacus et enim viverra pulvinar a a libero. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. In convallis eros in scelerisque blandit. Nam iaculis purus id dui tincidunt vehicula. Maecenas porttitor ex quis volutpat pretium. Sed nec ante lacus. Cras sollicitudin pulvinar lorem, et dignissim mi consectetur vel. Vestibulum sed feugiat felis.

## TPOT
 
[source](https://github.com/EpistasisLab/tpot) | [documentation](https://epistasislab.github.io/tpot/) | Python, CLI | Optimization: Genetic Programming

> Consider TPOT your Data Science Assistant.
> TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

TPOT provides a scikit-learn-like interface for use in Python, but can be called from the command line as well.
It constructs machine learning pipelines of arbitrary length using scikit-learn algorithms and, optionally, xgboost.
In its search, preprocessing and stacking are both considered.
After the search, it is able to export python code so that you may reconstruct the pipeline without dependencies on TPOT.


While technically pipelines can be of any length, TPOT performs multi-objective optimization: 
it aims to keep the number of components in the pipeline small while optimizing the main metric.
TPOT features support for sparse matrices, multiprocessing and custom pipeline components.
 
#### Papers

Randal S. Olson, Ryan J. Urbanowicz, Peter C. Andrews, Nicole A. Lavender, La Creis Kidd, and Jason H. Moore (2016).
[Automating biomedical data science through tree-based pipeline optimization](http://dx.doi.org/10.1007/978-3-319-31204-0_9).
*Applications of Evolutionary Computation*, pages 123-137.

Randal S. Olson, Nathan Bartley, Ryan J. Urbanowicz, and Jason H. Moore (2016).
[Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science](http://doi.acm.org/10.1145/2908812.2908918).
*Proceedings of GECCO 2016*, pages 485-492.  

