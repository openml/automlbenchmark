---
---
<!--- Comments --->

# AutoML Systems

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

##### Statement To Authors
We did our best to provide a reasonable description which highlights some unique or important aspects of each package.
If you want to change or add to the description and references of your AutoML package, please submit a pull request with your proposed changes. 

The description needs to be kept brief and factual.
The goal is to get an impression, based on which the reader can delve more in-depth in the provided documentation.

## auto-sklearn
[source](https://github.com/automl/auto-sklearn) |
[documentation](http://automl.github.io/auto-sklearn/stable/) |
Python |
Optimization: Bayesian Optimization |
3-clause BSD 

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
Meta-learning is used to warm-start the search procedure, this means that the search is more likely to start with good pipelines.

#### Papers

Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Springenberg, Manuel Blum and Frank Hutter (2015).
[Efficient and Robust Automated Machine Learning](http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf)
*Advances in Neural Information Processing Systems 28 (NIPS 2015)*.

## Auto-WEKA
[source](https://github.com/automl/autoweka) | 
[documentation](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/manual.pdf) |
Java, CLI, GUI |
Optimization: Bayesian Optimization |
GPLv3

> Our hope is that Auto-WEKA will help non-expert users to more effectively identify machine learning algorithms and
> hyperparameter settings appropriate to their applications, and hence to achieve improved performance.

Auto-WEKA is built on the Java machine learning package [WEKA](http://www.cs.waikato.ac.nz/ml/weka/).
Auto-WEKA can be used through a graphical user interface, which means there is no need to use a terminal or programming language.
It is one of the first systems to consider joint algorithm selection and hyperparameter optimization in addition to preprocessing steps.



#### Papers

Lars Kotthoff, Chris Thornton, Holger Hoos, Frank Hutter, and Kevin Leyton-Brown (2017).
[Auto-WEKA 2.0: Automatic model selection and hyperparameter optimization in WEKA](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/papers/16-599.pdf)
*JMLR. 18(25):1−5, 2017*

Chris Thornton, Frank Hutter, Holger Hoos, and Kevin Leyton-Brown (2013).
[Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/papers/autoweka.pdf)
*Proceedings of KDD 2013*.


## H2O AutoML
[source](https://github.com/h2oai/h2o-3) |
[documentation](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) |
Python, R |
Optimization: Random Search |
Apache-2.0

> H2O’s AutoML can be used for automating the machine learning workflow,
> which includes automatic training and tuning of many models within a user-specified time-limit.


H2O AutoML performs Random Search followed by a stacking stage.
By default it uses the H2O machine learning package, which supports distributed training.

*Maybe mention something similar to the following?*
H2O AutoML is developed by the company H2O.
This means that the package features frequent updates and is less likely to be abandoned.
On the other hand, as a free user your concerns might be overshadowed by users of their alternative, pay-to-use, driverless AI.

#### Papers
The booklets?

## hyperopt-sklearn 
[source](https://github.com/hyperopt/hyperopt-sklearn) |
[documentation](http://hyperopt.github.io/hyperopt-sklearn/) |
Python |
Optimization: Random Search, various SMBO |
3-clause BSD

> Hyperopt-sklearn is Hyperopt-based model selection among machine learning algorithms in scikit-learn.

Hyperopt-sklearn allows for different search strategies through a scikit-learn-like interface.
Besides random search, various sequential model based optimization (SMBO) techniques are available.
Amongst these are Tree of Parzen Estimators (TPE), Annealing and Gaussian Process Trees.

#### Papers

Komer, Brent, James Bergstra, and Chris Eliasmith (2014).
[Hyperopt-sklearn: automatic hyperparameter configuration for scikit-learn.](http://compneuro.uwaterloo.ca/files/publications/komer.2014b.pdf)
*ICML workshop on AutoML 2014*.

## OBOE 
[source](https://github.com/udellgroup/oboe) |
[documentation](https://github.com/udellgroup/oboe) |
Python |
Optimization: Collaborative Filtering |
License N/A

> Oboe is a data-driven Python algorithmic system for automated machine learning, and is based on matrix factorization and classical experiment design. 

OBOE is still in early stages of development.
It focuses on finding a good initial set of pipelines from which to start further optimization.
The focus is on time-constrained model selection and hyperparameter tuning, using meta-learning to find good pipelines.

OBOE searches for a good set of algorithm configurations to create an ensemble from, using meta-learning.
With collaborative filtering they estimate which algorithms are likely to do well on the new dataset.

#### Papers

Chengrun Yang, Yuji Akimoto, Dae Won Kim, Madeleine Udell (2018).
[OBOE: Collaborative Filtering for AutoML Initialization](https://arxiv.org/pdf/1808.03233.pdf).
*arXiv preprint*.

## TPOT 
[source](https://github.com/EpistasisLab/tpot) |
[documentation](https://epistasislab.github.io/tpot/) |
Python, CLI |
Optimization: Genetic Programming |
LGPL-3.0

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

