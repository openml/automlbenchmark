---
layout: category
title: AutoML Systems
sidebar_sort_order: 2
---

There is more to an AutoML system than just its performance.
An AutoML framework may only be available through an API for a specific programming language, while others can work stand-alone.
Some systems might output models which can be used without further dependency on the AutoML package,
in other cases the AutoML system is still required to use the model.
Some systems might be developed with a specific domain in mind. 
When choosing an AutoML system, it is essential to consider things that are important to you.

On this page a brief description and further references for the AutoML systems in the benchmark is provided.

List of AutoML systems in the benchmark, in alphabetical order:

- [auto-sklearn](#auto-sklearn)
- [Auto-WEKA](#auto-weka)
- [H2O AutoML](#h2o-automl)
- [TPOT](#tpot)

There are many more AutoML frameworks, and unfortunately we could not yet evaluate them all.
While we hope to cover them in the comparison in the future, for now we will
Some other frameworks worth mentioning are, again in alphabetical order:

- [autoxgboost](#autoxgboost)
- [FLAML](#flaml)
- [GAMA](#gama)
- [hyperopt-sklearn](#hyperopt-sklearn)
- [ML-Plan](#ml-plan)
- [mlr3automl](#mlr3automl)  
- [oboe](#oboe)

For completeness, the baseline methods are also described:

- [Constant Predictor](#constant-predictor)
- [Random Forest](#random-forest)
- [Tuned Random Forest](#tuned-random-forest)

##### Statement To Authors
We did our best to provide a reasonable description which highlights some unique or important aspects of each package.
If you want to change or add to the description and references of your AutoML package, please submit a pull request with your proposed changes. 

The description needs to be kept brief and factual.
The goal is to get an impression, based on which the reader can delve more in-depth in the provided documentation.

If your AutoML framework is not on this page and feel it should be, please open a PR with the proposed addition.
Keep the formatting consistent with the rest of the page.

-----

# Included AutoML Frameworks

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

#### Papers

\-


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



# Other AutoML Frameworks

## autoxgboost
[source](https://github.com/ja-thomas/autoxgboost) |
[documentation](https://github.com/ja-thomas/autoxgboost/blob/master/poster_2018.pdf) |
R |
Optimization: Bayesian Optimization | -

> autoxgboost aims to find an optimal xgboost model automatically using the machine learning framework mlr and the bayesian optimization framework mlrMBO.

Autoxgboost is different from most frameworks on this page in that it does not search over multiple learning algorithms.
Instead, it restricts itself to finding a good hyperparameter configuration for xgboost.
The exception to this is a preprocessing step for categorical variables, where the specific encoding strategy to use is tuned as well.

#### Papers

Janek Thomas, Stefan Coors and Bernd Bischl (2018). 
[Automatic Gradient Boosting](https://arxiv.org/pdf/1807.03873v2.pdf)
*International Workshop on Automatic Machine Learning at ICML 2018*

## FLAML
[source](https://github.com/microsoft/FLAML) |
[documentation](https://microsoft.github.io/FLAML/) |
Python |
Optimization: Configurable |
License MIT

> FLAML is a lightweight Python library that finds accurate machine learning models efficiently and economically. 

FLAML is powered by a new, cost-effective hyperparameter optimization and learner selection method invented by Microsoft Research. FLAML leverages the structure of the search space to choose a search order optimized for both cost and error. 
FLAML is fast and economical. The simple and lightweight design makes it easy to extend, such as adding customized learners or metrics.

#### Papers

Chi Wang, Qingyun Wu, Markus Weimer, and Erkang Zhu (2021).
[FLAML: A Fast and Lightweight AutoML Library](https://www.microsoft.com/en-us/research/publication/flaml-a-fast-and-lightweight-automl-library/)
*Proceedings of MLSys 2021*

Qingyun Wu, Chi Wang, and Silu Huang (2021).
[Frugal Optimization for Cost-related Hyperparameters](https://www.microsoft.com/en-us/research/publication/frugal-optimization-for-cost-related-hyperparameters/)
*Proceedings of AAAI 2021*

Chi Wang, Qingyun Wu, Silu Huang, and Amin Saied (2021).
[Economical Hyperparameter Optimization With Blended Search Strategy](https://www.microsoft.com/en-us/research/publication/economical-hyperparameter-optimization-with-blended-search-strategy/)
*The Ninth International Conference on Learning Representations (ICLR 2021)*

## GAMA 
[source](https://github.com/PGijsbers/gama) |
[documentation](https://pgijsbers.github.io/gama/) |
Python |
Optimization: Configurable |
License MIT

> GAMA is an AutoML tool for end-users and AutoML researchers with a configurable AutoML pipeline.

GAMA is a new framework under active development.
GAMA supports AutoML researchers through a configurable AutoML pipeline, extensive logging and visualization of the logs.
The configurable AutoML pipeline allows selection of the optimization and post-processing algorithms.

By default GAMA searches over linear machine learning pipelines and create an ensemble of them as a post-processing step.
Currently pipelines can be optimized with an asynchronous evolutionary algorithm or [ASHA](https://arxiv.org/abs/1810.05934).

#### Papers

Pieter Gijsbers, Joaquin Vanschoren (2019).
[GAMA: Genetic Automated Machine learning Assistant](https://joss.theoj.org/papers/10.21105/joss.01132).
*Journal of Open Source Software, 4(33), 1132*

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

## ML-Plan
[source](https://github.com/starlibs/AILibs) |
[documentation](https://starlibs.github.io/AILibs/projects/mlplan/) |
Java |
Optimization: Best-First Search on a search graph induced through Hierachical Task Network Planning | AGPL-3.0

> a new approach to AutoML based on hierarchical planning

ML-Plan organizes the search space of possible solution candidates via Hierarchical Task Network (HTN) planning.
It works with both WEKA and scikit-learn backends and can be used to deal with classification, regression, multi-label classification, and remaining useful lifetime estimation tasks.
ML-Plan is under active development.

#### Papers

Felix Mohr, Marcel Wever and Eyke Hüllermeier (2018).
[ML-Plan: Automated machine learning via hierarchical planning](https://link.springer.com/article/10.1007/s10994-018-5735-z)
*Machine Learning  107(8):1495–1515*

Marcel Wever, Felix Mohr and Eyke Hüllermeier (2018).
[ML-Plan for Unlimited-Length Machine Learning Pipelines](https://ris.uni-paderborn.de/download/3852/3853/38.pdf)
* ICML workshop on AutoML 2018*.

Marcel Wever, Felix Mohr and Eyke Hüllermeier (2018).
[Automated Multi-Label Classification based on ML-Plan](https://arxiv.org/abs/1811.04060)
*arXiv preprint*

Marcel Wever, Felix Mohr, Alexander Tornede and Eyke Hüllermeier (2019).
[Automating Multi-Label Classification Extending ML-Plan](https://ris.uni-paderborn.de/download/10232/13177/Automating_MultiLabel_Classification_Extending_ML-Plan.pdf)
* ICML workshop on AutoML 2019*.

## mlr3automl
[source](https://github.com/a-hanf/mlr3automl) |
[documentation](https://github.com/a-hanf/mlr3automl/blob/master/vignettes/mlr3automl.md) |
R |
Optimization: Hyperband | License LGPL-3.0

> mlr3automl combines a static portfolio with Hyperband tuning. 

mlr3automl is built on top of mlr3. It combines a static portfolio of known successful pipelines
with Hyperband tuning. mlr3automl currently supports classification and regression tasks.

#### Papers
\-

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


## Baselines

We compare the performance of AutoML frameworks not only to each other, but also to three baseline methods, these are:

## Constant Predictor 
[source](https://github.com/openml/automlbenchmark/tree/master/frameworks/constantpredictor)

Always predicts the class probabilities according to their occurrence in the dataset.
 
## Random Forest 
[source](https://github.com/openml/automlbenchmark/tree/master/frameworks/RandomForest)
  
The [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) of scikit-learn 0.20. 
All hyperparameters are set to their default value, except for the number of estimators, which is set to *2000*.
 
## Tuned Random Forest 
[source](https://github.com/openml/automlbenchmark/tree/master/frameworks/TunedRandomForest)

Uses the Random Forest setup as described above, but first optimizes the hyperparameter `max_features`.
It tries up to *11* different values of `max_features`. 
Five values uniformly picked from `[1, sqrt(p))`, five values from `(sqrt(p), p]` and finally `sqrt(p)`, where `p` if the number of features in the dataset.

It first evaluates `max_features=sqrt(p)` and then evaluates the other values in ascending order, until it completes them all or runs out of time.
Finally the model is fit to the entire training dataset with the best value for `max_features` according to the above cross-validation results.
