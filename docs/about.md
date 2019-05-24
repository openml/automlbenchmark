---
title: About
layout: category
sidebar_sort_order: 10
---

## Goals

We want to provide an ongoing benchmark with up-to-date results on realistic and current machine learning problems.
By making it open-source and open to contributions, we hope that all packages will be used as intended and evaluated fairly.
Fair results for each framework are enabled by allowing authors to contribute directly to the repository.
To ensure the benchmark accurately reflects the state of AutoML, evaluations will be rerun when frameworks get major updates,
and the selection of problems will be updated<sup>1</sup>.

Currently, we limit the datasets to involve single-label classification problems on i.i.d. tabular data optimizing for one of two metrics.
We would like to extend the types of tasks to include e.g. regression, multi-label classification and temporal data,
but also to include problem-specific metrics (e.g. have a false negative incur a higher cost than a false positive for a disease diagnosis problem).

## Open Science
Open science is important to us.
This is a transparent benchmark: no favorites, no cheating.
We require that all evaluated AutoML systems are open-source and all data to be freely available on [OpenML](https://www.openml.org/).
All the code required to run the benchmark is available on [Github](https://github.com/openml/automlbenchmark).

## Limitations
It is important to note that the current benchmark has some limitations.

First, we evaluate the AutoML systems by their default settings, only specifying the resources to be used (number of cores, wallclock time and memory).
We do not tune their search space or optimization hyperparameters, even though all packages allow at least some tuning.
There are of course valid reasons to tune these settings, such as only allowing a subset of models that are most interpretable.
However, in a general sense we feel that requiring tuning of AutoML frameworks defeats the purpose of AutoML, and thus opt not to do so.
That said, tuning the search space or hyperparameter values may drastically change the results.
Our hope is that authors of AutoML packages put more thought in picking good default settings, possibly dependent on the task at hand.
Over time, we hope this becomes a non-issue.

We must stress that this benchmark does *not* tell us what optimization technique is best.
For each package, the search space from which to construct a model is very different.
These differences are caused by many design differences.
These are differences in their representation of machine learning pipelines (e.g. fixed-length vs. unlimited-length), 
by the underlying machine learning packages (e.g. scikit-learn vs. WEKA), 
and even the selection of included algorithms and allowed hyperparameter values.
Finally some packages use meta-learning for warm-starting, or post-processing techniques to improve results.

There are also qualities of frameworks which are not evaluated.
Perhaps the most interesting one is the convergence rate, or how good the any-time stopping performance is of each framework along the optimization process.
But other qualities, such as ease of use or level of support can also be important to some users.


---
<sup>1</sup> Due to the high (computational) cost involved, we need to find a balance here.