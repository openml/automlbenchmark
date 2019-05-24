---
layout: category
title: Benchmark Datasets
sidebar_sort_order: 3
---

The benchmark aims to consist of datasets that represent real-world data science problems.
This means we want to include datasets of all sizes (including *big* ones), of different problem domains and with various levels of difficulty.

We also want to prevent AutoML tools from overfitting to our benchmark.
For this reason, we plan to change the selection of benchmark problems over time.
This should help prevent (some of the) bias that can be introduced by static benchmarks.

In our selection for the [paper](#paper.md), we drew datasets from [OpenML100](https://www.openml.org/s/14), [OpenML-CC18](https://www.openml.org/s/98) and [AutoML Challenges](http://automl.chalearn.org/data).
However, we did not include all datasets.
One reason was that some did not meet our criteria (more on that below), another that we wanted to keep some datasets of the future.
There are also a few datasets which we wanted to include, but could not include in the paper due to time constraints.

## Criteria
As stated before, we did not adopt all proposed datasets but made a selection.
Our criteria for adopting a dataset were as follows:

**difficulty** of the dataset has to be a sufficient.
If a problem is easily solved by just about any algorithm, it will not be able to differentiate the various AutoML frameworks.
This was the case for many of the OpenML 100 problems (see e.g. [this Github Issue](https://github.com/openml/OpenML/issues/491)),
but also some of the OpenML-CC18 problems (see e.g. [this task](https://www.openml.org/t/15)).

**representative of real-world** data science problems to be solved with the tool.
In particular we **limit artificial** problems. 
We included some, either based on their widespread use ([kr-vs-kp](https://www.openml.org/d/3)) or because they pose difficult problems.
But we do not want them to be a large part of the benchmark.
We also **limit image problems** because those problems are typically solved with solutions in the deep learning domain.
However they still make for realistic, interesting and hard problems, so we did not want to exclude them altogether.

**diversity** in the problem domains.
We do not want the benchmark to skew towards any domain in particular.
There are various software quality problems in the OpenML-CC18 (
[jm1](https://www.openml.org/d/1053),
[kc1](https://www.openml.org/d/1067), 
[kc2](https://www.openml.org/d/1063), 
[pc1](https://www.openml.org/d/1068), 
[pc3](https://www.openml.org/d/1050), 
[pc4](https://www.openml.org/d/1049)), but adopting them all would lead to a bias in the benchmark to this domain.

*We want to note however that being notified of new interesting problems in a domain that is already well-represented is still useful,
because we want to eventually replace datasets in the benchmark.*

**miscellaneous** reasons to *exclude* a dataset included label-leakage, near-duplicates (e.g. different only in categorical encoding or imputation) or violation of the i.i.d. assumption.
 


## Final List
The first iteration of our benchmark as presented in the paper contained 39 classification datasets.
For the full list of datasets and their characteristics see [OpenML Study 218](https://www.openml.org/s/218) or its [table view](https://www.openml.org/search?q=tags.tag%3Astudy_218&type=data&table=1&size=39).

## The Future
As stated before, we want the selection of benchmark problems to change over time.
If you find a good candidate dataset, you can [help us make it part of the benchmark](extending.md#adding-a-dataset).
While we are interested in all interesting datasets that match our criteria, we are particularly interested in bigger datasets (>100k rows).

We greatly appreciate any help to find new and interesting problems for the AutoML benchmark.