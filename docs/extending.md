# Extending the benchmark

Whether you want to add a dataset or a framework to the benchmark, you will first have to [fork our repository](https://help.github.com/en/articles/fork-a-repo).
By forking our repository, you can make and test changes without affecting the benchmark.
If you feel your changes should be included in the benchmark, set up a [pull request](https://help.github.com/en/articles/about-pull-requests).
When creating a pull request, indicate clearly the changes and why they are made.

## Adding a dataset

### What makes a good dataset
Before discussing on *how* to add your dataset to the benchmark, we want to briefly elaborate on what we think makes for an interesting dataset.

In our benchmark we aim to include machine learning problems which are representative of those encountered in practice.
In particular, problems of different domains, mixed data types and dataset sizes.
Currently, we would love some additional *big* datasets.

Another important aspect for inclusion in the benchmark would be that it is a hard problem.
Even if the data is interesting, if a (near-)perfect model can be created with a decision tree, it is not going to be useful to profile the AutoML systems with.
Ideal datasets are those where only certain algorithms (with certain hyperparameter configurations) work, or require non-trivial data preprocessing.
Running a few different algorithms, with and without preprocessing, with different configurations, is encouraged to show the problem is sufficiently difficult.

Perhaps your dataset does not match with the above description, or you lack the resources or know-how to evaluate the problem with different machine learning approaches.
If you think the problem is interesting regardless, do not hesitate to contact us anyway.
If possible, do this through a pull request as laid out in the following sections.
Otherwise, open an [issue](https://github.com/openml/automlbenchmark/issues).
Please title the issue '[DATA ADD] DATASETNAME' (replacing 'DATASETNAME' with the name of your dataset),
provide a link to the dataset on OpenML as well as motivation as to why you think the dataset is an interesting addition.
Following the steps below will make it more likely that we'll be able to review (and add) the dataset quickly.

### Uploading to OpenML
To add a dataset to the benchmark, it needs to be uploaded to OpenML.
This requires the dataset in [ARFF format](https://www.cs.waikato.ac.nz/ml/weka/arff.html). 
Read [here](https://docs.openml.org/#data) for more information on OpenML data,
and [here](https://www.openml.org/new/data) on how to actually upload it (this requires you to [sign up](https://www.openml.org/register) for OpenML).

After uploading the dataset, visit its page on OpenML and create a [task](https://docs.openml.org/#tasks) for it.
An OpenML task specifies the evaluation procedure (e.g. splits of a 10-fold cross-validation) and the target of the problem.
To create a task for your OpenML dataset, visit its webpage and find the 'define new task' button at the bottom.
After these steps we are ready to add the problem to a benchmark.

### Testing the task
First, to make sure everything was set up right, create a single-problem benchmark.
The easiest is to modify the [example benchmark](https://github.com/openml/automlbenchmark/blob/master/resources/benchmarks/example.yaml) by replacing the iris task information with your own.
Then run the benchmark: `python runbenchmark.py constantpredictor_enc example`.
<!--- If your task contains categorical variables, make sure use `constantpredictor_enc` instead.--->

Check results for errors.
If your task fails and it is unclear why, you can open an [issue](https://github.com/openml/automlbenchmark/issues).
If you do, please clearly indicate the related OpenML task id and steps to recreate it
and title the issue '[DATA HELP] DATASETNAME', replacing 'DATASETNAME' with the name of your dataset.

### Adding it to the real thing
If you've made sure everything works, modify one of the existing benchmark or create a new one with your task.
When extending an existing benchmark, make sure not to modify any of the existing problems for the task.
Finally commit your changes and set up a pull request.
___
**Please make sure the PR does not include the changes made to `example.yaml`**
___

In your PR include:
 - a link to the task and dataset on OpenML, where the OpenML dataset has meaningful meta-data (e.g. description)
 - a motivation as to why this is an interesting addition to the benchmark. 
 Preferably address the points from the [What makes a good dataset](#What-makes-a-good-dataset) section.
 The higher quality your motivation, the better we can come to a conclusion on whether to include the dataset or not.


## Adding an AutoML framework
TODO.