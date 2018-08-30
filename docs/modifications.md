# Required Modifications

Each method is given, unless otherwise specified or unavailable, information about resources:
* Memory
* Runtime
* Number of cores

And, additionally:
* Metric to optimize

## auto-sklearn
**!** No number of cores specified (not supported out of the box).

### Data preprocessing
Encode string data to numeric (labelencoding).

### Non-default arguments

## Auto-WEKA
logloss metric is specified as kBInformation.

### Data preprocessing
None, ARFF file used directly.
Output is rewritten so it fits `docker/common/evaluate.py` expectations.

### Non-default arguments

## H2O AutoML

### Data preprocessing
None, ARFF file used directly.

### Non-default arguments
Depending on the metric, a different combination of `stopping_metric` and `sort_metric` are used.
This will be changed so only one metric is specified.

## hyperopt-sklearn

### Data preprocessing
Encode string data to numeric (labelencoding).

### Non-default arguments
TPE optimizer.

## oboe

### Data preprocessing
Encode string data to numeric (labelencoding).

### Non-default arguments

## TPOT

### Data preprocessing
Encode string data to numeric (labelencoding).

### Non-default arguments

