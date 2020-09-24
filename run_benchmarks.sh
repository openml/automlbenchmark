#!/bin/sh

sudo env PATH="$PATH" python runbenchmark.py mlr3automl small 1h10c
sudo env PATH="$PATH" python runbenchmark.py mlr3automl medium 1h10c
sudo env PATH="$PATH" python runbenchmark.py mlr3automl large 1h10c
sudo env PATH="$PATH" python runbenchmark.py mlr3automl regression 1h10c
