#!/bin/sh

sudo env PATH="$PATH" python runbenchmark.py constantpredictor small 1h10c
sudo env PATH="$PATH" python runbenchmark.py constantpredictor medium 1h10c
sudo env PATH="$PATH" python runbenchmark.py constantpredictor large 1h10c
sudo env PATH="$PATH" python runbenchmark.py constantpredictor regression 1h10c
