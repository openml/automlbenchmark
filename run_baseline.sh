#!/bin/sh

sudo env PATH="$PATH" python runbenchmark.py ranger small 1h10c
sudo env PATH="$PATH" python runbenchmark.py ranger medium 1h10c
sudo env PATH="$PATH" python runbenchmark.py ranger large 1h10c
sudo env PATH="$PATH" python runbenchmark.py ranger regression 1h10c
