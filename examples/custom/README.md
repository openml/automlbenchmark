From the automlbenchmark app, you can now directly run additional frameworks (here various versions of H2O-3) and benchmarks (here `h2obench`):
```bash
python runbenchmark.py randomforest filedatasets -u examples/custom
python runbenchmark.py gradientboosting myvalidation 30m4c2f -t miniboone -u examples/custom
```

**Note:**
 you can also copy those files in automlbenchmark user config directory `~/.config/automlbenchmark` to be able to use them without having to specify the `-u examples/custom` argument.
