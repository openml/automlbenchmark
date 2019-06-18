Copy those files in automlbenchmark user config directory: ~/.config/automlbenchmark

From the automlbenchmark app, you can now directly run additional frameworks (here various versions of H2O-3) and benchmarks (here `h2obench`):
```bash
python runbenchmark.py H2OAutoML_nightly h2obench -t miniboone -m aws 2
```
