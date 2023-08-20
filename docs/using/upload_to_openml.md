
### Uploading results to OpenML
The `upload_results.py` script can be used to upload results to OpenML with the following usage:
```text
>python upload_results.py --help
usage: Script to upload results from the benchmark to OpenML. [-h] [-i INPUT_DIRECTORY] [-a APIKEY] [-m MODE] [-x] [-v] [-t TASK]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIRECTORY, --input-directory INPUT_DIRECTORY
                        Directory that stores results from the runbenchmark.py invocation. By default use the most recent folder in the results folder as
                        specified in the configuration.
  -a APIKEY, --api-key APIKEY
                        OpenML API key to use for uploading results.
  -m MODE, --mode MODE  Run mode (default=check).
                        • check: only report whether results can be uploaded.
                        • upload: upload all complete results.
  -x, --fail-fast       Stop as soon as a task fails to upload due to an error during uploading.
  -v, --verbose         Output progress to console.
  -t TASK, --task TASK  Only upload results for this specific task.
```

Note that the default behavior does not upload data but only verifies data is complete.
We strongly encourage you to only upload your data after verifying all expected results are complete.
The OpenML Python package is used for uploading results, so to ensure your API credentials are configured, please refer to their [configuration documentation](https://openml.github.io/openml-python/master/usage.html#installation-set-up).
Results obtained on tasks on the test server (e.g. through the `--test-server` option of `runbenchmark.py`) are uploaded to the test server and don't require additional authentication.
