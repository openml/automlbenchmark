# Configuration

The AutoML benchmark has a host of settings that can be configured from a `yaml` file.
It is possible to write your own configuration file that overrides the default behavior
in a flexible manner.

## Configuration Options

The default configuration options can be found in the 
[`resources/config.yaml`](GITHUB/resources/config.yaml) file.

```{ .yaml title="resources/config.yaml" .limit_max_height }
--8<-- "resources/config.yaml"
```

## Custom Configurations
To override default configuration, create your custom `config.yaml` file under the
`user_dir` (specified by the `--userdir` parameter of `runbenchmark.py`).
The application will automatically load this custom file and apply it on top of the defaults.

When specifying filepaths, configurations support the following placeholders:

| Placeholder | Replaced By Value Of | Default                     | Function                                                               |
|:------------|:---------------------|:----------------------------|:-----------------------------------------------------------------------|
| `{input}`   | `input_dir`          | `~/.openml/cache`           | Folder from which datasets are loaded (and/or downloaded)              |
| `{output}`  | `output_dir`         | `./results`                 | Folder where all outputs (results, logs, predictions, ...) are stored. |
| `{user}`    | `user_dir`           | `~/.config/automlbenchmark` | Folder containing custom configuration files.                          |
| `{root}`    | `root_dir`           | Detected at runtime         | The root folder of the `automlbenchmark` application.                  |

For example, including the following snippet in your custom configuration when
`user_dir` is `~/.config/automlbenchmark` (which it is by default) changes your 
input directory to `~/.config/automlbenchmark/data` :

```yaml title="examples/custom/config.yaml"
--8<-- "examples/custom/config.yaml:6:7"
```

!!! tip "Multiple Configuration Files"
    It is possible to have multiple configuration files: 
    just create a folder for each `config.yaml` file and use that folder as your 
    `user_dir` using `--userdir /path/to/config/folder` when invoking `runbenchmark.py`.


Below is an example of a configuration file which **1.** changes the directory the 
datasets are loaded from, **2.** specifies additional paths to look up framework,
benchmark, and constraint definitions, **3.** also makes those available in an S3 bucket 
when running in AWS mode, and **4.** changes the default EC2 instance type for AWS mode.

```yaml title="examples/custom/config.yaml"
--8<-- "examples/custom/config.yaml:3"
```
