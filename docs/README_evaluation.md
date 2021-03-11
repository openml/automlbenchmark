
# Installation and testing existing frameworks
1. Get the repo.

Clone the repo, switch to the **evaluation** branch, and go to the automlbenchmark folder: 
```
git clone git@github.com:qingyun-wu/automlbenchmark.git
git checkout evaluation
cd automlbenchmark
```
2. Installation.

Follow the Installation procedure in README.md. Below is a brief version
(Pre-requite: 1. a python>3.6 environment; 2. pip3)

python3 virtual environment
``` 
python3 -m venv ./venv
source venv/bin/activate
```

Install pip3
``` 
pip3 install -U pip
```

Then pip install the dependencies:
``` 
pip3 install -r requirements.txt
```

3. Quickstart.
``` 
python runbenchmark.py flaml  all 1m1c -t nomao -f 0
```
or 

``` 
python runbenchmark.py flaml  test
```

If they fail, run 

``` 
python runbenchmark.py flaml  -s only
```

# Adding your new framework (if needed)

Follow the instructions [here](https://github.com/openml/automlbenchmark/blob/master/docs/extending.md) to add a new framework or dataset.

A brief version:
1. Add a subfolder (with your framework name) in the frameworks folder
2. In your framework folder, and the following 4 files,


- Add your framework folder

```
__init__.py
exec.py
requirements.txt
setup.sh
```

3. Add your framework info in this yaml file `./resources/frameworks.yaml`

4. (optional) Add new datasets.
Find datasets info in yaml files in  `./resources/benchmarks/`
You can add new datasets or add customized yaml files

# Run experiments

## Example command line
Example command lines to run an experiment using flaml
```
python runbenchmark.py flaml_old  all 1m1c -t nomao -f 0
```

```
python runbenchmark.py flaml  all 1m1c -t nomao -f 0
```

- `flaml` is the name of the evaluated framework. 

- `all`  is the name of the yaml file (in `./resources/benchmarks/`, i.e. `./resources/benchmarks/all.yaml`) containing the dataset info. 

- `1m1c` specifies the resource constraints for experiment running (configured in `./resources/constraints.yaml`). 

- `-t nomao` specifies the dataset name of the taks (if not specified, will run all dataset in the resource yaml file sequentially)

- `-f 0 ` specifies the fold id of the dataset (if not specified, will run all 10 folds sequentially)

Note: if you are fine using existing evaluation configurations, you can just change the framework name.

## Available bash scripts

flaml_1c1h.sh

flaml_1c10m.sh

flaml_1c1m.sh

# Result aggregation

1. Results are saved in folder `./results`
2. Result aggregation

```
python get_res_avg.py 
```
Note: this script is not perfect.

3. Visualization and comparison

```
cd plot
bash run_plot_compare_flaml.sh
```

