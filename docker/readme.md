# Adding a Framework
To add your own framework you need add a folder in the `docker` directory containing two things: 
 1. A script or executable which runs the AutoML framework you want evaluated.
 
    This script/executable will be passed three parameters as input (in this order): 
    - A time limit (in seconds).
    - The number of cores that may be used.
    - The metric to optimize towards.
    
    It should should produce as output:
    - The evaluation score to stdout.
    
 2. A dockerfile which sets up the environment you need for your application to work, called `Dockerfile`.
    To make this process easier, we provide a way to extend our base docker file (this makes updating all Dockerfiles easier).
    The docker container will be started with the following extra parameters (in this order):
    `<OpenML Task ID> <Fold Number> <OpenML API Key> <Runtime (s)> <Number Cores> <Metric>`.
   

## 1. Create a script or executable
The first requirement to have a script or executable.
This script or executable will:

1. Use *train* data to do any AutoML search and optimization towards some *metric*.
2. Produce predictions for *test* data.
3. Evaluate these predictions and report them to `stdout`.

Explicit limits are given to the amount of time and cores that may be used for step 1.
For step 2 and 3, we allow *some* time **TODO: Specify exactly**.

The time limit (in seconds), maximum number of cores and the metric to optimize towards are passed as parameters,
in that order (e.g. `python randomforest.py 3600 4 accuracy` if `python randomforest.py` starts your AutoML script).
For a full list of used metrics, see **TODO: Add Metric Page**.

The train and test data can be found in `/bench/common/train.arff` and `/bench/common/test.arff` respectively.

<!--- In the repository there are already examples for [Python], [R], and [Java]. --->



## 2. Create a Dockerfile
A dockerfile for each AutoML submission is required.
The dockerfile should define a docker image which takes six run arguments: 
 - OpenML task ID, time limit and number of cores.
 - The fold of the task's cross-validation splits to evaluate on.
 - An OpenML API Key which can be used to read data from the server.
 - Maximum runtime in seconds.
 - Maximum number of cores that may be used.
 - Metric to optimize towards.

To make use of our base Dockerfile, take a look at the `DockerfileTemplate` file found in the `docker` directory.
You can extend this file directly, but we recommend that you specify the additional setup you need in a file called 
`CustomDockerCode` in your AutoML folder.

After creating the `CustomDockerCode` file, run `./generate_docker.sh DIRECTORY_NAME` (from the docker folder). 
Here `DIRECTORY_NAME` is the name of the folder you created that contains the `CustomDockerCode` file.
In the `CustomDockerCode` you must specify how to call your script/executable by setting an environment variable `start_call`,
by adding a line: `ENV start_call="how_to_run"`.
Replace `"how_to_run"` by the way the evaluation script should be called *from your subdirectory*, e.g. `"python3 run_tpot.py"`.

In either case, the finished dockerfile should be placed in the AutoML's folder with the name `Dockerfile`.

## 3. Building and running an image
Before submitting a pull request, it is important to make sure the docker image works.

When building, gain access to the repository with the base image:
`docker login`

From the docker directory, [build](https://docs.docker.com/engine/reference/commandline/build/#options):
`docker build -t name:tag -f DIRECTORY_NAME/Dockerfile .`

Run:
`docker run name:tag <OpenML Task ID> <Fold> <API key> <Runtime> <Cores> <Metric>`

e.g.
`docker run TPOT:0.9.2 59 0 abcdefghijklmnopqrstuvwxyz123456 3600 4 accuracy`

## 4. Submitting a pull request
