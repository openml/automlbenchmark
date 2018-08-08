# Adding a Framework
To add your own framework you need add a folder in the `docker` directory containing two things: 
 1. A script or executable which runs the AutoML framework you want evaluated.
 
    This script/executable will be passed three parameters as input: 
    - An OpenML task ID.
    - A time limit (in seconds).
    - The number of cores that may be used.
    
    It should should produce as output:
    - The evaluation score to stdout.
    
 2. A dockerfile which sets up the environment you need for your application to work, called `Dockerfile`.
    To make this process easier, we provide a way to extend our base docker file (this makes updating all Dockerfiles easier).
   

## 1. Create a script or executable
The first requirement to have a script or executable which evaluates the predictions produced by the AutoML method.
The provided OpenML task will define which dataset is used, what evaluation method and metric, and even the data splits.
The easiest way to retrieve the relevant information is to use one of [OpenML's APIs](https://openml.github.io/OpenML/APIs/).
<!--- In the repository there are already examples for [Python], [R], and [Java]. --->
For more information on OpenML tasks, see the [official documentation](https://docs.openml.org/#tasks).

The evaluation result should be reported within the specified time limit.
The AutoML system may not use more than the specified amount of cores.

## 2. Create a Dockerfile
A dockerfile for each AutoML submission is required.
The dockerfile should define a docker image which takes three run arguments: OpenML task ID, time limit and number of cores.

To make use of our base Dockerfile, take a look at the `DockerfileTemplate` file found in the `docker` directory.
You can extend this file directly, or specify the additional setup you need to a file called `CustomDockerCode` in your AutoML folder.
After creating the `CustomDockerCode` file, run `./generate_docker.sh DIRECTORY_NAME`, 
where `DIRECTORY_NAME` is the name of the folder you created that contains the `CustomDockerCode` file.
In the `CustomDockerCode` you must specify how to call your script/executable by setting an environment variable `start_call`,
by adding a line: `ENV start_call="how_to_run"`.
Replace `"how_to_run"` by the way the evaluation script should be called, e.g. `"python3 TPOT/run_tpot.py"`.

In either case, the finished dockerfile should be placed in the AutoML's folder with the name `Dockerfile`.

## 3. Building and running an image
Before submitting a pull request, it is important to make sure the docker image works.

When building, gain access to the repository with the base image:
`docker login`

From the directory with the Dockerfile, [build](https://docs.docker.com/engine/reference/commandline/build/#options):
`docker build -t name:tag .`

Run:
`docker run name:tag <OpenML Task ID> <Runtime> <Cores> <Apikey>`

e.g.
`docker run TPOT:0.9.2 59 3600 4 abcdefghijklmnopqrstuvwxyz123456`

## 4. Submitting a pull request
