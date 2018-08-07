# Adding a Framework
To add your own framework you need two things: 
 1. A script or executable which runs the AutoML framework you want evaluated.
 
    This script should take as input: 
    - An OpenML task ID.
    - A time limit (in seconds).
    - The number of cores that may be used.
    
    The script should produce as output:
    - The evaluation score.
    
 2. A dockerfile which sets up the environment you need for your application to work.
    To make this process easy, we provide a docker template which can easily be extended. 

## 1. Creating a script
The first requirement to have a script which evaluates the predictions produced by the AutoML method.
The provided OpenML task will define which dataset is used, what evaluation method and metric, and even the data splits.
The easiest way to retrieve the relevant information is to use one of [OpenML's APIs](https://openml.github.io/OpenML/APIs/).
<!--- In the repository there are already examples for [Python], [R], and [Java]. --->
For more information on OpenML tasks, see the [official documentation](https://docs.openml.org/#tasks).

## 2. Building an image

Log in to gain access to the repository with the base image:
`docker login`

Verify that automlbenchmark\docker is the current working directory:
`pwd`

docker build -t tagname .

## 3. Running an image
Before submitting a pull request, it is important to test the docker image works.

docker run image_name <OpenML Task ID> <Runtime> <Cores>

## 4. Submitting a pull request